"""Entrenamiento por fases con early stopping.

cnn        1 fase : end-to-end, todos los pesos desde cero.
mobilenet  2 fases: (1) extracción de características -> solo capas finales nuevas
                    (2) ajuste fino de los últimos bloques + capas finales
vit        2 fases: (1) solo cabeza lineal (linear probing)
                    (2) ajuste fino de los últimos bloques Transformer + cabeza
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import CLASSES, NOMBRES, t
from .datos import crear_loader, pesos_clase
from .modelos import contar_parametros, crear_modelo

# Configuración por defecto (pensada para GPU o CPU modesta; todo se puede cambiar por CLI)
FASES = {
    "cnn": [dict(nombre="end-to-end", epocas=35, lr=2e-3, descongelar=-1, wd=5e-4)],
    "mobilenet": [dict(nombre="capas-finales", epocas=3, lr=1e-3, descongelar=0, wd=1e-4),
                  dict(nombre="fine-tuning", epocas=10, lr=1e-3, lr_backbone=1e-4, descongelar=3, wd=1e-4)],
    "vit": [dict(nombre="cabeza", epocas=2, lr=2e-3, descongelar=0, wd=1e-4),
            dict(nombre="fine-tuning", epocas=5, lr=1e-3, lr_backbone=3e-5, descongelar=4, wd=0.05)],
}


def fijar_semilla(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metricas(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score

    return {"acc": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(CLASSES))),
                                       zero_division=0))}


@torch.inference_mode()
def predecir_loader(model, loader, device, max_batches=None, channels_last=False):
    model.eval()
    probs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=device == "cuda"):
            logits = model(x)
        probs.append(logits.float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def _scheduler(opt, pasos_totales, warmup):
    def f(paso):
        if paso < warmup:
            return (paso + 1) / warmup
        prog = (paso - warmup) / max(1, pasos_totales - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def entrenar(args):
    fijar_semilla(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                      f" ({torch.get_num_threads()} hilos)"))

    model = crear_modelo(args.model, preentrenado=True, pesos=args.weights)
    if args.img_size:
        model.spec["img_size"] = args.img_size
    spec = model.spec
    channels_last = args.model != "vit"
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_dl, train_ds = crear_loader(args.data, "train", spec["img_size"], spec["mean"], spec["std"],
                                      args.batch, True, args.workers)
    val_dl, val_ds = crear_loader(args.data, "val", spec["img_size"], spec["mean"], spec["std"],
                                  args.batch * 2, False, args.workers)
    w = torch.tensor(pesos_clase(train_ds.targets), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    print(f"{NOMBRES[args.model]}: {contar_parametros(model) / 1e6:.2f} M parámetros | "
          f"train={len(train_ds)} val={len(val_ds)} | img={spec['img_size']} | pesos clase={w.cpu().numpy().round(2)}")

    fases = [dict(f) for f in FASES[args.model]]
    if args.epochs is not None:
        fases[-1]["epocas"] = args.epochs
    if args.head_epochs is not None and len(fases) > 1:
        fases[0]["epocas"] = args.head_epochs
    if args.unfreeze is not None and len(fases) > 1:
        fases[-1]["descongelar"] = args.unfreeze
    if args.lr is not None:
        fases[-1]["lr"] = args.lr

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    historial, mejor_f1, epoca_global, sin_mejora = [], -1.0, 0, 0
    t0 = time.time()

    for fase in fases:
        if fase["epocas"] <= 0:
            continue
        if hasattr(model, "descongelar_ultimos"):
            model.descongelar_ultimos(fase["descongelar"])
        grupos = model.grupos_parametros()
        cabeza = [p for p in grupos["cabeza"] if p.requires_grad]
        backbone = [p for p in grupos["backbone"] if p.requires_grad]
        param_groups = [{"params": cabeza, "lr": fase["lr"]}]
        if backbone:
            param_groups.append({"params": backbone, "lr": fase.get("lr_backbone", fase["lr"])})
        opt = torch.optim.AdamW(param_groups, weight_decay=fase["wd"])
        pasos = fase["epocas"] * (min(len(train_dl), args.max_batches or 10**9))
        sched = _scheduler(opt, pasos, warmup=max(1, int(0.05 * pasos)))
        print(f"\n=== Fase '{fase['nombre']}': {fase['epocas']} épocas | entrenables "
              f"{contar_parametros(model, True) / 1e6:.2f} M | lr cabeza={fase['lr']:.0e}"
              + (f" backbone={fase.get('lr_backbone', fase['lr']):.0e}" if backbone else ""))
        sin_mejora = 0
        for ep in range(fase["epocas"]):
            epoca_global += 1
            model.train()
            t_ep, perdida, aciertos, vistos = time.time(), 0.0, 0, 0
            for i, (x, y) in enumerate(train_dl):
                if args.max_batches and i >= args.max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    logits = model(x)
                    loss = loss_fn(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                perdida += loss.item() * len(y)
                aciertos += (logits.argmax(1) == y).sum().item()
                vistos += len(y)
                if args.verbose and i % 20 == 0:
                    print(f"   lote {i}/{len(train_dl)} loss={loss.item():.3f}", flush=True)
            probs, yv = predecir_loader(model, val_dl, device, args.max_batches, channels_last)
            val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(probs) + 1e-9),
                                                    torch.tensor(yv)).item())
            mv = metricas(yv, probs.argmax(1))
            fila = {"epoca": epoca_global, "fase": fase["nombre"], "train_loss": perdida / vistos,
                    "train_acc": aciertos / vistos, "val_loss": val_loss, "val_acc": mv["acc"],
                    "val_f1_macro": mv["f1_macro"], "lr": opt.param_groups[0]["lr"],
                    "segundos": time.time() - t_ep}
            historial.append(fila)
            mejora = mv["f1_macro"] > mejor_f1
            print(f"[{args.model}] ep {epoca_global:02d} {fase['nombre']:<13} loss {fila['train_loss']:.3f} "
                  f"acc {fila['train_acc']:.3f} | val loss {val_loss:.3f} acc {mv['acc']:.3f} "
                  f"F1 {mv['f1_macro']:.3f} | {fila['segundos']:.0f}s" + ("  *mejor*" if mejora else ""),
                  flush=True)
            if mejora:
                mejor_f1, sin_mejora = mv["f1_macro"], 0
                sd = {k: (v.half() if args.half and v.is_floating_point() else v).cpu()
                      for k, v in model.state_dict().items()}
                torch.save({"state_dict": sd, "arquitectura": args.model, "clases": CLASSES,
                            "img_size": spec["img_size"], "mean": list(spec["mean"]), "std": list(spec["std"]),
                            "epoca": epoca_global, "fase": fase["nombre"], "val": mv,
                            "fases": fases, "seed": args.seed}, out / f"{args.model}.pt")
            else:
                sin_mejora += 1
            with open(out / f"{args.model}_historial.csv", "w", newline="", encoding="utf8") as f:
                wr = csv.DictWriter(f, fieldnames=list(historial[0]))
                wr.writeheader()
                wr.writerows(historial)
            if args.patience and sin_mejora >= args.patience and fase is fases[-1]:
                print(f"Early stopping: {args.patience} épocas sin mejorar F1 de validación")
                break

    resumen = {"modelo": args.model, "mejor_val_f1_macro": mejor_f1, "epocas": epoca_global,
               "minutos": (time.time() - t0) / 60, "dispositivo": device,
               "parametros": contar_parametros(model), "img_size": spec["img_size"], "fases": fases}
    (out / f"{args.model}_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                     encoding="utf8")
    graficar_historial(out, args.model)
    print(f"\nListo en {resumen['minutos']:.1f} min. Mejor F1 val = {mejor_f1:.4f} -> {out / (args.model + '.pt')}")


def graficar_historial(out: Path, nombre: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(out) / f"{nombre}_historial.csv"
    if not path.exists():
        return
    with open(path, encoding="utf8") as f:
        h = list(csv.DictReader(f))
    ep = [int(r["epoca"]) for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [float(r["train_loss"]) for r in h], "o-", label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in h], "o-", label="val")
    ax[0].set_title(f"{NOMBRES[nombre]} – " + t("pérdida", "loss"))
    ax[1].plot(ep, [float(r["train_acc"]) for r in h], "o-", label="train acc")
    ax[1].plot(ep, [float(r["val_acc"]) for r in h], "o-", label="val acc")
    ax[1].plot(ep, [float(r["val_f1_macro"]) for r in h], "s--", label=t("val F1 macro", "val macro-F1"))
    ax[1].set_title(f"{NOMBRES[nombre]} – " + t("exactitud / F1", "accuracy / F1"))
    cambios = [ep[i] - 0.5 for i in range(1, len(h)) if h[i]["fase"] != h[i - 1]["fase"]]
    for a in ax:
        for c in cambios:
            a.axvline(c, color="gray", ls=":", lw=1)
        a.set_xlabel(t("época", "epoch"))
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(Path(out) / f"{nombre}_curvas.png", dpi=150)
    plt.close(fig)
