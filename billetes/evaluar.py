"""Evaluación y comparación de los tres modelos.

Para cada checkpoint en runs/ calcula, por partición (val / test):
  exactitud, exactitud balanceada, precisión / recall / F1 macro, F1 por clase,
  matriz de confusión, ECE (calibración) y cobertura con umbral de confianza.
Además mide parámetros, GFLOPs, tamaño del archivo y latencia en CPU (lote 1).
Con --robustez evalúa el test con perturbaciones típicas de una cámara web.
Al final escribe comparacion.csv / .md, gráficos y seleccion.json.
"""
from __future__ import annotations

import csv
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from . import CLASS_NAMES, CLASSES, NOMBRES, t
from .datos import BilletesDataset, construir_transform
from .entrenar import predecir_loader
from .modelos import cargar_checkpoint, contar_parametros

# Paleta categórica validada (CVD-safe para 3 series) + tinta neutra
COLORES = {"cnn": "#2a78d6", "mobilenet": "#eb6834", "vit": "#1baf7a"}
TINTA, TINTA2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- métricas
def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metricas_completas(y, probs, umbral=0.7):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    conf = probs.max(1)
    acept = conf >= umbral
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
        "f1_por_clase": {CLASS_NAMES[i]: float(f[i]) for i in labels},
        "recall_por_clase": {CLASS_NAMES[i]: float(r[i]) for i in labels},
        "soporte": {CLASS_NAMES[i]: int(s[i]) for i in labels},
        "ece": ece(probs, y),
        "nll": float(-np.log(probs[np.arange(len(y)), y] + 1e-9).mean()),
        "cobertura_umbral": float(acept.mean()),
        "accuracy_aceptadas": float((pred[acept] == y[acept]).mean()) if acept.any() else float("nan"),
        "matriz": confusion_matrix(y, pred, labels=labels).tolist(),
        "n": int(len(y)),
    }


def gflops(model, size):
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.zeros(1, 3, size, size)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model.cpu()(x)
        return fc.get_total_flops() / 1e9
    except Exception:
        return float("nan")


@torch.inference_mode()
def latencia_ms(model, size, device="cpu", repeticiones=30):
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(5):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeticiones):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / repeticiones


def cpu_nombre():
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# --------------------------------------------------------------------------- perturbaciones
def _jpeg(img, q=15):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _ruido(img, sigma=20):
    a = np.asarray(img).astype(np.float32)
    a += np.random.default_rng(0).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _balance(img):
    a = np.asarray(img).astype(np.float32) * np.array([0.80, 1.0, 1.20])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _original(img):
    return img


def _oscuro(img):
    return ImageEnhance.Brightness(img).enhance(0.45)


def _sobreexpuesto(img):
    return ImageEnhance.Brightness(img).enhance(1.6)


def _desenfoque(img):
    return img.filter(ImageFilter.GaussianBlur(2.5))


def _rotado(img):
    return img.rotate(90, expand=True)


PERT_EN = {"original": "original", "oscuro (x0.45)": "dark (x0.45)", "sobreexpuesto (x1.6)": "overexposed (x1.6)",
           "desenfoque (r=2.5)": "blur (r=2.5)", "ruido (σ=20)": "noise (σ=20)", "JPEG q=15": "JPEG q=15",
           "balance de blancos": "white balance", "rotado 90°": "rotated 90°"}


# funciones con nombre (no lambdas) para que funcionen con num_workers>0 en Windows
PERTURBACIONES = {
    "original": _original,
    "oscuro (x0.45)": _oscuro,
    "sobreexpuesto (x1.6)": _sobreexpuesto,
    "desenfoque (r=2.5)": _desenfoque,
    "ruido (σ=20)": _ruido,
    "JPEG q=15": _jpeg,
    "balance de blancos": _balance,
    "rotado 90°": _rotado,
}


class _Perturbado(torch.utils.data.Dataset):
    def __init__(self, base: BilletesDataset, fn, tf):
        self.base, self.fn, self.tf = base, fn, tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        with Image.open(self.base.root / self.base.filas[i]["ruta"]) as im:
            img = im.convert("RGB")
        return self.tf(self.fn(img)), self.base.targets[i]


# --------------------------------------------------------------------------- gráficos
def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
        "ytick.color": TINTA2, "axes.titlecolor": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    })
    return plt


def graficar_confusiones(res, split, out):
    plt = _estilo()
    nombres = [n for n in res if split in res[n]]
    if not nombres:
        return
    fig, axes = plt.subplots(1, len(nombres), figsize=(4.2 * len(nombres), 4.0))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, nombres):
        cm = np.array(res[n][split]["matriz"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else TINTA)
        ax.set_xticks(range(4), CLASS_NAMES)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel(t("Predicción", "Predicted"))
        ax.set_ylabel(t("Real", "True"))
        ax.set_title(f"{NOMBRES[n]}\nF1 macro = {res[n][split]['f1_macro']:.3f}", fontsize=10)
    fig.suptitle(t(f"Matrices de confusión – {split}", f"Confusion matrices – {split}"), color=TINTA)
    fig.tight_layout()
    fig.savefig(out / f"confusion_{split}.png", bbox_inches="tight")
    plt.close(fig)


def graficar_comparacion(filas, out):
    plt = _estilo()
    nombres = [f["modelo"] for f in filas]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    metricas = [("val_f1_macro", t("F1 macro (validación)", "Macro-F1 (validation)")),
                ("test_f1_macro", t("F1 macro (test independiente)", "Macro-F1 (independent test)")),
                ("latencia_cpu_ms", t("Latencia CPU, lote 1 (ms) – menor es mejor",
                                      "CPU latency, batch 1 (ms) – lower is better"))]
    for ax, (k, titulo) in zip(axes, metricas):
        vals = [f.get(k, np.nan) for f in filas]
        bars = ax.bar(range(len(nombres)), vals, color=[COLORES[n] for n in nombres], width=0.6,
                      edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if v < 2 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color=TINTA)
        ax.set_xticks(range(len(nombres)), [NOMBRES[n] for n in nombres], fontsize=8.5)
        ax.set_title(titulo, fontsize=10)
        if "f1" in k:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "comparacion_modelos.png", bbox_inches="tight")
    plt.close(fig)


def graficar_robustez(rob, out):
    """Gráfico de puntos: F1 macro por perturbación (eje desde 0.7 para ver las diferencias)."""
    plt = _estilo()
    nombres = list(rob)
    perts = list(next(iter(rob.values())))
    x = np.arange(len(perts))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    marcadores = {"cnn": "o", "mobilenet": "s", "vit": "D"}
    for k, n in enumerate(nombres):
        dx = (k - (len(nombres) - 1) / 2) * 0.18
        v = [rob[n][p]["f1_macro"] for p in perts]
        ax.scatter(x + dx, v, s=70, color=COLORES[n], marker=marcadores.get(n, "o"), label=NOMBRES[n],
                   edgecolor="white", linewidth=1.2, zorder=3)
    for xi in x:
        ax.axvline(xi, color=GRID, lw=0.8, zorder=1)
    etiquetas = [PERT_EN.get(p, p) if t("es", "en") == "en" else p for p in perts]
    ax.set_xticks(x, etiquetas, rotation=15, fontsize=9)
    lo = min(rob[n][p]["f1_macro"] for n in nombres for p in perts)
    ax.set_ylim(max(0.0, np.floor(lo * 20) / 20 - 0.05), 1.0)
    ax.set_ylabel(t("F1 macro (test)", "Macro-F1 (test)"))
    ax.set_title(t("Robustez ante perturbaciones típicas de una cámara web",
                   "Robustness to typical webcam perturbations"))
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=len(nombres), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out / "robustez.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- principal
def evaluar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    out = Path(args.out)
    filas, res, rob = [], {}, {}
    for nombre in args.models:
        path = out / f"{nombre}.pt"
        if not path.exists():
            print(f"(sin checkpoint) {path}")
            continue
        model, ck = cargar_checkpoint(path, device)
        size = model.spec["img_size"]
        cl = nombre != "vit"
        if cl:
            model = model.to(memory_format=torch.channels_last)
        fila = {"modelo": nombre, "nombre": NOMBRES[nombre], "parametros_M": contar_parametros(model) / 1e6,
                "gflops": gflops(model, size), "tamano_MB": path.stat().st_size / 1e6, "img_size": size,
                "epoca_mejor": ck.get("epoca")}
        fila["latencia_cpu_ms"] = latencia_ms(model, size, "cpu")
        fila["fps_cpu"] = 1000 / fila["latencia_cpu_ms"]
        if device == "cuda":
            fila["latencia_gpu_ms"] = latencia_ms(model, size, "cuda")
        model.to(device)
        res[nombre] = {}
        for split in args.split:
            ds = BilletesDataset(args.data, split, construir_transform(size, model.spec["mean"], model.spec["std"],
                                                                       False))
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
            t = time.time()
            probs, y = predecir_loader(model, dl, device, channels_last=cl)
            m = metricas_completas(y, probs, args.threshold)
            res[nombre][split] = m
            for k in ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "ece",
                      "cobertura_umbral", "accuracy_aceptadas"):
                fila[f"{split}_{k}"] = m[k]
            print(f"{NOMBRES[nombre]:<24} {split:<5} acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f} "
                  f"ECE={m['ece']:.3f} ({time.time() - t:.0f}s)", flush=True)
            with open(out / f"pred_{nombre}_{split}.csv", "w", newline="", encoding="utf8") as f:
                w = csv.writer(f)
                w.writerow(["ruta", "real", "pred", "confianza"] + [f"p_{c}" for c in CLASSES])
                for fr, yi, pi in zip(ds.filas, y, probs):
                    w.writerow([fr["ruta"], CLASSES[yi], CLASSES[pi.argmax()], f"{pi.max():.4f}"]
                               + [f"{v:.4f}" for v in pi])
        if args.robustez and "test" in args.split:
            rob[nombre] = {}
            base = BilletesDataset(args.data, "test")
            tf = construir_transform(size, model.spec["mean"], model.spec["std"], False)
            for pn, fn in PERTURBACIONES.items():
                dl = torch.utils.data.DataLoader(_Perturbado(base, fn, tf), batch_size=args.batch,
                                                 num_workers=args.workers)
                probs, y = predecir_loader(model, dl, device, channels_last=cl)
                m = metricas_completas(y, probs, args.threshold)
                rob[nombre][pn] = {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
                print(f"   robustez {pn:<22} acc={m['accuracy']:.3f} F1={m['f1_macro']:.3f}", flush=True)
        filas.append(fila)
        del model

    if not filas:
        raise SystemExit("No hay checkpoints para evaluar.")
    entorno = {"cpu": cpu_nombre(), "hilos": torch.get_num_threads(), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    (out / "metricas_detalle.json").write_text(json.dumps({"entorno": entorno, "resultados": res,
                                                           "robustez": rob, "umbral": args.threshold},
                                                          indent=2, ensure_ascii=False), encoding="utf8")
    claves = list(dict.fromkeys(k for f in filas for k in f))
    with open(out / "comparacion.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=claves)
        w.writeheader()
        w.writerows(filas)

    # --- selección del modelo para la aplicación final ---------------------------
    # Se selecciona con VALIDACIÓN (no con test) para que el test siga siendo una estimación
    # no sesgada del desempeño final.
    split_sel = "val" if all("val_f1_macro" in f for f in filas) else args.split[0]
    mejor = max(f[f"{split_sel}_f1_macro"] for f in filas)
    candidatos = [f for f in filas if f[f"{split_sel}_f1_macro"] >= mejor - 0.01]
    elegido = min(candidatos, key=lambda f: f["latencia_cpu_ms"])
    sel = {"modelo": elegido["modelo"], "nombre": elegido["nombre"],
           "criterio": (f"Mayor F1 macro en {split_sel}; entre modelos a <=1 punto del mejor se elige el de "
                        "menor latencia en CPU (la app corre en tiempo real)."),
           "f1_macro": elegido[f"{split_sel}_f1_macro"], "latencia_cpu_ms": elegido["latencia_cpu_ms"],
           "f1_macro_test": elegido.get("test_f1_macro"),
           "entorno": entorno}
    (out / "seleccion.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf8")

    # --- tabla markdown -----------------------------------------------------------
    cols = [("Modelo", lambda f: f["nombre"]), ("Parám. (M)", lambda f: f"{f['parametros_M']:.2f}"),
            ("GFLOPs", lambda f: f"{f['gflops']:.2f}"), ("Tamaño (MB)", lambda f: f"{f['tamano_MB']:.1f}"),
            ("Latencia CPU (ms)", lambda f: f"{f['latencia_cpu_ms']:.1f}")]
    for s in args.split:
        cols += [(f"Acc {s}", lambda f, s=s: f"{f[f'{s}_accuracy']:.4f}"),
                 (f"F1 macro {s}", lambda f, s=s: f"{f[f'{s}_f1_macro']:.4f}"),
                 (f"ECE {s}", lambda f, s=s: f"{f[f'{s}_ece']:.3f}")]
    md = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fn(f) for _, fn in cols) + " |" for f in filas]
    md += ["", f"Entorno: {entorno['cpu']} ({entorno['hilos']} hilos), PyTorch {entorno['torch']}",
           f"Modelo seleccionado: **{sel['nombre']}** — {sel['criterio']}"]
    if rob:
        perts = list(PERTURBACIONES)
        md += ["", "Robustez (F1 macro en test):", "",
               "| Perturbación | " + " | ".join(NOMBRES[n] for n in rob) + " |", "|---|" + "---|" * len(rob)]
        md += [f"| {p} | " + " | ".join(f"{rob[n][p]['f1_macro']:.3f}" for n in rob) + " |" for p in perts]
    (out / "comparacion.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n" + "\n".join(md))

    for s in args.split:
        graficar_confusiones(res, s, out)
    graficar_comparacion(filas, out)
    if rob:
        graficar_robustez(rob, out)
    print(f"\nModelo seleccionado para la app: {sel['nombre']}  ->  {out / 'seleccion.json'}")
