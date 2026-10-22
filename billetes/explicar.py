"""Explicabilidad: ¿en qué parte de la imagen se fija cada modelo?

- CNN y MobileNetV3: Grad-CAM (Selvaraju et al., 2017) sobre el último bloque convolucional.
- ViT: Attention Rollout (Abnar & Zuidema, 2020) sobre las 12 capas de atención.
Genera runs/explicabilidad.png con una fila por imagen de test.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import CLASS_NAMES, CLASSES, MODELOS, NOMBRES, t
from .datos import BilletesDataset, construir_transform
from .modelos import cargar_checkpoint


def gradcam(model, capa, x, clase=None):
    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h1 = capa.register_forward_hook(fwd)
    model.zero_grad()
    logits = model(x)
    c = int(logits.argmax(1)) if clase is None else clase
    logits[0, c].backward()
    h1.remove()
    a, g = acts["a"][0], grads["g"][0]
    cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), logits.softmax(1)[0].detach().numpy()


def attention_rollout(model, x):
    bloques = model.backbone.blocks
    atts = []
    hooks = []
    for b in bloques:
        b.attn.fused_attn = False
        hooks.append(b.attn.attn_drop.register_forward_hook(lambda m, i, o: atts.append(i[0].detach())))
    with torch.no_grad():
        probs = model(x).softmax(1)[0].numpy()
    for h in hooks:
        h.remove()
    n = atts[0].shape[-1]
    r = torch.eye(n)
    for a in atts:
        a = a[0].mean(0)
        a = a + torch.eye(n)
        a = a / a.sum(-1, keepdim=True)
        r = a @ r
    cls = r[0, 1:]
    lado = int(len(cls) ** 0.5)
    mapa = cls.reshape(lado, lado).numpy()
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
    return mapa, probs


def explicar(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    modelos = {}
    for n in MODELOS:
        if (out / f"{n}.pt").exists():
            m, _ = cargar_checkpoint(out / f"{n}.pt", "cpu")
            modelos[n] = m
    if not modelos:
        raise SystemExit("No hay modelos entrenados.")
    ds = BilletesDataset(args.data, "test")
    rng = random.Random(7)
    idx = []
    for c in range(len(CLASSES)):
        cands = [i for i, t in enumerate(ds.targets) if t == c]
        idx += rng.sample(cands, min(len(cands), max(1, args.n // len(CLASSES) + (c < args.n % len(CLASSES)))))
    idx = idx[:args.n]

    # filas = imagen original + un mapa por modelo; columnas = imágenes de test
    filas = 1 + len(modelos)
    fig, axes = plt.subplots(filas, len(idx), figsize=(2.6 * len(idx), 2.75 * filas))
    axes = np.array(axes).reshape(filas, len(idx))
    for c, i in enumerate(idx):
        with Image.open(ds.root / ds.filas[i]["ruta"]) as im:
            img = im.convert("RGB")
        vis = construir_transform(224, (0, 0, 0), (1, 1, 1), False)(img).permute(1, 2, 0).numpy()
        axes[0, c].imshow(vis)
        axes[0, c].set_title(t("Real: ", "True: ") + CLASS_NAMES[ds.targets[i]], fontsize=10)
        for r, (n, m) in enumerate(modelos.items(), start=1):
            s = m.spec
            x = construir_transform(s["img_size"], s["mean"], s["std"], False)(img).unsqueeze(0)
            if n == "vit":
                mapa, probs = attention_rollout(m, x)
            else:
                x.requires_grad_(True)
                mapa, probs = gradcam(m, m.capa_gradcam(), x)
            mapa = np.array(Image.fromarray((mapa * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)) / 255
            j = int(probs.argmax())
            axes[r, c].imshow(vis)
            axes[r, c].imshow(mapa, cmap="jet", alpha=0.45)
            ok = "✓" if j == ds.targets[i] else "✗"
            axes[r, c].set_title(f"{ok} {CLASS_NAMES[j]} ({probs[j]:.0%})", fontsize=10)
            if c == 0:
                axes[r, c].text(-0.08, 0.5, NOMBRES[n], transform=axes[r, c].transAxes, rotation=90,
                                ha="right", va="center", fontsize=10)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle(t("Grad-CAM (CNN, MobileNetV3) y Attention Rollout (ViT) en el test independiente", "Grad-CAM (CNN, MobileNetV3) and attention rollout (ViT) on the independent test set"), fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(out / t("explicabilidad.png", "explainability.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", out / t("explicabilidad.png", "explainability.png"))
