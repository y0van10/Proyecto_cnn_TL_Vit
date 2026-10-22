"""Figuras del dataset para la exposición y el artículo.

python herramientas/figuras_dataset.py --data data --out runs
  -> dataset_muestras.png      ejemplos de train / val / test por clase
  -> dataset_distribucion.png  imágenes por clase y partición
  -> aumentos.png              ejemplos de aumento de datos
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from billetes import CLASS_NAMES, CLASSES, t  # noqa: E402
from billetes.datos import BilletesDataset, construir_transform  # noqa: E402

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plt_estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150})
    return plt


def cuadrado(img, lado=220):
    return construir_transform(lado, (0, 0, 0), (1, 1, 1), False)(img).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = plt_estilo()
    rng = random.Random(3)

    # 1. muestras
    splits = [("train", t("Entrenamiento", "Training")), ("val", t("Validación", "Validation")),
              ("test", t("Prueba", "Test"))]
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.6))
    for r, (s, titulo) in enumerate(splits):
        ds = BilletesDataset(a.data, s)
        for c in range(4):
            idx = rng.sample([i for i, t in enumerate(ds.targets) if t == c], 2)
            for k, i in enumerate(idx):
                ax = axes[r, c * 2 + k]
                with Image.open(ds.root / ds.filas[i]["ruta"]) as im:
                    ax.imshow(cuadrado(im.convert("RGB")))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(CLASS_NAMES[c], fontsize=11)
        axes[r, 0].set_ylabel(titulo, fontsize=11, color=TINTA)
    fig.suptitle(t("Muestras del dataset: entrenamiento (fotos planas) vs validación y prueba (billete en mano)",
                   "Dataset samples: training (flat photos) vs validation and test (note held in hand)"), fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "dataset_muestras.png", bbox_inches="tight")
    plt.close(fig)

    # 2. distribución
    man = json.loads((Path(a.data) / "manifest.json").read_text(encoding="utf8"))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(4)
    cols = {"train": AZUL, "val": NARANJA, "test": AQUA}
    nombres = {"train": "train", "val": "val", "test": "test"}
    for k, s in enumerate(["train", "val", "test"]):
        v = [man["conteos"][s][c] for c in CLASSES]
        b = ax.bar(x + (k - 1) * 0.27, v, 0.25, color=cols[s], label=f"{nombres[s]} (n={sum(v)})")
        for bi, vi in zip(b, v):
            ax.text(bi.get_x() + bi.get_width() / 2, vi + 15, str(vi), ha="center", fontsize=8, color=TINTA2)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel(t("imágenes", "images"))
    ax.set_title(t("Distribución por clase y partición", "Images per class and split"), color=TINTA)
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dataset_distribucion.png", bbox_inches="tight")
    plt.close(fig)

    # 3. aumentos
    ds = BilletesDataset(a.data, "train")
    tf = construir_transform(224, (0, 0, 0), (1, 1, 1), True)
    i = rng.choice([j for j, t in enumerate(ds.targets) if t == 2])
    with Image.open(ds.root / ds.filas[i]["ruta"]) as im:
        img = im.convert("RGB")
    fig, axes = plt.subplots(1, 7, figsize=(15, 2.6))
    axes[0].imshow(cuadrado(img))
    axes[0].set_title("original", fontsize=10)
    random.seed(5)
    import torch
    torch.manual_seed(5)
    for ax in axes[1:]:
        ax.imshow(tf(img).permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(t("aumentada", "augmented"), fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(t("Aumento de datos: rotación 90°/180°, recorte aleatorio, perspectiva, color, desenfoque y borrado",
                   "Data augmentation: 90°/180° rotation, random crop, perspective, colour, blur and erasing"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "aumentos.png", bbox_inches="tight")
    plt.close(fig)
    print("Figuras guardadas en", out)


if __name__ == "__main__":
    main()
