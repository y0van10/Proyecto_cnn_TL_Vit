"""Predictor reutilizable (CLI `predict` y app de cámara web)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import CLASS_NAMES, MODELOS, NOMBRES
from .datos import EXTS, construir_transform
from .modelos import cargar_checkpoint


def modelo_seleccionado(runs: str | Path) -> str | None:
    p = Path(runs) / "seleccion.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf8"))["modelo"]
    return None


class Predictor:
    """Carga un checkpoint y clasifica imágenes PIL o arreglos RGB (H, W, 3)."""

    def __init__(self, path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ck = cargar_checkpoint(path, self.device)
        self.nombre = self.ck["arquitectura"]
        self.titulo = NOMBRES[self.nombre]
        s = self.model.spec
        self.tf = construir_transform(s["img_size"], s["mean"], s["std"], train=False)
        self.ultimo_ms = 0.0

    @torch.inference_mode()
    def probabilidades(self, img) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        t = time.perf_counter()
        p = self.model(x).float().softmax(1)[0].cpu().numpy()
        self.ultimo_ms = (time.perf_counter() - t) * 1000
        return p


def cargar_predictores(runs: str | Path, cuales=None) -> dict[str, Predictor]:
    preds = {}
    for n in cuales or MODELOS:
        p = Path(runs) / f"{n}.pt"
        if p.exists():
            preds[n] = Predictor(p)
    return preds


def predecir_cli(args):
    nombre = modelo_seleccionado(args.out) if args.model == "auto" else args.model
    nombre = nombre or "vit"
    pred = Predictor(Path(args.out) / f"{nombre}.pt")
    entrada = Path(args.input)
    archivos = sorted(p for p in entrada.rglob("*") if p.suffix.lower() in EXTS) if entrada.is_dir() else [entrada]
    print(f"Modelo: {pred.titulo} | umbral {args.threshold:.2f}")
    for a in archivos:
        with Image.open(a) as im:
            p = pred.probabilidades(im.convert("RGB"))
        i = int(p.argmax())
        etiqueta = CLASS_NAMES[i] if p[i] >= args.threshold else "INCIERTO"
        detalle = "  ".join(f"{c}:{v:.2f}" for c, v in zip(CLASS_NAMES, p))
        print(f"{a.name:<30} -> {etiqueta:<9} ({p[i]:.1%})  [{detalle}]  {pred.ultimo_ms:.0f} ms")
