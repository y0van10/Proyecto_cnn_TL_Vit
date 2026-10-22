"""Clasificación de denominaciones de billetes peruanos (S/10, S/20, S/50, S/100).

Paquete del proyecto de la primera unidad de Visión Artificial (UNA Puno):
  - datos.py       preparación del dataset, transformaciones y DataLoaders
  - modelos.py     CNN desde cero, MobileNetV3 (transfer learning) y ViT-B/32
  - entrenar.py    ciclo de entrenamiento por fases con early stopping
  - evaluar.py     métricas, matrices de confusión, latencia, robustez
  - inferencia.py  predictor reutilizable por la app de cámara web
"""

import os

LANG = os.environ.get("BILLETES_LANG", "es")  # "en" = textos de gráficos en inglés


def t(es: str, en: str) -> str:
    """Texto según idioma (BILLETES_LANG=en para figuras en inglés)."""
    return en if LANG == "en" else es


CLASSES = ["10", "20", "50", "100"]          # orden natural de las denominaciones
CLASS_NAMES = [f"S/{c}" for c in CLASSES]
MODELOS = ["cnn", "mobilenet", "vit"]
NOMBRES = {
    "cnn": t("CNN desde cero", "CNN from scratch"),
    "mobilenet": "MobileNetV3-Large (TL)",
    "vit": "ViT-B/32 (TL)",
}
