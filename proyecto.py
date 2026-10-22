"""Clasificación de billetes peruanos: CNN desde cero vs MobileNetV3 (TL) vs ViT-B/32.

Comandos:
  python proyecto.py inspect  --source RUTA_DATASET
  python proyecto.py prepare  --source RUTA_DATASET            [--size 256] [--overwrite]
  python proyecto.py train    --model {cnn,mobilenet,vit}       [--epochs N] [--batch B]
  python proyecto.py evaluate                                   [--split test] [--robustez]
  python proyecto.py explain                                    (Grad-CAM / atención)
  python proyecto.py predict  --input imagen_o_carpeta          [--model vit]
  python proyecto.py webcam                                     [--camera 0] [--model auto]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    x = sub.add_parser("inspect", help="lista carpetas e imágenes del dataset original")
    x.add_argument("--source", required=True)

    x = sub.add_parser("prepare", help="agrupa 16 subclases en 4 denominaciones y reescala")
    x.add_argument("--source", required=True, help="carpeta con training_set/validation_set/testing_set")
    x.add_argument("--data", default=str(ROOT / "data"))
    x.add_argument("--size", type=int, default=256, help="lado menor en px (0 = sin reescalar)")
    x.add_argument("--test-labels", default=None, help="CSV con etiquetas del testing_set")
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--overwrite", action="store_true")

    x = sub.add_parser("train", help="entrena un modelo")
    x.add_argument("--model", choices=["cnn", "mobilenet", "vit"], required=True)
    x.add_argument("--data", default=str(ROOT / "data"))
    x.add_argument("--out", default=str(ROOT / "runs"))
    x.add_argument("--epochs", type=int, help="épocas de la última fase (fine-tuning / end-to-end)")
    x.add_argument("--head-epochs", type=int, help="épocas de la fase 1 (solo capas finales)")
    x.add_argument("--unfreeze", type=int, help="bloques a descongelar en fine-tuning (-1 = todos)")
    x.add_argument("--lr", type=float, help="tasa de aprendizaje de la última fase (capas finales)")
    x.add_argument("--batch", type=int, default=32)
    x.add_argument("--img-size", type=int)
    x.add_argument("--workers", type=int, default=2)
    x.add_argument("--patience", type=int, default=8)
    x.add_argument("--seed", type=int, default=42)
    x.add_argument("--weights", help="archivo .pth local con pesos preentrenados (modo sin internet)")
    x.add_argument("--half", action="store_true", help="guardar checkpoint en float16 (mitad de tamaño)")
    x.add_argument("--max-batches", type=int, help="(depuración) lotes por época")
    x.add_argument("--verbose", action="store_true")

    x = sub.add_parser("evaluate", help="compara los modelos entrenados")
    x.add_argument("--data", default=str(ROOT / "data"))
    x.add_argument("--out", default=str(ROOT / "runs"))
    x.add_argument("--split", nargs="+", default=["val", "test"])
    x.add_argument("--models", nargs="+", default=["cnn", "mobilenet", "vit"])
    x.add_argument("--batch", type=int, default=32)
    x.add_argument("--workers", type=int, default=2)
    x.add_argument("--threshold", type=float, default=0.70)
    x.add_argument("--robustez", action="store_true", help="evalúa el test con perturbaciones")

    x = sub.add_parser("explain", help="mapas Grad-CAM / atención sobre imágenes de test")
    x.add_argument("--data", default=str(ROOT / "data"))
    x.add_argument("--out", default=str(ROOT / "runs"))
    x.add_argument("--n", type=int, default=6)

    x = sub.add_parser("predict", help="clasifica una imagen o carpeta")
    x.add_argument("--input", required=True)
    x.add_argument("--out", default=str(ROOT / "runs"))
    x.add_argument("--model", default="auto")
    x.add_argument("--threshold", type=float, default=0.70)

    x = sub.add_parser("webcam", help="aplicación en tiempo real con la cámara web")
    x.add_argument("--out", default=str(ROOT / "runs"))
    x.add_argument("--camera", default="0", help="índice de cámara, archivo de video o carpeta de imágenes")
    x.add_argument("--model", default="auto", help="auto = el seleccionado en runs/seleccion.json")
    x.add_argument("--threshold", type=float, default=0.70)
    x.add_argument("--smooth", type=float, default=0.6, help="suavizado temporal EMA (0 = sin suavizar)")
    x.add_argument("--save-video", help="(opcional) guardar la sesión en un .mp4")
    x.add_argument("--max-frames", type=int, help="(pruebas) detener tras N cuadros")
    x.add_argument("--headless", action="store_true", help="(pruebas) no abrir ventana")
    x.add_argument("--compare", action="store_true", help="iniciar en modo comparación (3 modelos)")
    x.add_argument("--lang", choices=["es", "en"], help="idioma de la interfaz (en = inglés)")

    a = p.parse_args()
    if a.cmd == "inspect":
        from billetes.datos import inspeccionar
        inspeccionar(a.source)
    elif a.cmd == "prepare":
        from billetes.datos import preparar
        preparar(a.source, a.data, a.size, a.test_labels, a.overwrite, a.seed)
    elif a.cmd == "train":
        from billetes.entrenar import entrenar
        entrenar(a)
    elif a.cmd == "evaluate":
        from billetes.evaluar import evaluar
        evaluar(a)
    elif a.cmd == "explain":
        from billetes.explicar import explicar
        explicar(a)
    elif a.cmd == "predict":
        from billetes.inferencia import predecir_cli
        predecir_cli(a)
    elif a.cmd == "webcam":
        from app_webcam import ejecutar
        ejecutar(a)


if __name__ == "__main__":
    main()
