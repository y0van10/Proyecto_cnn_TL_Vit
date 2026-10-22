"""Aplicación de cámara web: clasificación de billetes peruanos en tiempo real.

Uso:
  python app_webcam.py                      # cámara 0, modelo seleccionado en runs/seleccion.json
  python app_webcam.py --camera 1           # otra cámara
  python app_webcam.py --camera video.mp4   # un video (o una carpeta de imágenes) en vez de cámara
  python app_webcam.py --model vit          # forzar un modelo

Teclas:
  1 / 2 / 3   CNN / MobileNetV3 / ViT
  C           modo comparación (los 3 modelos a la vez; o inicia con --compare)
  + / -       subir / bajar umbral de confianza
  F           congelar cuadro          M  espejo (solo visual)
  S           guardar captura en capturas/      Q o ESC  salir

Nota: clasifica la DENOMINACIÓN del billete dentro del recuadro; no verifica autenticidad.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import billetes
from billetes import CLASS_NAMES, MODELOS, NOMBRES
from billetes.datos import EXTS
from billetes.inferencia import cargar_predictores, modelo_seleccionado

ROOT = Path(__file__).resolve().parent
W, H, PANEL = 1280, 720, 430           # lienzo y ancho del panel lateral
# BGR (paleta categórica validada: azul, naranja, aqua) + tonos de interfaz
COLOR_MODELO = {"cnn": (214, 120, 42), "mobilenet": (52, 104, 235), "vit": (122, 175, 27)}
FONDO, FONDO2, TEXTO, TEXTO2 = (30, 24, 20), (48, 40, 34), (240, 240, 240), (170, 170, 165)
OK, DUDA = (90, 200, 80), (60, 170, 245)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
CORTO = {"cnn": "CNN", "mobilenet": "MobileNetV3", "vit": "ViT-B/32"}


def txt(img, s, x, y, escala=0.55, color=TEXTO, grosor=1):
    cv2.putText(img, s, (int(x), int(y)), FUENTE, escala, color, grosor, cv2.LINE_AA)


class Fuente:
    """Cámara (índice), archivo de video o carpeta de imágenes."""

    def __init__(self, ref: str):
        self.imagenes = None
        p = Path(ref)
        if ref.isdigit():
            idx = int(ref)
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(idx, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif p.is_dir():
            self.imagenes = sorted(x for x in p.rglob("*") if x.suffix.lower() in EXTS)
            self.i, self.rep = 0, 0
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(str(p))
        if self.imagenes is None and not self.cap.isOpened():
            raise SystemExit(f"No se pudo abrir la fuente '{ref}'. Prueba --camera 1 o revisa permisos.")

    def leer(self):
        if self.imagenes is not None:
            if self.i >= len(self.imagenes):
                return False, None
            img = cv2.imread(str(self.imagenes[self.i]))
            self.rep += 1
            if self.rep >= 12:          # cada imagen se muestra 12 cuadros
                self.i, self.rep = self.i + 1, 0
            h, w = img.shape[:2]
            s = 720 / max(h, w)
            return True, cv2.resize(img, (int(w * s), int(h * s)))
        return self.cap.read()

    def cerrar(self):
        if self.cap is not None:
            self.cap.release()


def roi_cuadrado(frame, frac=0.78):
    h, w = frame.shape[:2]
    lado = int(min(h, w) * frac)
    x, y = (w - lado) // 2, (h - lado) // 2
    return x, y, lado


def esquinas(img, x, y, lado, color, largo=34, g=4):
    for (cx, cy, dx, dy) in [(x, y, 1, 1), (x + lado, y, -1, 1), (x, y + lado, 1, -1), (x + lado, y + lado, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * largo, cy), color, g, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * largo), color, g, cv2.LINE_AA)


def barras(img, probs, x, y, ancho, color, alto=16, sep=30):
    for k, (c, p) in enumerate(zip(CLASS_NAMES, probs)):
        yy = y + k * sep
        txt(img, c, x, yy + 13, 0.5, TEXTO)
        bx = x + 52
        cv2.rectangle(img, (bx, yy), (bx + ancho, yy + alto), FONDO2, -1)
        cv2.rectangle(img, (bx, yy), (bx + max(2, int(ancho * p)), yy + alto), color, -1)
        txt(img, f"{p * 100:5.1f}%", bx + ancho + 8, yy + 13, 0.45, TEXTO2)


def ejecutar(args):
    if getattr(args, "lang", None):
        billetes.LANG = args.lang
    t = billetes.t
    nombre_m = {"cnn": t("CNN desde cero", "CNN from scratch"), "mobilenet": NOMBRES["mobilenet"], "vit": NOMBRES["vit"]}
    runs = Path(args.out)
    preds = cargar_predictores(runs)
    if not preds:
        raise SystemExit(f"No hay modelos entrenados en {runs}. Ejecuta primero: python proyecto.py train ...")
    activo = args.model if args.model in preds else (modelo_seleccionado(runs) or next(iter(preds)))
    activo = activo if activo in preds else next(iter(preds))
    print("Modelos cargados:", ", ".join(nombre_m[n] for n in preds), "| activo:", nombre_m[activo])

    fuente = Fuente(str(args.camera))
    umbral, alpha = args.threshold, args.smooth
    comparar, congelado, espejo = bool(getattr(args, "compare", False)), False, False
    ema = {n: None for n in preds}
    fps, t_prev, n_frames = 0.0, time.perf_counter(), 0
    writer = None
    (ROOT / "capturas").mkdir(exist_ok=True)
    frame = None

    while True:
        if not congelado or frame is None:
            ok, nuevo = fuente.leer()
            if not ok:
                break
            frame = nuevo
        x, y, lado = roi_cuadrado(frame)
        recorte = cv2.cvtColor(frame[y:y + lado, x:x + lado], cv2.COLOR_BGR2RGB)

        usar = list(preds) if comparar else [activo]
        for n in usar:
            p = preds[n].probabilidades(recorte)
            ema[n] = p if ema[n] is None or alpha <= 0 else alpha * ema[n] + (1 - alpha) * p

        # ---------------- lienzo -------------------------------------------------
        lienzo = np.full((H, W, 3), FONDO, np.uint8)
        vista = frame.copy()
        probs = ema[activo]
        i_best = int(np.argmax(probs))
        conf = float(probs[i_best])
        aceptado = conf >= umbral
        color_roi = OK if aceptado else DUDA
        cv2.rectangle(vista, (x, y), (x + lado, y + lado), (255, 255, 255), 1)
        esquinas(vista, x, y, lado, color_roi)
        if espejo:
            vista = cv2.flip(vista, 1)
        vh, vw = vista.shape[:2]
        s = min((W - PANEL) / vw, (H - 40) / vh)
        vista = cv2.resize(vista, (int(vw * s), int(vh * s)))
        ox, oy = ((W - PANEL) - vista.shape[1]) // 2, (H - 40 - vista.shape[0]) // 2
        lienzo[oy:oy + vista.shape[0], ox:ox + vista.shape[1]] = vista
        if not aceptado:
            txt(lienzo, t("Coloque UN billete dentro del recuadro", "Place ONE banknote inside the box"), ox + 14, oy + 30, 0.6, DUDA, 2)

        px = W - PANEL + 22
        cv2.rectangle(lienzo, (W - PANEL, 0), (W, H), FONDO2, -1)
        cv2.rectangle(lienzo, (W - PANEL + 8, 8), (W - 8, H - 48), FONDO, -1)
        txt(lienzo, t("BILLETES PERUANOS", "PERUVIAN BANKNOTES"), px, 42, 0.75, TEXTO, 2)
        txt(lienzo, t("Clasificacion de denominacion", "Denomination classifier"), px, 66, 0.5, TEXTO2)

        if not comparar:
            n = activo
            txt(lienzo, t("Modelo: ", "Model: ") + nombre_m[n], px, 106, 0.55, COLOR_MODELO[n], 2)
            txt(lienzo, t("Prediccion", "Prediction"), px, 150, 0.5, TEXTO2)
            etiqueta = CLASS_NAMES[i_best] if aceptado else t("INCIERTO", "UNCERTAIN")
            txt(lienzo, etiqueta, px, 205, 1.6, OK if aceptado else DUDA, 3)
            txt(lienzo, t("Confianza", "Confidence") + f" {conf * 100:.1f}%  (" + t("umbral", "threshold") + f" {umbral * 100:.0f}%)", px, 240, 0.5, TEXTO)
            txt(lienzo, t("PROBABILIDADES", "PROBABILITIES"), px, 285, 0.45, TEXTO2)
            barras(lienzo, probs, px, 298, 230, COLOR_MODELO[n])
            txt(lienzo, t("Inferencia", "Inference") + f": {preds[n].ultimo_ms:.0f} ms", px, 445, 0.5, TEXTO2)
            y0 = 490
            txt(lienzo, t("Modelos disponibles", "Available models"), px, y0, 0.45, TEXTO2)
            for k, m in enumerate(MODELOS):
                estado = t("activo", "active") if m == n else (t("listo", "ready") if m in preds else t("sin entrenar", "not trained"))
                c = COLOR_MODELO[m] if m in preds else (90, 90, 90)
                txt(lienzo, f"[{k + 1}] {CORTO[m]:<12} {estado}", px, y0 + 26 + 24 * k, 0.5, c)
        else:
            txt(lienzo, t("MODO COMPARACION", "COMPARISON MODE"), px, 106, 0.6, TEXTO, 2)
            for k, n in enumerate(preds):
                yb = 130 + k * 190
                pr = ema[n]
                j = int(np.argmax(pr))
                et = CLASS_NAMES[j] if pr[j] >= umbral else t("INCIERTO", "UNCERTAIN")
                txt(lienzo, f"{CORTO[n]}", px, yb + 20, 0.6, COLOR_MODELO[n], 2)
                txt(lienzo, f"{et}  {pr[j] * 100:.0f}%", px + 190, yb + 20, 0.6, OK if pr[j] >= umbral else DUDA, 2)
                barras(lienzo, pr, px, yb + 34, 200, COLOR_MODELO[n], alto=14, sep=26)
                txt(lienzo, f"{preds[n].ultimo_ms:.0f} ms", px + 300, yb + 20, 0.45, TEXTO2)

        ahora = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1 / max(ahora - t_prev, 1e-6)) if fps else 1 / max(ahora - t_prev, 1e-6)
        t_prev = ahora
        cv2.rectangle(lienzo, (0, H - 40), (W, H), (18, 14, 12), -1)
        ayuda = t("1 CNN  2 MobileNetV3  3 ViT  C comparar  +/- umbral  F congelar  M espejo  S captura  Q salir",
                  "1 CNN  2 MobileNetV3  3 ViT  C compare  +/- threshold  F freeze  M mirror  S screenshot  Q quit")
        txt(lienzo, f"{ayuda}   |   {fps:4.1f} FPS", 14, H - 14, 0.5, TEXTO2)
        txt(lienzo, t("No verifica autenticidad", "Does not check authenticity"), W - PANEL + 22, H - 60, 0.42, (120, 120, 120))

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
            writer.write(lienzo)
        n_frames += 1
        if args.max_frames and n_frames >= args.max_frames:
            cv2.imwrite(str(ROOT / "capturas" / "ultima.png"), lienzo)
            break
        if args.headless:
            if n_frames % 12 == 6:
                cv2.imwrite(str(ROOT / "capturas" / f"frame_{n_frames:04d}.png"), lienzo)
            continue

        cv2.imshow("Billetes peruanos - clasificacion en tiempo real", lienzo)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("1"), ord("2"), ord("3")) and MODELOS[k - ord("1")] in preds:
            activo, comparar = MODELOS[k - ord("1")], False
        elif k == ord("c"):
            comparar = not comparar
            ema = {n: None for n in preds}
        elif k in (ord("+"), ord("=")):
            umbral = min(0.99, umbral + 0.05)
        elif k == ord("-"):
            umbral = max(0.30, umbral - 0.05)
        elif k == ord("f"):
            congelado = not congelado
        elif k == ord("m"):
            espejo = not espejo
        elif k == ord("s"):
            nombre = ROOT / "capturas" / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(nombre), lienzo)
            print("Captura guardada:", nombre)

    fuente.cerrar()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--camera", default="0")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--smooth", type=float, default=0.6)
    ap.add_argument("--save-video")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--compare", action="store_true", help="iniciar en modo comparación")
    ap.add_argument("--lang", choices=["es", "en"], help="idioma de la interfaz (en = inglés, para la exposición)")
    ejecutar(ap.parse_args())
