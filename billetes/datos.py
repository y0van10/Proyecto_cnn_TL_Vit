"""Preparación del dataset, transformaciones y DataLoaders.

Dataset: "Peruvian Banknotes" (Caytuiro-Silva et al., 2023, Data in Brief 109715),
9 315 imágenes en 16 subcategorías (denominación x cara x serie). Aquí se agrupan
en 4 clases: S/10, S/20, S/50 y S/100.

Particiones usadas (las oficiales del dataset):
  train <- training_set   (6 568 imágenes, fotos en su mayoría planas/recortadas)
  val   <- validation_set (2 486 imágenes, billetes sostenidos en la mano)
  test  <- testing_set    (261 imágenes SIN etiqueta en el original; se etiquetaron
                           manualmente en datos/testing_set_labels.csv)
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import CLASSES

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SPLIT_ORIGEN = {"training_set": "train", "validation_set": "val", "testing_set": "test"}
RAIZ = Path(__file__).resolve().parent.parent
LABELS_TEST = RAIZ / "datos" / "testing_set_labels.csv"


# --------------------------------------------------------------------------- etiquetas
def parse_carpeta(nombre: str):
    """'billete20_reverso_nuevo' -> ('20', 'reverso', 'nuevo'). Devuelve None si no aplica."""
    s = nombre.lower()
    m = re.search(r"(?<!\d)(10|20|50|100)(?!\d)", s.replace("_", " ").replace("-", " "))
    if not m:
        return None
    cara = "anverso" if "anv" in s else ("reverso" if "rev" in s else "?")
    serie = "nuevo" if "nuev" in s else ("antiguo" if "antig" in s else "?")
    return m.group(1), cara, serie


def leer_labels_test(path: Path) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf8") as f:
        return {r["archivo"]: r["denominacion"] for r in csv.DictReader(f)}


# --------------------------------------------------------------------------- hashing
def dhash(img: Image.Image, n: int = 8) -> int:
    """Hash perceptual por diferencias (64 bits) para detectar casi-duplicados."""
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def casi_duplicados(h_a: list[int], h_b: list[int], umbral: int = 4) -> int:
    """Cuántas imágenes de B tienen un casi-duplicado (distancia Hamming <= umbral) en A."""
    if not h_a or not h_b:
        return 0
    a = np.array(h_a, dtype=np.uint64)
    total = 0
    for hb in h_b:
        d = _popcount64(np.bitwise_xor(a, np.uint64(hb)))
        total += int((d <= umbral).any())
    return total


# --------------------------------------------------------------------------- preparar
def inspeccionar(source: str):
    root = Path(source)
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        n = sum(1 for x in path.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
        if n:
            print(f"{path.relative_to(root)}: {n} imágenes | etiqueta: {parse_carpeta(path.name)}")


def preparar(source: str, data_dir: str, size: int = 256, test_labels: str | None = None,
             overwrite: bool = False, seed: int = 42):
    """Copia (y reescala) el dataset a data_dir con CSV por partición y manifest.json.

    size: lado menor de salida en píxeles (0 = copiar el original sin reescalar).
    """
    src = Path(source).resolve()
    dst = Path(data_dir).resolve()
    if not src.exists():
        raise SystemExit(f"No existe el dataset: {src}")
    if dst == src or src in dst.parents:
        raise SystemExit("La carpeta destino no puede estar dentro del dataset original")
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise SystemExit(f"{dst} no está vacío. Usa --overwrite para regenerarlo.")
        shutil.rmtree(dst)
    labels_test = leer_labels_test(Path(test_labels) if test_labels else LABELS_TEST)

    registros = []  # (ruta_origen, split, denominacion, cara, serie)
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        partes = [x.lower() for x in p.relative_to(src).parts[:-1]]
        split = next((SPLIT_ORIGEN[x] for x in partes if x in SPLIT_ORIGEN), None)
        if split is None:
            continue
        info = next((parse_carpeta(x) for x in reversed(partes) if parse_carpeta(x)), None)
        if info is None and split == "test":
            d = labels_test.get(p.name)
            info = (d, "?", "?") if d else None
        if info is None:
            continue
        registros.append((p, split, *info))

    if not registros:
        raise SystemExit("No se reconocieron imágenes. Ejecuta 'inspect' y revisa la estructura.")
    if not any(r[1] == "test" for r in registros):
        # Sin test etiquetado: se separa 50 % de validation_set por bloques contiguos
        # (las fotos consecutivas suelen ser de la misma escena; así se reduce la fuga).
        print("AVISO: testing_set sin etiquetas -> test interno = mitad de validation_set (por bloques)")
        por_sub = defaultdict(list)
        for i, r in enumerate(registros):
            if r[1] == "val":
                por_sub[(r[2], r[3], r[4])].append(i)
        for idx in por_sub.values():
            idx.sort(key=lambda i: registros[i][0].name)
            for i in idx[len(idx) // 2:]:
                registros[i] = (registros[i][0], "test", *registros[i][2:])

    filas = defaultdict(list)
    hashes = defaultdict(list)
    for k, (p, split, d, cara, serie) in enumerate(registros):
        sub = f"billete{d}_{cara}_{serie}" if split != "test" or cara != "?" else f"billete{d}"
        out = dst / "imagenes" / split / sub / (p.stem + ".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if size and min(im.size) > size:
                r = size / min(im.size)
                im = im.resize((round(im.width * r), round(im.height * r)), Image.BICUBIC)
            im.save(out, quality=95)
            hashes[split].append(dhash(im))
        filas[split].append({"ruta": out.relative_to(dst).as_posix(), "clase": d,
                             "cara": cara, "serie": serie, "origen": p.name})
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(registros)} imágenes procesadas")

    for split, rows in filas.items():
        with open(dst / f"{split}.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    conteos = {s: {c: sum(r["clase"] == c for r in filas[s]) for c in CLASSES} for s in filas}
    subclases = {s: dict(Counter(f"billete{r['clase']}_{r['cara']}_{r['serie']}" for r in filas[s]))
                 for s in filas}
    dup = {
        "val_con_duplicado_en_train": casi_duplicados(hashes["train"], hashes["val"]),
        "test_con_duplicado_en_train": casi_duplicados(hashes["train"], hashes["test"]),
        "test_con_duplicado_en_val": casi_duplicados(hashes["val"], hashes["test"]),
        "criterio": "dHash 64 bits, distancia de Hamming <= 4",
    }
    manifest = {
        "clases": CLASSES, "seed": seed, "origen": str(src), "lado_menor_px": size,
        "conteos": conteos, "subclases": subclases, "casi_duplicados": dup,
        "test_etiquetado_manual": bool(labels_test),
        "nota": ("train=training_set, val=validation_set, test=testing_set oficial "
                 "etiquetado manualmente (verificar datos/testing_set_labels.csv)."),
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"conteos": conteos, "casi_duplicados": dup}, indent=2, ensure_ascii=False))
    return manifest


# --------------------------------------------------------------------------- transformaciones
class RandomRot90:
    """Rota 0/90/180/270 grados: un billete puede mostrarse en cualquier orientación."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.rotate(random.choice([90, 180, 270]), expand=True)
        return img


def construir_transform(img_size: int, mean, std, train: bool):
    from torchvision import transforms as T

    if train:
        previo = [T.Resize(int(img_size * 1.25), antialias=True)] if img_size < 200 else []
        return T.Compose(previo + [
            RandomRot90(p=0.5),
            T.RandomRotation(12, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(img_size, scale=(0.30, 1.0), ratio=(0.75, 1.333), antialias=True),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random"),  # dedos / oclusiones
        ])
    # Evaluación y cámara: recorte cuadrado central SIN deformar la imagen.
    return T.Compose([
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# --------------------------------------------------------------------------- Dataset
class BilletesDataset:
    def __init__(self, data_dir: str, split: str, transform=None):
        self.root = Path(data_dir)
        csv_path = self.root / f"{split}.csv"
        if not csv_path.exists():
            raise SystemExit(f"No existe {csv_path}. Ejecuta primero: python proyecto.py prepare ...")
        with open(csv_path, encoding="utf8") as f:
            self.filas = list(csv.DictReader(f))
        self.targets = [CLASSES.index(r["clase"]) for r in self.filas]
        self.transform = transform

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        with Image.open(self.root / self.filas[i]["ruta"]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


def crear_loader(data_dir, split, img_size, mean, std, batch, train, workers=2):
    import torch

    ds = BilletesDataset(data_dir, split, construir_transform(img_size, mean, std, train))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=workers, drop_last=False,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    ), ds


def pesos_clase(targets, n_clases=len(CLASSES)):
    """Pesos inversamente proporcionales a la frecuencia (normalizados a media 1)."""
    c = np.bincount(targets, minlength=n_clases).astype(float)
    w = c.sum() / (n_clases * np.maximum(c, 1))
    return w / w.mean()
