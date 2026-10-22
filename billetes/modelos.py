"""Los tres modelos del proyecto.

1. cnn        CNN end-to-end entrenada desde cero (pesos aleatorios).
2. mobilenet  Transferencia de aprendizaje: MobileNetV3-Large preentrenada en ImageNet-1k
              + capas finales nuevas (MLP 1280 -> 256 -> 4).
3. vit        Vision Transformer ViT-B/32 (Dosovitskiy et al., 2021) preentrenado en
              ImageNet-21k + cabeza lineal nueva; ajuste fino de los últimos bloques.

Los pesos preentrenados se descargan de las *releases* de GitHub de timm (los mismos
pesos que publica timm en Hugging Face). Si GitHub falla se intenta Hugging Face.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GH = "https://github.com/huggingface/pytorch-image-models/releases/download"
SPEC = {
    "cnn": dict(img_size=128, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "mobilenet": dict(img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                      timm="mobilenetv3_large_100", hf="mobilenetv3_large_100.ra_in1k",
                      url=f"{GH}/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth"),
    "vit": dict(img_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                timm="vit_base_patch32_224", hf="vit_base_patch32_224.orig_in21k",
                url=f"{GH}/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth"),
}


# --------------------------------------------------------------------------- 1. CNN
def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BilletesCNN(nn.Module):
    """CNN tipo VGG compacta (~1.2 M parámetros, 8 capas convolucionales) para 128x128.

    128 -> conv s2 -> 64 -> [32] pool -> 32 -> [64,64] pool -> 16
        -> [128,128] pool -> 8 -> [256,256] -> GAP -> Dropout -> FC(4)
    """

    def __init__(self, n_clases=len(CLASSES), dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32, stride=2),
            conv_bn(32, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),
            conv_bn(128, 256), conv_bn(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_clases))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    # interfaz común
    def grupos_parametros(self):
        return {"cabeza": list(self.classifier.parameters()), "backbone": list(self.features.parameters())}

    def capa_gradcam(self):
        return self.features[-1]


# --------------------------------------------------------------------------- 2/3. Transfer learning
class ModeloTL(nn.Module):
    """Backbone preentrenado (timm, sin clasificador) + capas finales nuevas."""

    def __init__(self, nombre: str, backbone: nn.Module, cabeza: nn.Module):
        super().__init__()
        self.nombre = nombre
        self.backbone = backbone
        self.head = cabeza

    def forward(self, x):
        return self.head(self.backbone(x))

    # --- congelar / descongelar --------------------------------------------
    def _bloques(self):
        return list(self.backbone.blocks)

    def congelar_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def descongelar_ultimos(self, n: int):
        """Descongela los últimos n bloques (n<0 = todo el backbone) y las capas posteriores."""
        if n < 0:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        self.congelar_backbone()
        if n == 0:  # extracción de características: solo se entrenan las capas finales nuevas
            return
        bloques = self._bloques()
        for b in bloques[len(bloques) - n:]:
            for p in b.parameters():
                p.requires_grad = True
        # capas después de los bloques
        for nombre in ("conv_head", "norm", "fc_norm"):
            m = getattr(self.backbone, nombre, None)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm de capas congeladas en modo evaluación (conserva estadísticas de ImageNet)
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def grupos_parametros(self):
        return {"cabeza": list(self.head.parameters()), "backbone": list(self.backbone.parameters())}

    def capa_gradcam(self):
        return self.backbone.blocks[-1]


def _cargar_pesos(backbone: nn.Module, nombre: str, pesos: str | None):
    import timm

    spec = SPEC[nombre]
    sd = None
    fuente = None
    try:
        if pesos:
            sd = torch.load(pesos, map_location="cpu", weights_only=True)
            fuente = pesos
        else:
            sd = torch.hub.load_state_dict_from_url(spec["url"], map_location="cpu", progress=True,
                                                    check_hash=False)
            fuente = spec["url"]
    except Exception as e:  # GitHub no disponible -> Hugging Face vía timm
        print(f"[pesos] GitHub falló ({e}); intentando Hugging Face: {spec['hf']}")
        ref = timm.create_model(spec["hf"], pretrained=True, num_classes=0)
        sd = ref.state_dict()
        fuente = f"hf-hub:timm/{spec['hf']}"
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if nombre == "vit":
        from timm.models.vision_transformer import checkpoint_filter_fn
        sd = checkpoint_filter_fn(sd, backbone)
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "classifier.", "pre_logits.", "fc."))}
    faltan, sobran = backbone.load_state_dict(sd, strict=False)
    faltan = [k for k in faltan if not k.startswith(("head", "classifier", "fc_norm"))]
    if faltan:
        raise RuntimeError(f"Pesos incompletos para {nombre}: faltan {faltan[:5]}...")
    print(f"[pesos] {nombre}: preentrenado cargado desde {fuente} (ignoradas: {len(sobran)})")


def crear_modelo(nombre: str, preentrenado: bool = True, pesos: str | None = None) -> nn.Module:
    if nombre == "cnn":
        m = BilletesCNN()
    elif nombre in ("mobilenet", "vit"):
        import timm

        spec = SPEC[nombre]
        backbone = timm.create_model(spec["timm"], pretrained=False, num_classes=0)
        if preentrenado:
            _cargar_pesos(backbone, nombre, pesos)
        with torch.no_grad():  # dimensión real de salida del backbone (1280 en MobileNetV3-L, 768 en ViT-B)
            d = backbone.eval()(torch.zeros(1, 3, spec["img_size"], spec["img_size"])).shape[1]
        if nombre == "mobilenet":
            cabeza = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 256), nn.Hardswish(),
                                   nn.Dropout(0.2), nn.Linear(256, len(CLASSES)))
        else:
            cabeza = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, len(CLASSES)))
            nn.init.zeros_(cabeza[-1].weight)
            nn.init.zeros_(cabeza[-1].bias)
        m = ModeloTL(nombre, backbone, cabeza)
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    m.spec = dict(SPEC[nombre])
    return m


def contar_parametros(m: nn.Module, solo_entrenables=False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not solo_entrenables)


def cargar_checkpoint(path: str | Path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = crear_modelo(ck["arquitectura"], preentrenado=False)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    m.spec.update(img_size=ck["img_size"], mean=tuple(ck["mean"]), std=tuple(ck["std"]))
    if list(ck["clases"]) != CLASSES:
        raise RuntimeError(f"Orden de clases distinto en {path}: {ck['clases']}")
    return m.to(device).eval(), ck
