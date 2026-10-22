"""Regenera los gráficos de resultados desde runs/ sin volver a ejecutar los modelos.

python herramientas/graficos.py --runs runs --out figuras            # español
BILLETES_LANG=en python herramientas/graficos.py --runs runs --out figures_en
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from billetes import MODELOS, NOMBRES, t  # noqa: E402
from billetes.evaluar import (COLORES, TINTA2, _estilo, graficar_comparacion,  # noqa: E402
                              graficar_confusiones, graficar_robustez)


def curvas_juntas(runs: Path, out: Path):
    plt = _estilo()
    fig, ax = plt.subplots(figsize=(10, 4))
    for n in MODELOS:
        p = runs / f"{n}_historial.csv"
        if not p.exists():
            continue
        h = list(csv.DictReader(open(p, encoding="utf8")))
        ep = [int(r["epoca"]) for r in h]
        f1 = [float(r["val_f1_macro"]) for r in h]
        ax.plot(ep, f1, "-o", ms=3.5, lw=2, color=COLORES[n], label=NOMBRES[n])
        b = max(range(len(f1)), key=f1.__getitem__)
        off = {"cnn": (0, 8), "mobilenet": (-16, 9), "vit": (18, -16)}[n]
        ax.annotate(f"{f1[b]:.3f}", (ep[b], f1[b]), textcoords="offset points", xytext=off, ha="center",
                    fontsize=9, color=TINTA2)
    ax.set_xlabel(t("época", "epoch"))
    ax.set_ylabel(t("F1 macro (validación)", "Macro-F1 (validation)"))
    ax.set_ylim(0.45, 1.01)
    ax.set_title(t("Curvas de validación de los tres modelos", "Validation curves of the three models"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / t("curvas_validacion.png", "validation_curves.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figuras")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = json.loads((runs / "metricas_detalle.json").read_text(encoding="utf8"))
    filas = list(csv.DictReader(open(runs / "comparacion.csv", encoding="utf8")))
    for f in filas:
        for k, v in f.items():
            try:
                f[k] = float(v)
            except (TypeError, ValueError):
                pass
    for split in ("val", "test"):
        graficar_confusiones(det["resultados"], split, out)
    graficar_comparacion(filas, out)
    if det.get("robustez"):
        graficar_robustez(det["robustez"], out)
    curvas_juntas(runs, out)
    print("Gráficos en", out)


if __name__ == "__main__":
    main()
