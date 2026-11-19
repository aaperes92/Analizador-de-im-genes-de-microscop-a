"""Prueba del Bloque 2 de la Etapa 3: tamaño y forma.

Uso (desde la raíz del proyecto):
    uv run python scripts/morfologia/bloque2_tamano_forma.py IMAGEN
    uv run python scripts/morfologia/bloque2_tamano_forma.py CARPETA

Con una imagen: imprime las estadísticas de tamaño y forma y abre una
figura con el histograma de áreas y la log-normal, circularidad contra
relación de aspecto, y las distribuciones de solidez y excentricidad.
Con una carpeta: imprime una fila resumen por imagen y fase.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.morphology import (  # noqa: E402
    _aspect_ratio,
    _circularity,
    _classify,
    _interior,
    analyze_shape,
    analyze_size_distribution,
    extract_region_properties,
)

# Color de cada forma en el gráfico de dispersión
COLORES_FORMA = {
    "circular": "tab:green", "elongada": "tab:red",
    "irregular": "tab:purple", "compacta": "tab:blue",
}


def propiedades(path: Path, silencioso: bool) -> dict:
    """Corre las Etapas 1 y 2 y el Bloque 1 de la Etapa 3.

    Args:
        path: Ruta a la imagen.
        silencioso: Si es True, no imprime los mensajes.

    Returns:
        Resultado de extract_region_properties.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
    salida = io.StringIO() if silencioso else sys.stdout
    with contextlib.redirect_stdout(salida):
        return extract_region_properties(seg["mask"], seg["processed"])


def probar_imagen(path: Path) -> None:
    """Analiza tamaño y forma de una imagen y lo grafica.

    Args:
        path: Ruta a la imagen.
    """
    props = propiedades(path, silencioso=False)
    print("\n=== analyze_size_distribution ===")
    sizes = analyze_size_distribution(props)
    print("\n=== analyze_shape ===")
    shapes = analyze_shape(props)

    fases = [f for f in props if _interior(props[f])]
    if not fases:
        print("No hay regiones para graficar.")
        return
    fig, axes = plt.subplots(len(fases), 4, figsize=(20, 4.5 * len(fases)),
                             squeeze=False)
    for fila, fase in zip(axes, fases):
        regiones = _interior(props[fase])
        areas = np.array([r["area_px"] for r in regiones])

        # Histograma de áreas (escala log) y log-normal ajustada
        bins = np.logspace(np.log10(areas.min()) - 0.05,
                           np.log10(areas.max()) + 0.05, 30)
        fila[0].hist(areas, bins=bins, density=True, color="tab:blue",
                     alpha=0.6, label="áreas")
        ajuste = sizes[fase]["ajuste_lognormal"]
        if ajuste:
            x = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 200)
            pdf = stats.lognorm.pdf(x, ajuste["shape"], ajuste["loc"],
                                    ajuste["scale"])
            fila[0].plot(x, pdf, color="tab:red",
                         label=f"log-normal (KS p="
                               f"{ajuste['ks_pvalue']:.3f})")
        fila[0].set_xscale("log")
        fila[0].set_xlabel("Área (px, escala log)")
        fila[0].set_ylabel("Densidad")
        fila[0].set_title(f"{fase}: {len(regiones)} regiones")
        fila[0].legend()

        # Circularidad contra relación de aspecto, color por forma
        for r in regiones:
            c, a = _circularity(r), _aspect_ratio(r)
            if a is None:
                continue
            forma = _classify(c, a, r["solidez"])
            fila[1].plot(a, c, "o", color=COLORES_FORMA[forma],
                         markersize=4, alpha=0.7)
        for forma, color in COLORES_FORMA.items():
            fila[1].plot([], [], "o", color=color,
                         label=f"{forma} "
                               f"({shapes[fase]['conteo_formas'][forma]})")
        fila[1].set_xlabel("Relación de aspecto")
        fila[1].set_ylabel("Circularidad")
        fila[1].set_title(
            f"Forma predominante: "
            f"'{shapes[fase]['forma_predominante']}'"
        )
        fila[1].legend(fontsize=8)

        fila[2].hist([r["solidez"] for r in regiones], bins=20,
                     range=(0, 1), color="tab:purple", alpha=0.7)
        fila[2].set_xlabel("Solidez")
        fila[2].set_title(
            f"Solidez media {shapes[fase]['solidez_media']:.2f}"
        )

        fila[3].hist([r["excentricidad"] for r in regiones], bins=20,
                     range=(0, 1), color="tab:orange", alpha=0.7)
        fila[3].set_xlabel("Excentricidad")
        fila[3].set_title(
            f"Excentricidad media "
            f"{shapes[fase]['excentricidad_media']:.2f}"
        )

    fig.suptitle(f"Bloque 2 — {path.name}", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume tamaño y forma de todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS
    )
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'media':>7} "
          f"{'mediana':>7} {'KS p':>6}  forma")
    for path in files:
        props = propiedades(path, silencioso=True)
        with contextlib.redirect_stdout(io.StringIO()):
            sizes = analyze_size_distribution(props)
            shapes = analyze_shape(props)
        for fase in props:
            s, f = sizes[fase], shapes[fase]
            if not s["n_regiones"]:
                print(f"{path.name:36} {fase:7} {0:>4}  (sin regiones)")
                continue
            ks = (f"{s['ajuste_lognormal']['ks_pvalue']:.3f}"
                  if s["ajuste_lognormal"] else "-")
            print(f"{path.name:36} {fase:7} {s['n_regiones']:>4} "
                  f"{s['area_media_px']:>7.0f} {s['area_mediana_px']:>7.0f} "
                  f"{ks:>6}  {f['forma_predominante']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
