"""Prueba del Bloque 1 de la Etapa 3: propiedades por región.

Uso (desde la raíz del proyecto):
    uv run python scripts/morfologia/bloque1_propiedades.py IMAGEN
    uv run python scripts/morfologia/bloque1_propiedades.py CARPETA

Con una imagen: corre las Etapas 1 y 2 sin abrir figuras, imprime las
primeras regiones de cada fase y abre una figura con los centroides y
las bounding boxes (en rojo las que tocan el borde).
Con una carpeta: imprime cuántas regiones hay por fase en cada imagen.

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

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.morphology import extract_region_properties  # noqa: E402
from tem.segmentation import PHASE_COLORS  # noqa: E402

# Cantidad de regiones por fase que se listan en pantalla
FILAS_TABLA = 15


def segmentar(path: Path) -> dict:
    """Corre las Etapas 1 y 2 sin abrir figuras ni imprimir.

    Args:
        path: Ruta a la imagen.

    Returns:
        Resultado de pipeline_segmentation.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        pre = pipeline_tem(path, show=False)
        return pipeline_segmentation(pre, show=False)


def probar_imagen(path: Path) -> None:
    """Extrae las propiedades de una imagen y las muestra.

    Args:
        path: Ruta a la imagen.
    """
    seg = segmentar(path)
    print(f"{path.name}: {seg['n_phases']} fases, método "
          f"'{seg['threshold_method']}'\n")
    print("=== extract_region_properties ===")
    props = extract_region_properties(seg["mask"], seg["processed"])

    for fase, regiones in props.items():
        print(f"\n{fase} (primeras {min(FILAS_TABLA, len(regiones))} "
              f"de {len(regiones)}):")
        print(f"  {'id':>4} {'área':>8} {'perím.':>7} {'eje may':>7} "
              f"{'eje men':>7} {'orient°':>7} {'solidez':>7} "
              f"{'excentr':>7} borde")
        for r in regiones[:FILAS_TABLA]:
            print(f"  {r['id']:>4} {r['area_px']:>8.0f} "
                  f"{r['perimetro_px']:>7.1f} {r['eje_mayor_px']:>7.1f} "
                  f"{r['eje_menor_px']:>7.1f} {r['orientacion_deg']:>7.1f} "
                  f"{r['solidez']:>7.3f} {r['excentricidad']:>7.3f} "
                  f"{'sí' if r['toca_borde'] else ''}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(seg["original"], cmap="gray")
    for fase, regiones in props.items():
        color = PHASE_COLORS[int(fase.split("_")[1]) % len(PHASE_COLORS)]
        for r in regiones:
            y0, x0, y1, x1 = r["bbox"]
            borde = "red" if r["toca_borde"] else color
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=borde,
                                       linewidth=1))
            ax.plot(r["centroide"][1], r["centroide"][0], "+",
                    color=borde, markersize=6)
    resumen = ", ".join(f"{f}: {len(r)}" for f, r in props.items())
    ax.set_title(f"Bloque 1 — {path.name}\nRegiones: {resumen} "
                 "(rojo = tocan el borde)")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Cuenta las regiones por fase de todas las imágenes.

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
    print(f"{'archivo':36} regiones por fase (tocan el borde)")
    for path in files:
        seg = segmentar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            props = extract_region_properties(seg["mask"],
                                              seg["processed"])
        resumen = "  ".join(
            f"{f}: {len(r)} ({sum(x['toca_borde'] for x in r)})"
            for f, r in props.items()
        )
        print(f"{path.name:36} {resumen}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
