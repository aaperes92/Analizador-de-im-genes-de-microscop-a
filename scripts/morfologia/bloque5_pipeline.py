"""Prueba del Bloque 5 de la Etapa 3: las tres etapas desde el archivo.

Corre pipeline_tem (Etapa 1), pipeline_segmentation (Etapa 2) y
pipeline_morphology (Etapa 3).

Uso (desde la raíz del proyecto):
    uv run python scripts/morfologia/bloque5_pipeline.py IMAGEN [FASES]
    uv run python scripts/morfologia/bloque5_pipeline.py CARPETA

FASES es opcional: fases a analizar separadas por coma (por ejemplo
"1" o "0,2"). Por defecto, todas menos la mayoritaria.
Con una imagen: abre la rosa de orientaciones y la figura morfológica
(las figuras de las Etapas 1 y 2 se guardan sin abrir ventana).
Con una carpeta: procesa todo sin abrir ventanas e imprime una tabla.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.morphology import pipeline_morphology  # noqa: E402


def procesar(path: Path, fases: list[int] | None, show: bool) -> dict:
    """Corre las tres etapas sobre una imagen.

    Args:
        path: Ruta a la imagen.
        fases: Fases a analizar (None: todas menos la mayoritaria).
        show: Si es True, muestra las figuras de la Etapa 3 y los
            mensajes.

    Returns:
        Resultado de pipeline_morphology.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print(f"=== Etapa 3: pipeline_morphology — {path.name} "
              f"({seg['n_phases']} fases, '{seg['threshold_method']}') ===")
        return pipeline_morphology(seg, phases=fases, show=show)


def procesar_carpeta(folder: Path) -> None:
    """Procesa todas las imágenes de una carpeta e imprime un resumen.

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
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'área media':>10} "
          f"{'forma':>10} {'S':>5}  orientación   {'seg':>5}")
    for path in files:
        t = time.perf_counter()
        r = procesar(path, None, show=False)
        seg = time.perf_counter() - t
        for fase, size in r["size_stats"].items():
            if not size["n_regiones"]:
                print(f"{path.name:36} {fase:7} {0:>4}  (sin regiones)")
                continue
            o = r["orientation_stats"][fase]
            s = (f"{o['indice_orden_S']:.2f}"
                 if o["indice_orden_S"] is not None else "-")
            print(f"{path.name:36} {fase:7} {size['n_regiones']:>4} "
                  f"{size['area_media_px']:>10.0f} "
                  f"{r['shape_stats'][fase]['forma_predominante']:>10} "
                  f"{s:>5}  {str(o['clasificacion']):13} {seg:>5.1f}")
    print("Figuras guardadas en data/samples/")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    fases = ([int(x) for x in sys.argv[2].split(",")]
             if len(sys.argv) == 3 else None)
    if target.is_dir():
        procesar_carpeta(target)
    else:
        procesar(target, fases, show=True)
