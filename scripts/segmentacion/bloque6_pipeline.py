"""Prueba del Bloque 6: proceso completo desde el archivo.

Corre la Etapa 1 (pipeline_tem) y la Etapa 2 (pipeline_segmentation).

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque6_pipeline.py IMAGEN [MÉTODO]
    uv run python scripts/segmentacion/bloque6_pipeline.py CARPETA [MÉTODO]

MÉTODO es opcional: auto (por defecto), otsu, gmm o adaptive.
Con una imagen: muestra la figura final de la segmentación (la de
preprocesamiento se guarda sin abrir ventana).
Con una carpeta: procesa todas las imágenes, guarda las figuras en
data/samples/ sin abrir ventanas e imprime una tabla resumen.

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

from tem import pipeline_tem  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    SEGMENTATION_METHODS,
    pipeline_segmentation,
)


def procesar(path: Path, method: str, show: bool) -> dict:
    """Corre las Etapas 1 y 2 sobre una imagen.

    Args:
        path: Ruta a la imagen.
        method: Método de umbralización.
        show: Si es True, muestra la figura final y los mensajes.

    Returns:
        Resultado de pipeline_segmentation.
    """
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print("=== Etapa 1: pipeline_tem ===")
        pre = pipeline_tem(path, show=False)
        print("\n=== Etapa 2: pipeline_segmentation ===")
        return pipeline_segmentation(pre, method=method, show=show)


def procesar_carpeta(folder: Path, method: str) -> None:
    """Procesa todas las imágenes de una carpeta e imprime un resumen.

    Args:
        folder: Carpeta con imágenes.
        method: Método de umbralización.
    """
    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS
    )
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':16} {'fases':>5} {'método':>8} {'seg':>5}  "
          "área % por fase (regiones)")
    for path in files:
        t = time.perf_counter()
        r = procesar(path, method, show=False)
        fases = "  ".join(
            f"{s['area_pct']:5.1f} ({s['n_regiones']})"
            for k, s in r["stats"].items() if k != "interfaces"
        )
        print(f"{path.name:16} {r['n_phases']:>5} "
              f"{r['threshold_method']:>8} "
              f"{time.perf_counter() - t:>5.1f}  {fases}")
    print("Figuras guardadas en data/samples/")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    metodo = sys.argv[2] if len(sys.argv) == 3 else "auto"
    if metodo not in SEGMENTATION_METHODS:
        print(f"Método '{metodo}' no válido. Opciones: "
              f"{', '.join(SEGMENTATION_METHODS)}")
        sys.exit(1)
    if target.is_dir():
        procesar_carpeta(target, metodo)
    else:
        procesar(target, metodo, show=True)
