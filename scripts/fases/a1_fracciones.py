"""Prueba del Bloque A1 de la Etapa 4: fracciones de área y volumen.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/a1_fracciones.py IMAGEN [FONDO]
    uv run python scripts/fases/a1_fracciones.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: corre las Etapas 1 y 2 sin abrir ventanas y muestra la
tabla de fracciones y la advertencia de la fracción de volumen.
Con una carpeta: imprime una fila por imagen.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.fraction import (  # noqa: E402
    compute_area_fractions,
    estimate_volume_fractions,
    phase_title,
)
from tem.loader import SUPPORTED_FORMATS  # noqa: E402


def mascara(path: Path) -> dict:
    """Corre las Etapas 1 y 2 sin abrir ventanas ni imprimir.

    Args:
        path: Ruta a la imagen.

    Returns:
        Resultado de pipeline_segmentation.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return pipeline_segmentation(pipeline_tem(path, show=False),
                                     show=False)


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Calcula y muestra las fracciones de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg = mascara(path)
    print(f"{path.name}: {seg['n_phases']} fases en la máscara, método "
          f"'{seg['threshold_method']}'\n")
    print("=== compute_area_fractions ===")
    areas = compute_area_fractions(seg["mask"], background=fondo)
    print("\n=== estimate_volume_fractions ===")
    estimate_volume_fractions(areas)


def probar_carpeta(folder: Path) -> None:
    """Resume las fracciones de todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fondo%':>7} {'interf%':>7}  "
          "fases (% sin fondo / % sobre el total)")
    for path in files:
        seg = mascara(path)
        with contextlib.redirect_stdout(io.StringIO()):
            areas = compute_area_fractions(seg["mask"])
        fases = "  ".join(
            f"{phase_title(k)}: {v['area_pct']:.1f}/"
            f"{v['area_pct_con_fondo']:.1f}"
            for k, v in areas.items() if k.startswith("fase_")
        )
        print(f"{path.name:36} {areas['fondo']['area_pct']:>7.2f} "
              f"{areas['interfaces']['area_pct']:>7.2f}  {fases}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target,
                      int(sys.argv[2]) if len(sys.argv) == 3 else None)
