"""Prueba del Bloque A4 de la Etapa 4: Módulo A completo.

Corre las Etapas 1 a 3 sin abrir ventanas y después pipeline_fraction.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/a4_pipeline.py IMAGEN [FONDO]
    uv run python scripts/fases/a4_pipeline.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre la figura de fracciones.
Con una carpeta: genera todas las figuras sin abrir ventanas e imprime
una tabla.

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

from tem import (  # noqa: E402
    pipeline_morphology,
    pipeline_segmentation,
    pipeline_tem,
)
from tem.fraction import (  # noqa: E402
    phase_names,
    phase_title,
    pipeline_fraction,
)
from tem.loader import SUPPORTED_FORMATS  # noqa: E402


def procesar(path: Path, fondo: int | None, show: bool) -> dict:
    """Corre las Etapas 1 a 3 en silencio y el Módulo A.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
        show: Si es True, muestra la figura y los mensajes del
            Módulo A.

    Returns:
        Resultado de pipeline_fraction.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        _, names = phase_names(seg["mask"], fondo)
        mor = pipeline_morphology(seg, phases=list(names.values()),
                                  show=False)
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print(f"=== Módulo A: pipeline_fraction — {path.name} ===")
        return pipeline_fraction(mor, background=fondo, show=show)


def procesar_carpeta(folder: Path) -> None:
    """Procesa todas las imágenes de una carpeta e imprime un resumen.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'% total':>7} {'n':>4} "
          f"{'dens/1000px²':>12} {'R':>5} {'interf%':>7} {'seg':>5}")
    for path in files:
        t = time.perf_counter()
        r = procesar(path, None, show=False)
        seg = time.perf_counter() - t
        interf = r["interface_fraction"]["porcentaje_total_interfaces"]
        for name, a in r["area_fractions"].items():
            if not name.startswith("fase_"):
                continue
            d = r["number_density"][name]
            n = "-" if d["n_total"] is None else d["n_total"]
            dens = ("-" if d["densidad_por_1000px2"] is None
                    else f"{d['densidad_por_1000px2']:.4f}")
            rr = ("-" if d["indice_clark_evans"] is None
                  else f"{d['indice_clark_evans']:.2f}")
            print(f"{path.name:36} {phase_title(name):7} "
                  f"{a['area_pct_con_fondo']:>7.2f} {n:>4} {dens:>12} "
                  f"{rr:>5} {interf:>7.2f} {seg:>5.1f}")
    print("Figuras guardadas en data/samples/")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        procesar_carpeta(target)
    else:
        procesar(target, int(sys.argv[2]) if len(sys.argv) == 3 else None,
                 show=True)
