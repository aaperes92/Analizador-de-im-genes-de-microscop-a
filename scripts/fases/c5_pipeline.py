"""Prueba del Bloque C5 de la Etapa 4: Módulo C completo.

Corre las Etapas 1 a 3 y los Módulos A y B sin abrir ventanas, y
después pipeline_spatial.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/c5_pipeline.py IMAGEN [FONDO]
    uv run python scripts/fases/c5_pipeline.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre la figura espacial.
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
from tem.compositional import pipeline_compositional  # noqa: E402
from tem.fraction import (  # noqa: E402
    phase_names,
    phase_title,
    pipeline_fraction,
)
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.spatial import pipeline_spatial  # noqa: E402


def procesar(path: Path, fondo: int | None, show: bool) -> dict:
    """Corre todo en silencio hasta el Módulo B, y después el Módulo C.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
        show: Si es True, muestra la figura y los mensajes del
            Módulo C.

    Returns:
        Resultado de pipeline_spatial.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        _, names = phase_names(seg["mask"], fondo)
        mor = pipeline_morphology(seg, phases=list(names.values()),
                                  show=False)
        fra = pipeline_fraction(mor, background=fondo, show=False)
        com = pipeline_compositional(fra, show=False)
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print(f"=== Módulo C: pipeline_spatial — {path.name} ===")
        return pipeline_spatial(com, show=show)


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
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'RDF':>12} "
          f"{'densidad':>16} {'Hopkins':>12} {'grad.':>5} {'seg':>5}")
    for path in files:
        t = time.perf_counter()
        r = procesar(path, None, show=False)
        seg = time.perf_counter() - t
        for name, rdf in r["rdf"].items():
            dens = r["density_map"][name]["clasificacion"]
            dens = dens.replace(" (compatible con azar)", "*")
            hop = r["clustering"][name]["hopkins_clasificacion"]
            n_g = len(r["spatial_gradients"][name][
                "gradientes_significativos"])
            print(f"{path.name:36} {phase_title(name):7} {rdf['n']:>4} "
                  f"{rdf['clasificacion']:>12} {dens[:16]:>16} "
                  f"{hop:>12} {n_g:>5} {seg:>5.1f}")
    print("* = compatible con una distribución al azar")
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
