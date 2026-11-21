"""Prueba del Bloque B3 de la Etapa 4: Módulo B completo.

Corre las Etapas 1 a 3 y el Módulo A sin abrir ventanas, y después
pipeline_compositional.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/b3_pipeline.py IMAGEN [FONDO]
    uv run python scripts/fases/b3_pipeline.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre la figura composicional.
Con una carpeta: genera todas las figuras sin abrir ventanas e imprime
una tabla (las 22 imágenes de prueba tardan unos 6-7 minutos).

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
from tem.compositional import (  # noqa: E402
    _display_name,
    _width_text,
    pipeline_compositional,
)
from tem.fraction import phase_names, pipeline_fraction  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402


def procesar(path: Path, fondo: int | None, show: bool) -> dict:
    """Corre las Etapas 1 a 3 y el Módulo A en silencio, y el Módulo B.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
        show: Si es True, muestra la figura y los mensajes del
            Módulo B.

    Returns:
        Resultado de pipeline_compositional.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        _, names = phase_names(seg["mask"], fondo)
        mor = pipeline_morphology(seg, phases=list(names.values()),
                                  show=False)
        fra = pipeline_fraction(mor, background=fondo, show=False)
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print(f"=== Módulo B: pipeline_compositional — {path.name} ===")
        return pipeline_compositional(fra, show=show)


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
    print(f"{'archivo':36} {'par':18} {'contraste':>9} {'ancho':>22} "
          f"{'inhom.':>6} {'seg':>5}")
    for path in files:
        t = time.perf_counter()
        r = procesar(path, None, show=False)
        seg = time.perf_counter() - t
        n_inhom = sum(d["n_alta"] + d["n_baja"]
                      for d in r["inhomogeneities"].values())
        for key, c in r["intensity_profiles"]["contraste"].items():
            g = r["interface_gradients"].get(key)
            ancho = _width_text(g) if g else "-"
            par = "/".join(_display_name(n) for n in key.split("-"))
            print(f"{path.name:36} {par:18} {c['indice']:>9.2f} "
                  f"{ancho:>22} {n_inhom:>6} {seg:>5.1f}")
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
