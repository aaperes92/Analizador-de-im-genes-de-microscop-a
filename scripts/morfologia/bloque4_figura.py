"""Prueba del Bloque 4 de la Etapa 3: figura morfológica.

Uso (desde la raíz del proyecto):
    uv run python scripts/morfologia/bloque4_figura.py IMAGEN
    uv run python scripts/morfologia/bloque4_figura.py CARPETA

Con una imagen: corre todo hasta el Bloque 3 y abre la figura (que
también se guarda en data/samples/).
Con una carpeta: genera las figuras de todas las imágenes sin abrir
ventanas.

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
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.morphology import (  # noqa: E402
    analyze_shape,
    analyze_size_distribution,
    extract_region_properties,
    show_morphology_results,
)


def generar_figura(path: Path, show: bool) -> Path:
    """Corre las Etapas 1 y 2 y los Bloques 1, 2 y 4 de la Etapa 3.

    Args:
        path: Ruta a la imagen.
        show: Si es True, abre la ventana y muestra los mensajes.

    Returns:
        Ruta del PNG guardado.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        props = extract_region_properties(seg["mask"], seg["processed"])
        sizes = analyze_size_distribution(props)
        shapes = analyze_shape(props)
        print("\n=== show_morphology_results ===")
        return show_morphology_results(seg["original"], seg["mask"],
                                       props, sizes, shapes, path,
                                       show=show)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        for p in sorted(target.iterdir()):
            if p.suffix.lower() in SUPPORTED_FORMATS:
                print(f"{p.name:36} → {generar_figura(p, show=False)}")
    else:
        generar_figura(target, show=True)
