"""Prueba del Bloque 5: figura final de la segmentación.

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque5_figura.py IMAGEN
    uv run python scripts/segmentacion/bloque5_figura.py CARPETA

Con una imagen: corre los Bloques 1 a 4 y abre la figura final (que
también se guarda en data/samples/).
Con una carpeta: genera la figura de cada imagen en data/samples/ sin
abrir ventanas.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tem import load_tem_image, preprocess_tem_image  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    apply_segmentation,
    clean_mask,
    compute_phase_stats,
    detect_interfaces,
    detect_n_phases,
    select_threshold_method,
    separate_touching_regions,
    show_segmentation_results,
)


def generar_figura(path: Path, show: bool) -> Path:
    """Corre los Bloques 1 a 4 y genera la figura del Bloque 5.

    Args:
        path: Ruta a la imagen.
        show: Si es True, abre la ventana de la figura.

    Returns:
        Ruta del PNG guardado.
    """
    salida = sys.stdout if show else io.StringIO()
    with contextlib.redirect_stdout(salida):
        original = load_tem_image(path)
        image = preprocess_tem_image(original, "auto")
        n_phases = detect_n_phases(image)
        _, umbrales = select_threshold_method(image, n_phases)
        mask = apply_segmentation(image, umbrales)
        mask = detect_interfaces(
            separate_touching_regions(clean_mask(mask))
        )
        stats = compute_phase_stats(image, mask)
        print("\n=== show_segmentation_results ===")
        return show_segmentation_results(
            original, mask, stats, path, thresholds=umbrales, show=show
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in SUPPORTED_FORMATS
        )
        for p in files:
            print(f"{p.name:16} → {generar_figura(p, show=False)}")
    else:
        generar_figura(target, show=True)
