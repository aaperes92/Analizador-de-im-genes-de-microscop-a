"""Prueba del Bloque 3 de la Etapa 3: orientación.

Uso (desde la raíz del proyecto):
    uv run python scripts/morfologia/bloque3_orientacion.py IMAGEN
    uv run python scripts/morfologia/bloque3_orientacion.py CARPETA
    uv run python scripts/morfologia/bloque3_orientacion.py --sintetico

Con una imagen: imprime la estadística de orientación y abre la rosa.
Con una carpeta: imprime una fila por imagen y fase.
Con --sintetico: compara dos imágenes generadas, una con elipses todas
horizontales (S cerca de 1) y otra con elipses al azar (S chico).

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from skimage.draw import ellipse  # noqa: E402

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.morphology import (  # noqa: E402
    analyze_orientation,
    extract_region_properties,
    plot_orientation_rose,
)


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


def mascara_elipses(alineadas: bool, n: int = 40) -> np.ndarray:
    """Máscara sintética con elipses (fase 0) sobre fondo (fase 1).

    Args:
        alineadas: Si es True, todas horizontales; si no, al azar.
        n: Cantidad de elipses (en una grilla, sin tocarse).

    Returns:
        Máscara int32 de 600 x 600.
    """
    rng = np.random.default_rng(7)
    mask = np.ones((600, 600), dtype=np.int32)
    lado = int(np.ceil(np.sqrt(n)))
    paso = 600 // lado
    for i in range(n):
        cy = paso // 2 + (i // lado) * paso
        cx = paso // 2 + (i % lado) * paso
        angulo = 0.0 if alineadas else rng.uniform(0, np.pi)
        rr, cc = ellipse(cy, cx, 8, 30, shape=mask.shape,
                         rotation=angulo)
        mask[rr, cc] = 0
    return mask


def probar_sintetico() -> None:
    """Compara orientación alineada contra orientación al azar."""
    for alineadas, nombre in ((True, "sintetico_alineadas"),
                              (False, "sintetico_azar")):
        print(f"\n=== {nombre} ===")
        mask = mascara_elipses(alineadas)
        props = extract_region_properties(mask, np.zeros(mask.shape))
        stats = analyze_orientation(props)
        plot_orientation_rose(props, f"{nombre}.png", stats)


def probar_imagen(path: Path) -> None:
    """Analiza la orientación de una imagen y muestra la rosa.

    Args:
        path: Ruta a la imagen.
    """
    props = propiedades(path, silencioso=False)
    print("\n=== analyze_orientation ===")
    stats = analyze_orientation(props)
    print("\n=== plot_orientation_rose ===")
    plot_orientation_rose(props, path, stats)


def probar_carpeta(folder: Path) -> None:
    """Resume la orientación de todas las imágenes de una carpeta.

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
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'medio°':>6} {'S':>5} "
          f"{'S azar':>6}  clasificación")
    for path in files:
        props = propiedades(path, silencioso=True)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = analyze_orientation(props)
        for fase, s in stats.items():
            if s["indice_orden_S"] is None:
                print(f"{path.name:36} {fase:7} {0:>4}  (sin regiones)")
                continue
            print(f"{path.name:36} {fase:7} {s['n_regiones']:>4} "
                  f"{s['angulo_medio_deg']:>6.0f} "
                  f"{s['indice_orden_S']:>5.2f} "
                  f"{s['S_esperado_aleatorio']:>6.2f}  "
                  f"{s['clasificacion']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--sintetico":
        probar_sintetico()
        sys.exit(0)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
