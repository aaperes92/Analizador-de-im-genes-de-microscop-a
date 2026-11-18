"""Prueba del Bloque A2 de la Etapa 4: densidad numérica.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/a2_densidad.py IMAGEN [FONDO]
    uv run python scripts/fases/a2_densidad.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: corre las Etapas 1 a 3 sin abrir ventanas, muestra la
tabla de densidad y abre una figura con los centroides unidos a su
vecino más cercano.
Con una carpeta: imprime una fila por imagen y fase.

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
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from tem import (  # noqa: E402
    extract_region_properties,
    pipeline_segmentation,
    pipeline_tem,
)
from tem.fraction import (  # noqa: E402
    compute_number_density,
    phase_names,
    phase_title,
)
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import PHASE_COLORS  # noqa: E402


def preparar(path: Path, fondo: int | None) -> tuple[dict, dict]:
    """Corre las Etapas 1 y 2 y el Bloque 1 de la Etapa 3, en silencio.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).

    Returns:
        Tupla (resultado de pipeline_segmentation, props).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        background, names = phase_names(seg["mask"], fondo)
        props = extract_region_properties(seg["mask"], seg["processed"],
                                          phases=list(names.values()))
    return seg, props


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Calcula la densidad de una imagen y la grafica.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg, props = preparar(path, fondo)
    print(f"{path.name}\n\n=== compute_number_density ===")
    density = compute_number_density(props, seg["mask"], fondo)

    _, names = phase_names(seg["mask"], fondo)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(seg["original"], cmap="gray")
    for name, label in names.items():
        regiones = props.get(f"fase_{label}", [])
        if len(regiones) < 2:
            continue
        color = PHASE_COLORS[label % len(PHASE_COLORS)]
        c = np.array([r["centroide"] for r in regiones])
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        vecino = d.argmin(axis=1)
        for i, j in enumerate(vecino):
            ax.plot([c[i, 1], c[j, 1]], [c[i, 0], c[j, 0]], color=color,
                    linewidth=0.8)
        ax.plot(c[:, 1], c[:, 0], "o", color=color, markersize=3,
                label=f"{phase_title(name)}: n={density[name]['n_total']}, "
                      f"R={density[name]['indice_clark_evans']:.2f}")
    ax.set_title(f"Bloque A2 — {path.name}\nCada centroide unido a su "
                 "vecino más cercano (R = índice de Clark-Evans)")
    ax.legend(loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume la densidad de todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'dens/1000px²':>12} "
          f"{'sep.':>6} {'azar':>6} {'R':>5}")
    for path in files:
        seg, props = preparar(path, None)
        with contextlib.redirect_stdout(io.StringIO()):
            density = compute_number_density(props, seg["mask"])
        for name, d in density.items():
            if not d["n_total"] or d["separacion_media_px"] is None:
                print(f"{path.name:36} {phase_title(name):7} "
                      f"{d['n_total'] or 0:>4}  (menos de 2 regiones)")
                continue
            print(f"{path.name:36} {phase_title(name):7} {d['n_total']:>4} "
                  f"{d['densidad_por_1000px2']:>12.4f} "
                  f"{d['separacion_media_px']:>6.0f} "
                  f"{d['separacion_esperada_azar_px']:>6.0f} "
                  f"{d['indice_clark_evans']:>5.2f}")


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
