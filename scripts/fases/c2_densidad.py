"""Prueba del Bloque C2 de la Etapa 4: mapa de densidad 2D.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/c2_densidad.py IMAGEN [FONDO]
    uv run python scripts/fases/c2_densidad.py CARPETA
    uv run python scripts/fases/c2_densidad.py --sintetico

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: muestra el mapa de densidad suavizado sobre la
micrografía, con la zona de mayor concentración (cruz) y la de menor
(círculo).
Con una carpeta: imprime una fila por imagen y fase.
Con --sintetico: compara partículas al azar con partículas
concentradas en una mitad de la imagen.

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
from skimage.draw import disk  # noqa: E402

from tem import (  # noqa: E402
    extract_region_properties,
    pipeline_segmentation,
    pipeline_tem,
)
from tem.fraction import phase_names, phase_title  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.spatial import compute_density_map  # noqa: E402


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
        _, names = phase_names(seg["mask"], fondo)
        props = extract_region_properties(seg["mask"], seg["processed"],
                                          phases=list(names.values()))
    return seg, props


def graficar(ax: plt.Axes, imagen: np.ndarray, d: dict,
             titulo: str) -> None:
    """Dibuja el mapa de densidad suavizado sobre la imagen.

    Args:
        ax: Eje donde dibujar.
        imagen: Imagen de fondo.
        d: Resultado de compute_density_map para una fase.
        titulo: Título del gráfico.
    """
    alto, ancho = imagen.shape
    ax.imshow(imagen, cmap="gray")
    if d["mapa_suavizado"] is not None:
        shown = ax.imshow(d["mapa_suavizado"], cmap="hot", alpha=0.5,
                          extent=(0, ancho, alto, 0),
                          interpolation="bilinear")
        plt.colorbar(shown, ax=ax, fraction=0.046, pad=0.04,
                     label="Partículas por celda (suavizado)")
        ax.plot(d["posicion_max_px"][1], d["posicion_max_px"][0], "x",
                color="cyan", markersize=14, mew=3, label="máxima")
        ax.plot(d["posicion_min_px"][1], d["posicion_min_px"][0], "o",
                color="cyan", markersize=12, mfc="none", mew=2,
                label="mínima")
        ax.legend(loc="lower right")
        cv = f"CV={d['cv']:.2f} (azar ≤{d['cv_azar_p975']:.2f})"
    else:
        cv = "sin partículas"
    ax.set_title(f"{titulo}\n{d['n']} partículas, grilla "
                 f"{d['grid_size']}×{d['grid_size']}, {cv}\n"
                 f"→ {d['clasificacion']}", fontsize=10)
    ax.axis("off")


def mascara_sintetica(concentrada: bool) -> np.ndarray:
    """Máscara de 1200 x 1200 con 150 círculos.

    Args:
        concentrada: Si es True, el 85 % de los círculos va en la mitad
            izquierda; si no, todos al azar.

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    rng = np.random.default_rng(5)
    mask = np.ones((1200, 1200), dtype=np.int32)
    centros: list[np.ndarray] = []
    while len(centros) < 150:
        izquierda = concentrada and rng.random() < 0.85
        c = rng.uniform([15, 15], [1185, 585 if izquierda else 1185])
        if all(np.hypot(*(c - o)) > 22 for o in centros):
            centros.append(c)
    for c in centros:
        rr, cc = disk(tuple(c), 10, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def probar_sintetico() -> None:
    """Compara una distribución al azar con una concentrada."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, concentrada in zip(axes, (False, True)):
        mask = mascara_sintetica(concentrada)
        titulo = "concentradas a la izquierda" if concentrada else "al azar"
        with contextlib.redirect_stdout(io.StringIO()):
            props = extract_region_properties(mask, np.zeros(mask.shape))
        print(f"=== {titulo}")
        d = compute_density_map(props, mask)["fase_1"]
        graficar(ax, np.where(mask == 0, 0.2, 0.8), d, titulo)
    fig.suptitle("Bloque C2 — mapas de densidad sintéticos", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Calcula y muestra el mapa de densidad de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg, props = preparar(path, fondo)
    print(f"{path.name}\n\n=== compute_density_map ===")
    mapas = compute_density_map(props, seg["mask"], background=fondo)
    fig, axes = plt.subplots(1, len(mapas), figsize=(9 * len(mapas), 9),
                             squeeze=False)
    for ax, (name, d) in zip(axes[0], mapas.items()):
        graficar(ax, seg["original"], d, f"{path.name} — "
                                         f"{phase_title(name)}")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el mapa de densidad de todas las imágenes.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'grilla':>6} {'CV':>5} "
          f"{'azar p97.5':>10}  clasificación")
    for path in files:
        seg, props = preparar(path, None)
        with contextlib.redirect_stdout(io.StringIO()):
            mapas = compute_density_map(props, seg["mask"])
        for name, d in mapas.items():
            cv = "-" if d["cv"] is None else f"{d['cv']:.2f}"
            p = "-" if d["cv_azar_p975"] is None else (
                f"{d['cv_azar_p975']:.2f}")
            print(f"{path.name:36} {phase_title(name):7} {d['n']:>4} "
                  f"{d['grid_size']:>3}×{d['grid_size']:<2} {cv:>5} "
                  f"{p:>10}  {d['clasificacion']}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--sintetico":
        probar_sintetico()
        sys.exit(0)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target,
                      int(sys.argv[2]) if len(sys.argv) == 3 else None)
