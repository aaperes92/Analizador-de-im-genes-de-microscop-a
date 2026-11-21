"""Prueba del Bloque 2: métodos de umbralización.

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque2_umbrales.py IMAGEN
    uv run python scripts/segmentacion/bloque2_umbrales.py CARPETA

Con una imagen: corre los tres métodos (Otsu multinivel, GMM y umbral
local adaptativo) y abre una figura para compararlos, marcando cuál
elegiría el modo automático.
Con una carpeta: imprime una tabla con las fases, la medida de
iluminación, el método elegido y sus umbrales para cada imagen.

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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from tem import load_tem_image, preprocess_tem_image  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    detect_n_phases,
    histogram_skewness,
    illumination_uniformity,
    select_threshold_method,
    threshold_adaptive_local,
    threshold_gmm,
    threshold_otsu_multi,
    threshold_triangle_method,
)

# Colores de las fases (de la más oscura a la más clara)
COLORES_FASES = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
]


def _cargar(path: Path, silencioso: bool = False) -> np.ndarray:
    """Carga y preprocesa la imagen con la Etapa 1.

    Args:
        path: Ruta a la imagen.
        silencioso: Si es True, oculta los mensajes de la Etapa 1.

    Returns:
        Imagen preprocesada en [0, 1].
    """
    salida = io.StringIO() if silencioso else sys.stdout
    with contextlib.redirect_stdout(salida):
        return preprocess_tem_image(load_tem_image(path), "auto")


def _mostrar_fases(ax: plt.Axes, labels: np.ndarray, n: int,
                   titulo: str) -> None:
    """Dibuja una imagen de etiquetas con un color por fase.

    Args:
        ax: Eje donde dibujar.
        labels: Etiquetas 0, 1, ..., n - 1.
        n: Número de fases.
        titulo: Título del panel.
    """
    colores = COLORES_FASES[:n]
    ax.imshow(labels, cmap=ListedColormap(colores), vmin=-0.5,
              vmax=n - 0.5, interpolation="nearest")
    ax.set_title(titulo)
    ax.axis("off")


def probar_imagen(path: Path) -> None:
    """Corre los cuatro métodos sobre una imagen y los compara.

    Args:
        path: Ruta a la imagen.
    """
    image = _cargar(path)

    print("\n=== Bloque 1: detect_n_phases ===")
    n_phases = detect_n_phases(image)

    print("\n=== select_threshold_method (modo auto) ===")
    elegido, _ = select_threshold_method(image, n_phases)

    print("\n=== Los cuatro métodos, para comparar ===")
    tiempos: dict[str, float] = {}
    t = time.perf_counter()
    otsu = threshold_otsu_multi(image, n_phases)
    tiempos["otsu"] = time.perf_counter() - t
    t = time.perf_counter()
    gmm = threshold_gmm(image, n_phases)
    tiempos["gmm"] = time.perf_counter() - t
    t = time.perf_counter()
    triangulo = threshold_triangle_method(image)
    tiempos["triangle"] = time.perf_counter() - t
    t = time.perf_counter()
    adaptive = threshold_adaptive_local(image)
    tiempos["adaptive"] = time.perf_counter() - t
    print("Tiempos: " + ", ".join(
        f"{k} {v:.1f} s" for k, v in tiempos.items()
    ))

    def marca(nombre: str) -> str:
        return "  ← elegido (auto)" if nombre == elegido else ""

    fig = plt.figure(figsize=(20, 9))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.2, 1))
    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title("Imagen preprocesada")
    ax_img.axis("off")

    _mostrar_fases(fig.add_subplot(grid[0, 1]),
                   np.digitize(image, otsu), n_phases,
                   f"Otsu multinivel{marca('otsu')}")
    _mostrar_fases(fig.add_subplot(grid[0, 2]),
                   np.digitize(image, gmm), n_phases,
                   f"GMM{marca('gmm')}")
    _mostrar_fases(fig.add_subplot(grid[0, 3]),
                   np.digitize(image, triangulo), 2,
                   f"Triángulo (2 fases){marca('triangle')}")
    _mostrar_fases(fig.add_subplot(grid[0, 4]),
                   adaptive.astype(int), 2,
                   "Umbral local (solo 2 fases)")

    ax_hist = fig.add_subplot(grid[1, :])
    ax_hist.hist(image.ravel(), bins=256, range=(0, 1),
                 color="lightgray")
    for i, t in enumerate(otsu):
        ax_hist.axvline(t, color="black", linestyle="--",
                        label="Umbrales Otsu" if i == 0 else None)
    for i, t in enumerate(gmm):
        ax_hist.axvline(t, color="tab:red", linestyle=":", linewidth=2,
                        label="Umbrales GMM" if i == 0 else None)
    ax_hist.axvline(triangulo[0], color="tab:green", linestyle="-.",
                    linewidth=2, label="Umbral triángulo")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Intensidad normalizada")
    ax_hist.set_ylabel("Cantidad de píxeles")
    ax_hist.legend(loc="upper right")

    fig.suptitle(
        f"Bloque 2 — {path.name}: {n_phases} fases, asimetría "
        f"{histogram_skewness(image):.2f}, iluminación "
        f"{illumination_uniformity(image):.3f}, auto elige "
        f"'{elegido}'", fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Corre la selección automática sobre todas las imágenes.

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
    print(f"{'archivo':16} {'fases':>5} {'asim.':>6} {'ilumin.':>7} "
          f"{'método':>8}  umbrales")
    for path in files:
        image = _cargar(path, silencioso=True)
        with contextlib.redirect_stdout(io.StringIO()):
            n_phases = detect_n_phases(image)
            metodo, umbrales = select_threshold_method(image, n_phases)
        ilum = illumination_uniformity(image)
        aviso = " (!)" if ilum > 0.15 else ""
        valores = ", ".join(f"{u:.3f}" for u in umbrales)
        print(f"{path.name:16} {n_phases:>5} "
              f"{histogram_skewness(image):>6.2f} {ilum:>7.3f} "
              f"{metodo:>8}  [{valores}]{aviso}")
    print("(!) = la medida de iluminación supera 0.15: revisar la imagen")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
