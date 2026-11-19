"""Prueba del Bloque 1: detección automática de fases.

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque1_fases.py IMAGEN
    uv run python scripts/segmentacion/bloque1_fases.py CARPETA

Con una imagen: imprime el análisis y abre una figura con el
histograma, los picos y los valles.
Con una carpeta: imprime una tabla con el número de fases de cada
imagen soportada que contenga.

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
from matplotlib.colors import ListedColormap  # noqa: E402

from tem import load_tem_image, preprocess_tem_image  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    _merge_peaks,
    _valleys_between,
    analyze_histogram,
    detect_n_phases,
)

# Colores de las fases en la vista previa (de la más oscura a la más
# clara)
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


def _picos_finales(info: dict) -> tuple[list[float], list[float]]:
    """Repite la fusión de picos para saber cuáles quedaron.

    Usa las mismas funciones internas que detect_n_phases, así la
    figura muestra exactamente lo que decidió el Bloque 1.

    Args:
        info: Resultado de analyze_histogram.

    Returns:
        Tupla (posiciones de los picos que quedan, posiciones de los
        valles entre ellos), ambas en [0, 1].
    """
    edges = info["bin_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = info["histogram_smooth"]
    peaks = [int(np.argmin(np.abs(centers - p)))
             for p in info["peaks_positions"]]
    if len(peaks) <= 1:
        return [float(centers[p]) for p in peaks], []
    with contextlib.redirect_stdout(io.StringIO()):
        kept, _ = _merge_peaks(smooth, peaks, centers)
    valleys = _valleys_between(smooth, kept)
    return [float(centers[p]) for p in kept], [
        float(centers[v]) for v in valleys
    ]


def probar_imagen(path: Path) -> None:
    """Corre el Bloque 1 sobre una imagen y muestra los resultados.

    La figura tiene la imagen, una vista previa de las fases (cada
    píxel coloreado según el tramo del histograma en que cae) y el
    histograma con los picos que quedan y los que se fusionaron.

    Args:
        path: Ruta a la imagen.
    """
    image = _cargar(path)
    print("\n=== analyze_histogram ===")
    info = analyze_histogram(image)
    print("\n=== detect_n_phases ===")
    n_phases = detect_n_phases(image)

    kept, cortes = _picos_finales(info)
    fusionados = [p for p in info["peaks_positions"] if p not in kept]
    # Vista previa: tramo del histograma de cada píxel (0, 1, 2, ...)
    preview = np.digitize(image, cortes)
    colores = COLORES_FASES[: len(cortes) + 1]

    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    ax_img = fig.add_subplot(grid[0, 0])
    ax_prev = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, :])

    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title("Imagen preprocesada")
    ax_img.axis("off")

    ax_prev.imshow(preview, cmap=ListedColormap(colores),
                   vmin=-0.5, vmax=len(colores) - 0.5,
                   interpolation="nearest")
    ax_prev.set_title(
        "Vista previa de fases (cortes en los valles; aproximada)"
    )
    ax_prev.axis("off")

    edges = info["bin_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    # Fondo del histograma coloreado por tramo, igual que la vista
    # previa
    limites = [0.0, *cortes, 1.0]
    for i, color in enumerate(colores):
        ax_hist.axvspan(limites[i], limites[i + 1], color=color,
                        alpha=0.12)
    ax_hist.bar(centers, info["histogram"], width=edges[1] - edges[0],
                color="lightgray", label="Histograma")
    ax_hist.plot(centers, info["histogram_smooth"], color="black",
                 label="Histograma suavizado")
    for i, pos in enumerate(kept):
        ax_hist.axvline(pos, color="tab:red", linewidth=2,
                        label="Picos que quedan" if i == 0 else None)
    for i, pos in enumerate(fusionados):
        ax_hist.axvline(pos, color="gray", linestyle=":",
                        label="Picos fusionados" if i == 0 else None)
    for i, pos in enumerate(cortes):
        ax_hist.axvline(pos, color="black", linestyle="--",
                        label="Cortes entre fases" if i == 0 else None)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Intensidad normalizada")
    ax_hist.set_ylabel("Cantidad de píxeles")
    ax_hist.legend(loc="upper right")

    fig.suptitle(
        f"Bloque 1 — {path.name}: {info['n_peaks']} pico(s), "
        f"{n_phases} fases detectadas", fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Corre detect_n_phases sobre todas las imágenes de una carpeta.

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
    print(f"{'archivo':20} {'picos':>5} {'fases':>5}  posiciones")
    for path in files:
        image = _cargar(path, silencioso=True)
        with contextlib.redirect_stdout(io.StringIO()):
            info = analyze_histogram(image)
            n_phases = detect_n_phases(image)
        pos = ", ".join(f"{p:.2f}" for p in info["peaks_positions"])
        print(f"{path.name:20} {info['n_peaks']:>5} {n_phases:>5}  [{pos}]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
