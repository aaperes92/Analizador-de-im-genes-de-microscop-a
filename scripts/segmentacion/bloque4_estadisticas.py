"""Prueba del Bloque 4: estadísticas por fase.

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque4_estadisticas.py IMAGEN
    uv run python scripts/segmentacion/bloque4_estadisticas.py CARPETA

Con una imagen: corre los Bloques 1 a 3, calcula las estadísticas y
abre una figura con la máscara final, el área de cada fase y la
distribución de tamaños de sus regiones.
Con una carpeta: imprime una fila resumen por imagen.

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
from skimage import measure  # noqa: E402

from tem import load_tem_image, preprocess_tem_image  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    CLEAN_MIN_FRACTION,
    apply_segmentation,
    clean_mask,
    compute_phase_stats,
    detect_interfaces,
    detect_n_phases,
    select_threshold_method,
    separate_touching_regions,
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


def _mascara_final(image: np.ndarray, silencioso: bool) -> np.ndarray:
    """Corre los Bloques 1 a 3 y devuelve la máscara final.

    Args:
        image: Imagen preprocesada.
        silencioso: Si es True, oculta los mensajes.

    Returns:
        Máscara con -1 = interfaces y 0, 1, ... = fases.
    """
    salida = io.StringIO() if silencioso else sys.stdout
    with contextlib.redirect_stdout(salida):
        print("\n=== Bloques 1 a 3 ===")
        n_phases = detect_n_phases(image)
        _, umbrales = select_threshold_method(image, n_phases)
        mask = apply_segmentation(image, umbrales)
        mask = separate_touching_regions(clean_mask(mask))
        return detect_interfaces(mask)


def probar_imagen(path: Path) -> None:
    """Calcula las estadísticas de una imagen y las grafica.

    Args:
        path: Ruta a la imagen.
    """
    image = _cargar(path)
    mask = _mascara_final(image, silencioso=False)
    print("\n=== compute_phase_stats ===")
    stats = compute_phase_stats(image, mask)
    fases = [k for k in stats if k != "interfaces"]
    suma = sum(stats[k]["area_pct"] for k in fases)
    print(f"\nSuma de porcentajes de las fases: {suma:.6f} %")

    n = len(fases)
    colores = COLORES_FASES[:n]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].imshow(mask + 1, cmap=ListedColormap(["white", *colores]),
                   vmin=-0.5, vmax=n + 0.5, interpolation="nearest")
    axes[0].set_title("Máscara final (blanco = interfaces)")
    axes[0].axis("off")

    pct = [stats[k]["area_pct"] for k in fases]
    barras = axes[1].bar(fases, pct, color=colores)
    axes[1].bar_label(barras, fmt="%.1f %%")
    axes[1].set_ylabel("% del área (sin interfaces)")
    axes[1].set_title(
        f"Área por fase (interfaces: "
        f"{stats['interfaces']['area_pct']:.2f} % del total)"
    )

    for k, color in zip(fases, colores):
        etiqueta = int(k.split("_")[1])
        labels = measure.label(mask == etiqueta, connectivity=1)
        sizes = np.bincount(labels.ravel())[1:]
        # Mismo criterio que compute_phase_stats: sin fragmentos
        sizes = sizes[sizes >= max(1, round(CLEAN_MIN_FRACTION * mask.size))]
        if sizes.size:
            bins = np.logspace(np.log10(sizes.min()) - 0.1,
                               np.log10(sizes.max()) + 0.1, 30)
            axes[2].hist(sizes, bins=bins, color=color, alpha=0.6,
                         label=f"{k} ({sizes.size} regiones, "
                               f"{stats[k]['n_fragmentos']} fragm.)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Tamaño de la región (px, escala log)")
    axes[2].set_ylabel("Cantidad de regiones")
    axes[2].set_title("Distribución de tamaños de regiones")
    axes[2].legend()

    fig.suptitle(f"Bloque 4 — {path.name}: {n} fases", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Calcula las estadísticas de todas las imágenes de una carpeta.

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
    print(f"{'archivo':16} {'interf.%':>8}  "
          "fases (área % / regiones / fragmentos)")
    for path in files:
        image = _cargar(path, silencioso=True)
        mask = _mascara_final(image, silencioso=True)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = compute_phase_stats(image, mask)
        fases = "  ".join(
            f"{k[5:]}: {s['area_pct']:5.1f}/{s['n_regiones']}/"
            f"{s['n_fragmentos']:<4}"
            for k, s in stats.items() if k != "interfaces"
        )
        print(f"{path.name:16} {stats['interfaces']['area_pct']:>8.2f}  "
              f"{fases}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
