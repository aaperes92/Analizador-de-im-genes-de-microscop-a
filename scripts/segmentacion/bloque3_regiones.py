"""Prueba del Bloque 3: separación y limpieza de regiones.

Uso (desde la raíz del proyecto):
    uv run python scripts/segmentacion/bloque3_regiones.py IMAGEN
    uv run python scripts/segmentacion/bloque3_regiones.py CARPETA

Con una imagen: corre los Bloques 1 y 2 y después los cuatro pasos del
Bloque 3, y abre una figura con la máscara después de cada paso
(arriba) y un recorte en tamaño real de la misma zona (abajo).
Con una carpeta: imprime una tabla resumen por imagen.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import time
from pathlib import Path

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

from tem import load_tem_image, preprocess_tem_image  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import (  # noqa: E402
    apply_segmentation,
    clean_mask,
    detect_interfaces,
    detect_n_phases,
    select_threshold_method,
    separate_touching_regions,
)

# Colores de las fases (de la más oscura a la más clara); las -1
# (interfaces y cortes) se dibujan en blanco
COLORES_FASES = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
]
# Lado del recorte en tamaño real (píxeles)
LADO_RECORTE = 300


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


def _correr_bloque3(image: np.ndarray) -> dict[str, np.ndarray]:
    """Corre los Bloques 1 y 2 y los cuatro pasos del Bloque 3.

    Args:
        image: Imagen preprocesada.

    Returns:
        Diccionario con la máscara después de cada paso.
    """
    print("\n=== Bloques 1 y 2 ===")
    n_phases = detect_n_phases(image)
    _, umbrales = select_threshold_method(image, n_phases)

    pasos: dict[str, np.ndarray] = {}
    print("\n=== 3a apply_segmentation ===")
    pasos["1. Umbrales aplicados"] = apply_segmentation(image, umbrales)
    print("\n=== 3b clean_mask ===")
    t = time.perf_counter()
    pasos["2. Limpieza"] = clean_mask(pasos["1. Umbrales aplicados"])
    print(f"  ({time.perf_counter() - t:.1f} s)")
    print("\n=== 3c separate_touching_regions ===")
    t = time.perf_counter()
    pasos["3. Separación (watershed)"] = separate_touching_regions(
        pasos["2. Limpieza"]
    )
    print(f"  ({time.perf_counter() - t:.1f} s)")
    print("\n=== 3d detect_interfaces ===")
    pasos["4. Interfaces"] = detect_interfaces(
        pasos["3. Separación (watershed)"]
    )
    return pasos


def _colorear(mask: np.ndarray, n: int) -> tuple[np.ndarray, ListedColormap]:
    """Prepara una máscara para dibujarla con un color por fase.

    Args:
        mask: Etiquetas -1, 0, 1, ...
        n: Número de fases.

    Returns:
        Tupla (índices de color, mapa de colores): 0 = blanco para las
        -1 y 1..n para las fases.
    """
    cmap = ListedColormap(["white", *COLORES_FASES[:n]])
    return mask + 1, cmap


def _zona_recorte(final: np.ndarray) -> tuple[slice, slice]:
    """Elige la zona con más interfaces y cortes para el recorte.

    Args:
        final: Máscara final (con -1).

    Returns:
        Slices (filas, columnas) del recorte.
    """
    lado = min(LADO_RECORTE, *final.shape)
    densidad = ndimage.uniform_filter(
        (final == -1).astype(np.float32), size=lado
    )
    # Se descartan los centros que harían salir el recorte de la imagen
    medio = lado // 2
    densidad[:medio, :] = densidad[-medio:, :] = -1
    densidad[:, :medio] = densidad[:, -medio:] = -1
    fila, col = np.unravel_index(np.argmax(densidad), densidad.shape)
    fila0, col0 = max(0, fila - medio), max(0, col - medio)
    return slice(fila0, fila0 + lado), slice(col0, col0 + lado)


def probar_imagen(path: Path) -> None:
    """Corre el Bloque 3 sobre una imagen y muestra cada paso.

    Args:
        path: Ruta a la imagen.
    """
    image = _cargar(path)
    pasos = _correr_bloque3(image)
    final = pasos["4. Interfaces"]
    n = int(final.max()) + 1
    filas, cols = _zona_recorte(final)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    for col, (titulo, mask) in enumerate(pasos.items()):
        datos, cmap = _colorear(mask, n)
        for fila, recorte in ((0, (slice(None), slice(None))),
                              (1, (filas, cols))):
            ax = axes[fila, col]
            ax.imshow(datos[recorte], cmap=cmap, vmin=-0.5,
                      vmax=n + 0.5, interpolation="nearest")
            ax.axis("off")
        axes[0, col].set_title(titulo)
        axes[1, col].set_title(
            f"Recorte {filas.start}:{filas.stop}, "
            f"{cols.start}:{cols.stop} (tamaño real)", fontsize=9
        )
        # Marca la zona del recorte en la vista completa
        axes[0, col].add_patch(plt.Rectangle(
            (cols.start, filas.start), cols.stop - cols.start,
            filas.stop - filas.start, fill=False, edgecolor="black",
            linewidth=1.5,
        ))

    fig.suptitle(
        f"Bloque 3 — {path.name}: {n} fases "
        "(blanco = interfaces y cortes)", fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Corre el Bloque 3 sobre todas las imágenes de una carpeta.

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
    print(f"{'archivo':16} {'fases':>5} {'eliminadas':>10} "
          f"{'watershed':>12} {'interf.':>7} {'cortes':>6}")
    for path in files:
        image = _cargar(path, silencioso=True)
        texto = io.StringIO()
        with contextlib.redirect_stdout(texto):
            pasos = _correr_bloque3(image)
        log = texto.getvalue()
        final = pasos["4. Interfaces"]
        eliminadas = sum(
            int(x) for x in re.findall(r"(\d+) región\(es\) chica", log)
        )
        ws = re.search(r"Watershed aplicado en \d+ fase\(s\): (\[.*?\])",
                       log)
        lineas = float(np.mean(pasos["3. Separación (watershed)"] == -1))
        interf = float(np.mean(final == -1)) - lineas
        print(f"{path.name:16} {int(final.max()) + 1:>5} {eliminadas:>10} "
              f"{ws.group(1) if ws else '-':>12} {interf:>7.2%} "
              f"{lineas:>6.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
