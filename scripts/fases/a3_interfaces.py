"""Prueba del Bloque A3 de la Etapa 4: fracción de interfaces.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/a3_interfaces.py IMAGEN [FONDO]
    uv run python scripts/fases/a3_interfaces.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: imprime la tabla de pares y abre una figura con las
interfaces coloreadas por par sobre la imagen, más un recorte en
tamaño real de la zona con más interfaces.
Con una carpeta: imprime una fila por imagen.

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
from scipy import ndimage  # noqa: E402

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.fraction import (  # noqa: E402
    _interface_neighbors,
    compute_interface_fraction,
    phase_title,
)
from tem.loader import SUPPORTED_FORMATS  # noqa: E402

# Colores de los pares (en orden de longitud) y de los contactos
COLORES_PARES = ["tab:red", "tab:blue", "tab:green", "tab:orange",
                 "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
COLOR_CONTACTOS = "yellow"
LADO_RECORTE = 300


def mascara(path: Path) -> dict:
    """Corre las Etapas 1 y 2 sin abrir ventanas ni imprimir.

    Args:
        path: Ruta a la imagen.

    Returns:
        Resultado de pipeline_segmentation.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return pipeline_segmentation(pipeline_tem(path, show=False),
                                     show=False)


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Muestra las interfaces de una imagen coloreadas por par.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg = mascara(path)
    print(f"{path.name}\n\n=== compute_interface_fraction ===")
    lineas = seg["lineas_watershed"]
    result = compute_interface_fraction(seg["mask"], fondo, lineas)
    _, reach = _interface_neighbors(seg["mask"], fondo, lineas)

    # Imagen RGB: gris de fondo y cada par en su color
    gris = seg["original"]
    rgb = np.dstack([gris, gris, gris])
    leyenda = []
    for color, key in zip(COLORES_PARES, result["pares_adyacentes"]):
        a, b = key.split("-")
        pixeles = reach[a] & reach[b]
        rgb[pixeles] = plt.matplotlib.colors.to_rgb(color)
        leyenda.append((color, f"{phase_title(a)} / {phase_title(b)}"))
    # Contactos entre partículas: las líneas de corte del watershed
    rgb[lineas] = plt.matplotlib.colors.to_rgb(COLOR_CONTACTOS)
    if lineas.any():
        leyenda.append((COLOR_CONTACTOS, "contactos misma fase"))

    # Recorte: zona con más interfaces
    lado = min(LADO_RECORTE, *gris.shape)
    dens = ndimage.uniform_filter((seg["mask"] == -1).astype(float), lado)
    medio = lado // 2
    dens[:medio], dens[-medio:] = -1, -1
    dens[:, :medio], dens[:, -medio:] = -1, -1
    f, c = np.unravel_index(dens.argmax(), dens.shape)
    filas = slice(f - medio, f - medio + lado)
    cols = slice(c - medio, c - medio + lado)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(rgb)
    axes[0].add_patch(plt.Rectangle((cols.start, filas.start), lado, lado,
                                    fill=False, edgecolor="white"))
    axes[0].set_title(f"Interfaces: {result['porcentaje_total_interfaces']:.2f}"
                      " % de la imagen")
    axes[1].imshow(rgb[filas, cols], interpolation="nearest")
    axes[1].set_title("Recorte en tamaño real (zona con más interfaces)")
    for color, texto in leyenda:
        axes[1].plot([], [], "s", color=color, label=texto)
    if leyenda:
        axes[1].legend(loc="lower right")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Bloque A3 — {path.name}", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume las interfaces de todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'interf%':>7} {'pares':>5}  {'par principal':24} "
          f"{'contactos px':>12}")
    for path in files:
        seg = mascara(path)
        with contextlib.redirect_stdout(io.StringIO()):
            r = compute_interface_fraction(
                seg["mask"], watershed_lines=seg["lineas_watershed"]
            )
        pares = r["pares_adyacentes"]
        if pares:
            key, p = next(iter(pares.items()))
            a, b = key.split("-")
            principal = (f"{phase_title(a)}/{phase_title(b)} "
                         f"({p['fraccion']:.0%})")
        else:
            principal = "-"
        contactos = sum(c["longitud_px"]
                        for c in r["contactos_misma_fase"].values())
        print(f"{path.name:36} {r['porcentaje_total_interfaces']:>7.2f} "
              f"{len(pares):>5}  {principal:24} {contactos:>12.0f}")


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
