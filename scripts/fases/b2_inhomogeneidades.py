"""Prueba del Bloque B2 de la Etapa 4: mapas e inhomogeneidades.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/b2_inhomogeneidades.py IMAGEN [FONDO]
    uv run python scripts/fases/b2_inhomogeneidades.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre una figura con el mapa de z-score de toda la
imagen y la imagen original con las zonas anómalas contorneadas (rojo:
alta intensidad, azul: baja), e imprime la lista de inhomogeneidades.
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

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.compositional import (  # noqa: E402
    build_phase_intensity_map,
    detect_inhomogeneities,
)
from tem.fraction import phase_title  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402


def nombre(clave: str) -> str:
    """'fondo' → 'Fondo', 'fase_1' → 'Fase 1'."""
    return phase_title(clave).replace("fondo", "Fondo")


def segmentar(path: Path) -> dict:
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
    """Muestra el mapa de z-score y las inhomogeneidades de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg = segmentar(path)
    image, mask = seg["processed"], seg["mask"]
    print(f"{path.name}\n\n=== build_phase_intensity_map ===")
    mapas = build_phase_intensity_map(image, mask, fondo)
    for clave, m in mapas.items():
        if clave != "zscore_global":
            print(f"  {nombre(clave)}: media {m['media']:.3f}, desvío "
                  f"{m['std']:.3f}")
    print("\n=== detect_inhomogeneities ===")
    inhom = detect_inhomogeneities(image, mask, fondo)
    for clave, d in inhom.items():
        for tipo in ("alta", "baja"):
            for z in d[tipo]:
                print(f"    {nombre(clave)} {tipo}: {z['area_px']} px en "
                      f"(y={z['centroide'][0]:.0f}, x="
                      f"{z['centroide'][1]:.0f}), z medio "
                      f"{z['z_medio']:+.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    shown = axes[0].imshow(mapas["zscore_global"], cmap="RdBu_r",
                           vmin=-3, vmax=3, interpolation="nearest")
    fig.colorbar(shown, ax=axes[0], fraction=0.046, pad=0.04,
                 label="z-score respecto de su fase")
    axes[0].set_title("z-score de cada píxel respecto de su fase")
    axes[1].imshow(seg["original"], cmap="gray")
    for clave, d in inhom.items():
        for tipo, color in (("alta", "red"), ("baja", "blue")):
            for z in d[tipo]:
                y0, x0, y1, x1 = z["bbox"]
                axes[1].add_patch(plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False,
                    edgecolor=color, linewidth=1.5))
    axes[1].plot([], [], "s", color="red", label="alta intensidad")
    axes[1].plot([], [], "s", color="blue", label="baja intensidad")
    axes[1].legend(loc="lower right")
    total = sum(d["n_alta"] + d["n_baja"] for d in inhom.values())
    axes[1].set_title(f"Inhomogeneidades detectadas: {total}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Bloque B2 — {path.name}", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume las inhomogeneidades de todas las imágenes.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'altas':>5} {'bajas':>5} "
          f"{'% anómalo':>9}")
    for path in files:
        seg = segmentar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            inhom = detect_inhomogeneities(seg["processed"], seg["mask"])
        for clave, d in inhom.items():
            print(f"{path.name:36} {nombre(clave):7} {d['n_alta']:>5} "
                  f"{d['n_baja']:>5} {d['area_anomala_pct']:>9.2f}")


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
