"""Prueba del Bloque 2 de la Etapa 4.5: centro del haz y preprocesamiento.

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b2_centro.py IMAGEN
    uv run python scripts/difraccion/b2_centro.py CARPETA

Con una imagen:
    1. Imprime los cuatro métodos de find_beam_center y el elegido.
    2. Abre la ventana de correct_center_interactive: "Aceptar centro"
       o "Corregir manualmente" (click en el centro y "Confirmar"). Los
       círculos verdes son los anillos detectados: con el centro bien
       puesto caen sobre los anillos.
    3. Muestra la imagen con los cuatro centros candidatos y la imagen
       preprocesada, con lo enmascarado (haz directo y beam stopper) en
       rojo.
Con una carpeta: imprime una fila por imagen (sin abrir ventanas).

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tem.diffraction import (  # noqa: E402
    _center_candidates,
    _downsample,
    _log_view,
    _pattern_features,
    _to_small,
    correct_center_interactive,
    find_beam_center,
    preprocess_diffraction,
)
from tem.loader import SUPPORTED_FORMATS, load_tem_image  # noqa: E402

COLORES = {"simetria": "yellow", "nitidez": "lime",
           "centro_masa": "magenta", "gaussiana": "cyan"}


def cargar(path: Path) -> np.ndarray:
    """Carga la imagen sin imprimir nada."""
    with contextlib.redirect_stdout(io.StringIO()):
        return load_tem_image(path)


def probar_imagen(path: Path) -> None:
    """Corre el Bloque 2 sobre una imagen y muestra las figuras.

    Args:
        path: Ruta a la imagen.
    """
    image = cargar(path)
    print("=== find_beam_center ===")
    center = find_beam_center(image)
    info = _center_candidates(image)

    print("\n=== correct_center_interactive (ventana) ===")
    rings = _pattern_features(image, center)
    guides = [r for r, c in zip(rings["radios_anillos"],
                                rings["coberturas"]) if c >= 0.5]
    center = correct_center_interactive(image, center, guide_radii=guides)
    print(f"Centro final: (y, x) = ({center[0]:.1f}, {center[1]:.1f}) px")

    print("\n=== preprocess_diffraction ===")
    prep = preprocess_diffraction(image, center)
    graficar(path, image, info, center, prep)


def graficar(path: Path, image: np.ndarray, info: dict,
             center: tuple[float, float], prep: np.ndarray) -> None:
    """Candidatos de centro y resultado del preprocesamiento.

    Args:
        path: Ruta a la imagen (para el título).
        image: Imagen normalizada.
        info: Resultado de _center_candidates.
        center: Centro final.
        prep: Imagen preprocesada (NaN en lo enmascarado).
    """
    small, f = _downsample(image, 1024)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    ax1.imshow(_log_view(small), cmap="gray")
    for name, point in info["candidatos"].items():
        sy, sx = _to_small(point, f)
        ax1.plot(sx, sy, "x", color=COLORES[name], markersize=12,
                 markeredgewidth=2,
                 label=f"{name} (unif. {info['uniformidad'][name]:.3f})")
    sy, sx = _to_small(center, f)
    ax1.plot(sx, sy, "r+", markersize=22, markeredgewidth=2,
             label="centro final")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Centros candidatos (más uniforme = mejor)")
    ax1.axis("off")

    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("red")
    step = max(1, int(np.ceil(max(prep.shape) / 1024)))
    ax2.imshow(prep[::step, ::step], cmap=cmap)
    ax2.set_title(f"Preprocesada: log, suavizado; en rojo lo enmascarado "
                  f"({100 * np.isnan(prep).mean():.1f} %)")
    ax2.axis("off")
    fig.suptitle(f"{path.name} — centro (y, x) = ({center[0]:.1f}, "
                 f"{center[1]:.1f}) px")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el Bloque 2 para todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':30} {'centro (y, x)':>18} {'método':>12} "
          f"{'dispersión':>10} {'enmascarado':>11}")
    for path in files:
        image = cargar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            info = _center_candidates(image)
            prep = preprocess_diffraction(image, info["centro"])
        cy, cx = info["centro"]
        aviso = "  <- revisar a mano" if info["dispersion_px"] > 5 else ""
        print(f"{path.name[:30]:30} {f'({cy:.1f}, {cx:.1f})':>18} "
              f"{info['metodo']:>12} {info['dispersion_px']:>8.1f} px "
              f"{100 * np.isnan(prep).mean():>9.1f} %{aviso}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target)
