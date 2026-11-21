"""Prueba del Bloque 1 de la Etapa 4.5: tipo de imagen y metadatos.

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b1_tipo_y_metadatos.py IMAGEN
    uv run python scripts/difraccion/b1_tipo_y_metadatos.py IMAGEN --preguntar
    uv run python scripts/difraccion/b1_tipo_y_metadatos.py CARPETA

Con una imagen: imprime los votos de is_diffraction_pattern, el tipo de
patrón y la tabla de metadatos, y muestra una figura con el centro, los
anillos detectados y el perfil radial. Con --preguntar, los datos que
faltan en los metadatos se piden por teclado (si no, se usan los
valores por defecto).
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
    _downsample,
    _log_view,
    _pattern_features,
    detect_pattern_type,
    is_diffraction_pattern,
    read_camera_metadata,
)
from tem.loader import SUPPORTED_FORMATS, load_tem_image  # noqa: E402


def cargar(path: Path) -> np.ndarray:
    """Carga la imagen sin imprimir nada."""
    with contextlib.redirect_stdout(io.StringIO()):
        return load_tem_image(path)


def probar_imagen(path: Path, preguntar: bool) -> None:
    """Corre el Bloque 1 sobre una imagen y muestra la figura.

    Args:
        path: Ruta a la imagen.
        preguntar: Si es True, pide por teclado los metadatos que faltan.
    """
    image = cargar(path)
    print("=== read_camera_metadata ===")
    meta = read_camera_metadata(path, interactive=preguntar)
    print("\n=== is_diffraction_pattern (solo la imagen) ===")
    es_dif = is_diffraction_pattern(image)
    if meta["modo_operacion"] == "DIFFRACTION":
        print("  (los metadatos también dicen modo DIFFRACTION)")
    if not es_dif:
        print("\nNo es un patrón de difracción: no se busca el tipo.")
        return
    print("\n=== detect_pattern_type ===")
    info = _pattern_features(image, None)
    kind = detect_pattern_type(image, center=info["centro"])
    graficar(path, image, info, kind, meta)


def graficar(path: Path, image: np.ndarray, info: dict, kind: str,
             meta: dict) -> None:
    """Imagen (log) con el centro y los anillos, y perfil radial.

    Args:
        path: Ruta a la imagen (para el título).
        image: Imagen normalizada.
        info: Resultado de _pattern_features.
        kind: Tipo de patrón.
        meta: Metadatos (para pasar los radios a d en Å).
    """
    small, f = _downsample(image, 1024)
    view = _log_view(small)
    cy, cx = info["centro"][0] / f, info["centro"][1] / f
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    ax1.imshow(view, cmap="gray")
    ax1.plot(cx, cy, "r+", markersize=18, markeredgewidth=2)
    cal = meta["calibracion_inv_angs_px"]
    for r, cov in zip(info["radios_anillos"], info["coberturas"]):
        continuo = cov >= 0.5
        ax1.add_patch(plt.Circle(
            (cx, cy), r / f, fill=False, lw=1.2,
            color="lime" if continuo else "orange",
            ls="-" if continuo else "--"))
        ax1.text(cx + r / f * 0.71, cy - r / f * 0.71,
                 f"{1 / (r * cal):.2f} Å\n{100 * cov:.0f} %",
                 color="lime" if continuo else "orange", fontsize=8)
    ax1.set_title("Centro (+) y picos radiales: verde = anillo continuo "
                  "(>= 50 %), naranja = no")
    ax1.axis("off")

    # Perfil radial (media azimutal en la vista log) desde el centro
    y, x = np.indices(view.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    r_max = int(min(cy, cx, view.shape[0] - cy, view.shape[1] - cx))
    sel = r < r_max
    prof = (np.bincount(r[sel], view[sel], minlength=r_max)
            / np.maximum(np.bincount(r[sel], minlength=r_max), 1))
    radios = np.arange(r_max) * f
    ax2.plot(radios, prof, color="0.3")
    for rr, cov in zip(info["radios_anillos"], info["coberturas"]):
        ax2.axvline(rr, color="lime" if cov >= 0.5 else "orange", lw=1,
                    ls="-" if cov >= 0.5 else "--")
    ax2.set_xlabel("radio (px de la imagen completa)")
    ax2.set_ylabel("intensidad media, log(1 + 100·I)")
    ax2.set_title("Perfil radial")
    fig.suptitle(f"{path.name} — tipo: {kind} — calibración "
                 f"{cal:.6f} 1/Å por px ({meta['fuente']})")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el Bloque 1 para todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':34} {'difracción':>10} {'tipo':>9} "
          f"{'modo (metadatos)':>17} {'calibración':>12} {'fuente':>12}")
    for path in files:
        image = cargar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            meta = read_camera_metadata(path, interactive=False)
            es_dif = is_diffraction_pattern(image)
            kind = detect_pattern_type(image) if es_dif else "-"
        print(f"{path.name[:34]:34} {'sí' if es_dif else 'no':>10} "
              f"{kind:>9} {meta['modo_operacion'] or '-':>17} "
              f"{meta['calibracion_inv_angs_px']:>12.6f} "
              f"{meta['fuente']:>12}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--preguntar"]
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)
    target = Path(args[0])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target, preguntar="--preguntar" in sys.argv)
