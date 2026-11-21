"""Prueba del Bloque 3 de la Etapa 4.5: anillos (policristalino).

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b3_anillos.py IMAGEN [ANCHO_INSTR]
    uv run python scripts/difraccion/b3_anillos.py CARPETA

ANCHO_INSTR es opcional: el ancho instrumental en píxeles que se resta
en Scherrer (por defecto 0: los tamaños son mínimos).
Con una imagen: corre los Bloques 1 y 2 sin abrir ventanas (con los
metadatos por defecto si faltan), imprime las tablas de anillos y de
Scherrer y muestra la imagen con los anillos y el perfil radial.
Con una carpeta: imprime una fila por imagen.

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
    _to_small,
    compute_radial_profile,
    detect_diffraction_rings,
    estimate_crystallite_size,
    find_beam_center,
    preprocess_diffraction,
    read_camera_metadata,
)
from tem.loader import SUPPORTED_FORMATS, load_tem_image  # noqa: E402


def preparar(path: Path) -> tuple[np.ndarray, dict, tuple, np.ndarray]:
    """Bloques 1 y 2 sin ventanas ni mensajes.

    Returns:
        (imagen, metadatos, centro, imagen lineal con NaN en lo
        enmascarado).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        image = load_tem_image(path)
        meta = read_camera_metadata(path, interactive=False)
        center = find_beam_center(image)
        prep = preprocess_diffraction(image, center)
    return image, meta, center, np.where(np.isnan(prep), np.nan, image)


def probar_imagen(path: Path, instrumental: float) -> None:
    """Corre el Bloque 3 sobre una imagen y muestra la figura.

    Args:
        path: Ruta a la imagen.
        instrumental: Ancho instrumental en píxeles para Scherrer.
    """
    image, meta, center, masked = preparar(path)
    print(f"{path.name}: centro (y, x) = ({center[0]:.1f}, "
          f"{center[1]:.1f}), calibración "
          f"{meta['calibracion_inv_angs_px']:.6f} 1/Å por px "
          f"({meta['fuente']})\n")
    print("=== compute_radial_profile ===")
    profile = compute_radial_profile(masked, center, meta)
    print(f"r de 0 a {profile['r_px'][-1]:.0f} px; d de "
          f"{profile['d_angs'][-1]:.3f} Å (borde) hacia arriba\n")
    print("=== detect_diffraction_rings ===")
    rings = detect_diffraction_rings(profile)
    print("\n=== estimate_crystallite_size ===")
    sizes = estimate_crystallite_size(rings,
                                      instrumental_fwhm_px=instrumental)
    graficar(path, image, center, profile, rings, sizes)


def graficar(path: Path, image: np.ndarray, center: tuple, profile: dict,
             rings: dict, sizes: dict) -> None:
    """Imagen con los anillos y perfil radial con fondo y FWHM.

    Args:
        path: Ruta a la imagen (para el título).
        image: Imagen normalizada.
        center: Centro del haz.
        profile: Resultado de compute_radial_profile.
        rings: Resultado de detect_diffraction_rings.
        sizes: Resultado de estimate_crystallite_size.
    """
    small, f = _downsample(image, 1024)
    cy, cx = _to_small(center, f)
    colors = plt.cm.tab10(np.arange(max(rings["n_anillos"], 1)) % 10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.imshow(_log_view(small), cmap="gray")
    ax1.plot(cx, cy, "r+", markersize=16, markeredgewidth=2)
    for ring, color in zip(rings["anillos"], colors):
        ax1.add_patch(plt.Circle((cx, cy), ring["r_px"] / f, fill=False,
                                 color=color, lw=1.2))
        ax1.text(cx + ring["r_px"] / f * 0.71, cy - ring["r_px"] / f * 0.71,
                 f"{ring['d_angs']:.2f} Å", color=color, fontsize=9,
                 fontweight="bold")
    ax1.set_title("Anillos detectados (d en Å)")
    ax1.axis("off")

    r = profile["r_px"]
    ax2.semilogy(r, profile["intensity_smooth"], color="0.25",
                 label="I(r) suavizado")
    ax2.semilogy(r, rings["fondo"], color="c", lw=0.8,
                 label="fondo para detectar")
    sizes_by_id = {row["id"]: row["D_nm"] for row in sizes["por_anillo"]}
    for ring, color in zip(rings["anillos"], colors):
        ax2.axvline(ring["r_px"], color=color, lw=1)
        half = ring["fwhm_px"] / 2
        level = profile["intensity_smooth"][int(round(ring["r_px"]))]
        # La barra del FWHM va a media altura sobre la base local
        half_level = level - ring["intensidad"] / 2
        ax2.plot([ring["r_px"] - half, ring["r_px"] + half],
                 [half_level, half_level], color=color, lw=3)
        size = sizes_by_id.get(ring["id"])
        ax2.text(ring["r_px"], level * 1.08,
                 f"{ring['d_angs']:.2f} Å\n"
                 + (f"{size:.1f} nm" if size else "-"),
                 color=color, fontsize=8, ha="center")
    ax2.set_xlim(profile["r_inicio"], r[-1])
    ax2.set_xlabel("r (px)")
    ax2.set_ylabel("intensidad media")
    ax2.legend(loc="upper right", fontsize=8)
    mean = sizes["D_medio_nm"]
    ax2.set_title("Perfil radial — barra = FWHM; abajo del d, Scherrer "
                  + ("(mínimo)" if sizes["es_minimo"] else ""))
    # Eje superior en d (Å)
    top = ax2.secondary_xaxis(
        "top", functions=(
            lambda x: 1 / (np.maximum(x, 1e-9)
                           * profile["calibracion_inv_angs_px"]),
            lambda d: 1 / (np.maximum(d, 1e-9)
                           * profile["calibracion_inv_angs_px"])))
    top.set_xlabel("d (Å)")
    fig.suptitle(f"{path.name} — {rings['n_anillos']} anillos"
                 + (f" — tamaño medio {mean:.1f} nm" if mean else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el Bloque 3 para todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':24} {'anillos':>7} {'D medio':>8}  "
          "d (Å) [(d1/d)²]")
    for path in files:
        _, meta, center, masked = preparar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            profile = compute_radial_profile(masked, center, meta)
            rings = detect_diffraction_rings(profile)
            sizes = estimate_crystallite_size(rings)
        mean = sizes["D_medio_nm"]
        ds = "  ".join(f"{a['d_angs']:.3f} [{a['q2_relativo']:.2f}]"
                       for a in rings["anillos"])
        print(f"{path.name[:24]:24} {rings['n_anillos']:>7} "
              f"{(f'{mean:.1f} nm' if mean else '-'):>8}  {ds}")
    print("\n(D medio: Scherrer sin restar el ancho instrumental, es un "
          "tamaño mínimo. Los d dependen de la calibración: revisá la "
          "columna 'fuente' del Bloque 1.)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target,
                      float(sys.argv[2]) if len(sys.argv) == 3 else 0.0)
