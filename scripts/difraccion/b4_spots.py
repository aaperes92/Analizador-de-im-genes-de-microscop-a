"""Prueba del Bloque 4 de la Etapa 4.5: spots (monocristal / NBED).

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b4_spots.py IMAGEN
    uv run python scripts/difraccion/b4_spots.py CARPETA
    uv run python scripts/difraccion/b4_spots.py --sintetico

Con una imagen: corre los Bloques 1 y 2 sin abrir ventanas, imprime los
spots, la simetría (ley de Friedel), las familias de planos y la red
estimada, y muestra cuatro paneles: spots por familia, red recíproca
con a* y b*, histograma de d y ángulos entre spots de una misma familia.
Con una carpeta: imprime una fila por imagen.
Con --sintetico: lo mismo con una red hexagonal sintética conocida
(a* = b* = 45 px, γ* = 60°), porque no hay patrones reales de un
monocristal.

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
    _log_view,
    analyze_spot_geometry,
    detect_diffraction_spots,
    electron_wavelength,
    find_beam_center,
    preprocess_diffraction,
    read_camera_metadata,
)
from tem.loader import SUPPORTED_FORMATS, load_tem_image  # noqa: E402


def preparar(path: Path) -> tuple[dict, tuple, np.ndarray]:
    """Bloques 1 y 2 sin ventanas ni mensajes.

    Returns:
        (metadatos, centro, imagen lineal con NaN en lo enmascarado).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        image = load_tem_image(path)
        meta = read_camera_metadata(path, interactive=False)
        center = find_beam_center(image)
        prep = preprocess_diffraction(image, center)
    return meta, center, np.where(np.isnan(prep), np.nan, image)


def sintetico() -> tuple[str, dict, tuple, np.ndarray]:
    """Red hexagonal sintética: a* = b* = 45 px, γ* = 60°, con halo y
    ruido."""
    rng = np.random.default_rng(0)
    side, center = 512, (270.0, 245.0)
    y, x = np.indices((side, side))
    r = np.hypot(y - center[0], x - center[1])
    img = 1.0 / (1 + r / 30.0) + rng.normal(0, 0.002, (side, side))
    a = np.array([0.0, 45.0])
    b = np.array([45 * np.sin(np.radians(60)), 45 * np.cos(np.radians(60))])
    for h in range(-6, 7):
        for k in range(-6, 7):
            if h == k == 0:
                continue
            py, px = np.array(center) + h * a + k * b
            img += 0.05 * np.exp(-0.5 * ((y - py) ** 2 + (x - px) ** 2)
                                 / 1.5**2)
    meta = {"calibracion_inv_angs_px": 0.002,
            "lambda_angs": electron_wavelength(200)}
    return "red hexagonal sintética", meta, center, img


def probar(name: str, meta: dict, center: tuple,
           image: np.ndarray) -> None:
    """Corre el Bloque 4 y muestra la figura.

    Args:
        name: Nombre para el título.
        meta: Metadatos (calibración).
        center: Centro del haz.
        image: Imagen lineal (con NaN en lo enmascarado).
    """
    print(f"{name}: centro (y, x) = ({center[0]:.1f}, {center[1]:.1f})\n")
    print("=== detect_diffraction_spots ===")
    spots = detect_diffraction_spots(image, center, meta)
    print("\n=== analyze_spot_geometry ===")
    geo = analyze_spot_geometry(spots, center, meta)
    graficar(name, image, center, spots, geo)


def graficar(name: str, image: np.ndarray, center: tuple, spots: dict,
             geo: dict) -> None:
    """Cuatro paneles del análisis de spots.

    Args:
        name: Nombre para el título.
        image: Imagen lineal.
        center: Centro del haz.
        spots: Resultado de detect_diffraction_spots.
        geo: Resultado de analyze_spot_geometry.
    """
    items = spots["spots"]
    fams = geo["familias"]
    family_of = {}
    for fam in fams:
        for phi in fam["phi_deg"]:
            family_of[round(phi, 6)] = fam["id"]
    cmap = plt.get_cmap("tab20")
    color = [cmap((family_of.get(round(s["phi_deg"], 6), 0) - 1) % 20)
             for s in items]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    ax = axs[0, 0]
    step = max(1, int(np.ceil(max(image.shape) / 1024)))
    view = _log_view(np.nan_to_num(image, nan=0.0))[::step, ::step]
    ax.imshow(view, cmap="gray")
    ax.scatter([s["x_px"] / step for s in items],
               [s["y_px"] / step for s in items], s=60,
               facecolors="none", edgecolors=color, linewidths=1.5)
    ax.plot(center[1] / step, center[0] / step, "r+", markersize=16,
            markeredgewidth=2)
    ax.set_title(f"{spots['n_spots']} spots (color = familia de planos)")
    ax.axis("off")

    ax = axs[0, 1]
    dx = np.array([s["x_px"] - center[1] for s in items])
    dy = np.array([center[0] - s["y_px"] for s in items])  # y hacia arriba
    ax.scatter(dx, dy, c=color, s=30)
    ax.scatter(dx, dy, s=8, c=["k" if s["tiene_par_simetrico"] else "w"
                               for s in items], zorder=3)
    ax.plot(0, 0, "r+", markersize=14)
    red = geo["red"]
    if red is not None:
        for vec, label, col in zip(red["vectores_px"], ("a*", "b*"),
                                   ("red", "blue")):
            ax.annotate("", xy=(vec[1], -vec[0]), xytext=(0, 0),
                        arrowprops={"arrowstyle": "->", "color": col,
                                    "lw": 2})
            ax.text(vec[1] * 1.15, -vec[0] * 1.15, label, color=col,
                    fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x - cx (px)")
    ax.set_ylabel("cy - y (px)")
    index = spots["indice_simetria"]
    ax.set_title("Espacio recíproco (punto negro: tiene par de Friedel)"
                 + (f"\nsimetría {index:.2f}" if index is not None else "")
                 + (f", indexados {100 * geo['fraccion_indexada']:.0f} %"
                    if geo["fraccion_indexada"] is not None else ""))

    ax = axs[1, 0]
    d = [s["d_angs"] for s in items if s["d_angs"]]
    if d:
        ax.hist(d, bins=40, color="0.5")
        for fam in fams:
            if fam["n_spots"] >= 3 and fam["d_angs"]:
                ax.axvline(fam["d_angs"], color="r", lw=0.8)
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("spots")
    ax.set_title("Distancias d de los spots (rojo: familias con 3+ spots)")

    ax = axs[1, 1]
    angles = []
    for fam in fams:
        phis = np.radians(fam["phi_deg"])
        for i in range(len(phis)):
            for j in range(i + 1, len(phis)):
                diff = np.degrees(abs(np.angle(np.exp(1j * (phis[i]
                                                           - phis[j])))))
                angles.append(diff)
    ax.hist(angles, bins=np.arange(0, 182, 2), color="0.5")
    for ang, _ in geo["angulos_caracteristicos"][:3]:
        ax.axvline(ang, color="r", lw=0.8)
        ax.text(ang, ax.get_ylim()[1] * 0.9, f"{ang:.0f}°", color="r",
                ha="center")
    ax.set_xlabel("ángulo entre spots de una misma familia (°)")
    ax.set_title("Ángulos característicos")

    text = "Red: -" if red is None else (
        f"Red: |a*| = {red['a_estrella_px']:.1f} px, |b*| = "
        f"{red['b_estrella_px']:.1f} px, γ* = "
        f"{red['gamma_estrella_deg']:.1f}°"
        + (f"  ->  a = {red['a_angs']:.2f} Å, b = {red['b_angs']:.2f} Å, "
           f"γ = {red['gamma_deg']:.1f}°" if red["a_angs"] else ""))
    fig.suptitle(f"{name}\n{text}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el Bloque 4 para todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':24} {'spots':>6} {'simetría':>9} {'indexados':>10} "
          f"{'familias':>9}  ángulos característicos")
    for path in files:
        meta, center, image = preparar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            spots = detect_diffraction_spots(image, center, meta)
            geo = analyze_spot_geometry(spots, center, meta)
        sym = spots["indice_simetria"]
        idx = geo["fraccion_indexada"]
        angles = ", ".join(f"{a:.0f}°"
                           for a, _ in geo["angulos_caracteristicos"][:3])
        print(f"{path.name[:24]:24} {spots['n_spots']:>6} "
              f"{('-' if sym is None else f'{sym:.2f}'):>9} "
              f"{('-' if idx is None else f'{100 * idx:.0f} %'):>10} "
              f"{len(geo['familias']):>9}  {angles}")
    print("\n(Un monocristal da simetría e indexados cerca de 1; un "
          "policristal con pocos granos, valores bajos.)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--sintetico":
        probar(*sintetico())
    else:
        target = Path(sys.argv[1])
        if target.is_dir():
            probar_carpeta(target)
        else:
            probar(target.name, *preparar(target))
