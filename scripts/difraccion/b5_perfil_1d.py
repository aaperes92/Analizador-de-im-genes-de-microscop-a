"""Prueba del Bloque 5 de la Etapa 4.5: perfil 1D.

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b5_perfil_1d.py ARCHIVO [ORIGEN]
    uv run python scripts/difraccion/b5_perfil_1d.py --desde-patron IMAGEN
    uv run python scripts/difraccion/b5_perfil_1d.py --sintetico

ARCHIVO puede ser una imagen angosta (perfil como franja) o un texto
(.txt / .csv) con el perfil en la última columna. ORIGEN es opcional:
la posición del haz directo en píxeles (si no, se detecta).
--desde-patron: arma un perfil 1D a partir de un patrón 2D (integración
azimutal de los Bloques 1-3, en franja de 15 filas con un corte de dos
lados) y lo analiza: los d tienen que coincidir con los del Bloque 3.
--sintetico: perfil con tres picos conocidos (80, 130 y 190 px).

La calibración sale de los metadatos del archivo; para un texto o un
sintético se usa 0.002 1/Å por píxel (cambiala en CALIBRACION).

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
    analyze_1d_profile,
    compute_radial_profile,
    electron_wavelength,
    find_beam_center,
    preprocess_diffraction,
    read_camera_metadata,
)
from tem.loader import load_tem_image  # noqa: E402

# Calibración (1/Å por píxel) para perfiles en texto y sintéticos
CALIBRACION = 0.002


def meta_por_defecto() -> dict:
    """Metadatos mínimos con CALIBRACION a 200 kV."""
    return {"calibracion_inv_angs_px": CALIBRACION,
            "lambda_angs": electron_wavelength(200)}


def cargar(path: Path) -> tuple[np.ndarray, dict]:
    """Carga el perfil (texto o imagen) y sus metadatos.

    Returns:
        (perfil o imagen, metadatos).
    """
    if path.suffix.lower() in (".txt", ".csv"):
        delim = "," if path.suffix.lower() == ".csv" else None
        data = np.loadtxt(path, delimiter=delim, ndmin=2)
        return data[:, -1], meta_por_defecto()
    with contextlib.redirect_stdout(io.StringIO()):
        image = load_tem_image(path)
        meta = read_camera_metadata(path, interactive=False)
    return image, meta


def desde_patron(path: Path) -> tuple[np.ndarray, dict]:
    """Perfil de dos lados (franja de 15 filas) a partir de un patrón 2D.

    Returns:
        (franja, metadatos del patrón).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        image = load_tem_image(path)
        meta = read_camera_metadata(path, interactive=False)
        center = find_beam_center(image)
        prep = preprocess_diffraction(image, center)
        prof = compute_radial_profile(np.where(np.isnan(prep), np.nan,
                                               image), center, meta)
    line = prof["intensity"]
    line = np.where(np.isfinite(line), line, np.nanmin(line))
    two_sided = np.concatenate([line[::-1], line[1:]])
    rng = np.random.default_rng(0)
    strip = np.tile(two_sided, (15, 1))
    strip += rng.normal(0, 0.002, strip.shape)
    print(f"Perfil armado desde {path.name}: {len(two_sided)} puntos, "
          f"haz directo en {len(line) - 1} px\n")
    return strip, meta


def sintetico() -> tuple[np.ndarray, dict]:
    """Perfil desde el haz directo con picos en 80, 130 y 190 px."""
    rng = np.random.default_rng(0)
    r = np.arange(400, dtype=float)
    line = 1.0 / (1 + r / 30.0)
    for p in (80, 130, 190):
        line += 0.03 * np.exp(-0.5 * ((r - p) / 3.0) ** 2)
    return line + rng.normal(0, 0.0005, len(r)), meta_por_defecto()


def probar(name: str, data: np.ndarray, meta: dict,
           origin: float | None) -> None:
    """Corre el Bloque 5 y muestra la figura.

    Args:
        name: Nombre para el título.
        data: Perfil (vector) o imagen angosta.
        meta: Metadatos (calibración y λ).
        origin: Posición del haz directo (None: se detecta).
    """
    print("=== analyze_1d_profile ===")
    res = analyze_1d_profile(data, meta, origin=origin)
    prof = res["perfil"]
    peaks = res["picos"]["anillos"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 11))
    ax = axs[0]
    if np.ndim(data) == 2:
        ax.imshow(data, cmap="gray", aspect="auto")
        ax.set_title(f"Imagen ({res['eje']})")
    else:
        ax.plot(data, color="0.3")
        ax.axvline(res["origen_px"], color="r", lw=1)
        ax.set_title("Perfil original (rojo: origen)")
    r = prof["r_px"][1:]
    d = prof["d_angs"][1:]
    q = prof["q_inv_angs"][1:]
    smooth = res["intensity_savgol"][1:]
    ax = axs[1]
    ax.plot(d, smooth, color="0.25")
    for p in peaks:
        ax.axvline(p["d_angs"], color="r", lw=0.8)
        ax.text(p["d_angs"], ax.get_ylim()[1] * 0.97,
                f"{p['d_angs']:.2f}", color="r", ha="center", va="top",
                fontsize=8)
    ax.set_xlim(d[-1], min(d[prof["r_inicio"]], 4 * d[-1] * 10))
    ax.invert_xaxis()
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("I (Savitzky-Golay)")
    ax.set_title("I vs d")
    ax = axs[2]
    ax.plot(q, smooth, color="0.25")
    for p in peaks:
        ax.axvline(p["q_inv_angs"], color="r", lw=0.8)
        half = p["fwhm_q_inv_angs"] / 2
        ax.axvspan(p["q_inv_angs"] - half, p["q_inv_angs"] + half,
                   color="r", alpha=0.1)
    ax.set_xlabel("q = 1/d (1/Å)")
    ax.set_ylabel("I")
    ax.set_title("I vs q (sombreado: FWHM)")
    mean = res["cristalito"]["D_medio_nm"]
    fig.suptitle(f"{name} — {len(peaks)} picos"
                 + (f" — Scherrer (mínimo) {mean:.1f} nm" if mean else "")
                 + f"\norigen: {res['criterio_origen']}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--sintetico"]:
        probar("perfil sintético", *sintetico(), None)
    elif len(args) == 2 and args[0] == "--desde-patron":
        target = Path(args[1])
        probar(f"{target.name} (perfil 1D)", *desde_patron(target), None)
    elif len(args) in (1, 2) and not args[0].startswith("--"):
        target = Path(args[0])
        probar(target.name, *cargar(target),
               float(args[1]) if len(args) == 2 else None)
    else:
        print(__doc__)
        sys.exit(1)
