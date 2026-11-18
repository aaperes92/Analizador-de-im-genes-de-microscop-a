"""Prueba del Bloque B1 de la Etapa 4: perfiles de intensidad.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/b1_perfiles.py IMAGEN [FONDO]
    uv run python scripts/fases/b1_perfiles.py CARPETA

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre una figura con los histogramas de intensidad del
fondo y de cada fase, y el perfil de cada interfaz con el ancho de
transición (10-90 %) sombreado.
Con una carpeta: imprime una fila por imagen y par de fases.

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

from tem import pipeline_segmentation, pipeline_tem  # noqa: E402
from tem.compositional import (  # noqa: E402
    PLATEAU_PX,
    compute_intensity_gradients,
    compute_intensity_profiles,
)
from tem.fraction import phase_title  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.segmentation import PHASE_COLORS  # noqa: E402


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
    """Calcula y grafica los perfiles de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg = segmentar(path)
    print(f"{path.name}\n\n=== compute_intensity_profiles ===")
    perfiles = compute_intensity_profiles(seg["processed"], seg["mask"],
                                          fondo)
    print("\n=== compute_intensity_gradients ===")
    gradientes = compute_intensity_gradients(
        seg["processed"], seg["mask"], fondo, seg["lineas_watershed"]
    )

    n_pares = max(1, len(gradientes))
    fig, axes = plt.subplots(1, 1 + n_pares,
                             figsize=(7 * (1 + n_pares), 5.5),
                             squeeze=False)
    ax_h = axes[0, 0]
    for clave, p in perfiles["perfiles"].items():
        color = PHASE_COLORS[p["etiqueta_mascara"] % len(PHASE_COLORS)]
        centros = (p["bin_edges"][:-1] + p["bin_edges"][1:]) / 2
        ax_h.fill_between(centros, p["histograma"], color=color,
                          alpha=0.5, step="mid", label=nombre(clave))
        ax_h.axvline(p["media"], color=color, linestyle="--")
    ax_h.set_xlabel("Intensidad (imagen preprocesada)")
    ax_h.set_ylabel("Píxeles")
    ax_h.set_title("Intensidad por fase (línea: media)")
    ax_h.legend()

    for ax, (clave, g) in zip(axes[0, 1:], gradientes.items()):
        a, b = (nombre(n) for n in clave.split("-"))
        d, perfil = g["distancias_px"], g["perfil"]
        ax.plot(d, perfil, "o-", color="black", markersize=3)
        ax.axvline(0, color="gray", linestyle=":")
        inicio = perfil[:PLATEAU_PX].mean()
        fin = perfil[-PLATEAU_PX:].mean()
        ax.axhline(inicio, color="tab:blue", linestyle="--", linewidth=1)
        ax.axhline(fin, color="tab:red", linestyle="--", linewidth=1)
        if g["ancho_transicion_px"] is not None:
            norm = (perfil - inicio) / (fin - inicio)
            x10 = np.interp(0.1, norm, d) if norm[0] < norm[-1] else None
            x90 = np.interp(0.9, norm, d) if norm[0] < norm[-1] else None
            if x10 is None:  # perfil decreciente: se invierte
                x10 = np.interp(0.1, norm[::-1], d[::-1])
                x90 = np.interp(0.9, norm[::-1], d[::-1])
            ax.axvspan(min(x10, x90), max(x10, x90), color="gold",
                       alpha=0.4, label="ancho 10-90 %")
            ax.legend()
        if g["ancho_transicion_px"] is None:
            ancho = "-"
        elif g["meseta_alcanzada"]:
            ancho = f"{g['ancho_transicion_px']:.1f} px"
        else:
            ancho = f"≥ {g['ancho_transicion_px']:.1f} px (sin meseta)"
        ax.set_title(f"{a} → {b}\nancho {ancho}, pendiente máx. "
                     f"{g['pendiente_max']:.3f}/px")
        ax.set_xlabel(f"Distancia al borde (px): − lado {a}, + lado {b}")
        ax.set_ylabel("Intensidad media")
    if not gradientes:
        axes[0, 1].axis("off")
        axes[0, 1].text(0.5, 0.5, "Sin pares con interfaz suficiente",
                        ha="center")

    fig.suptitle(f"Bloque B1 — {path.name}", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume contraste e interfaces de todas las imágenes.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'par':18} {'contraste':>9} {'clase':>6} "
          f"{'ancho px':>8} {'pend./px':>8}")
    for path in files:
        seg = segmentar(path)
        with contextlib.redirect_stdout(io.StringIO()):
            perf = compute_intensity_profiles(seg["processed"], seg["mask"])
            grad = compute_intensity_gradients(
                seg["processed"], seg["mask"],
                watershed_lines=seg["lineas_watershed"],
            )
        for clave, c in perf["contraste"].items():
            g = grad.get(clave)
            if not g or g["ancho_transicion_px"] is None:
                ancho = "-"
            else:
                ancho = (("" if g["meseta_alcanzada"] else "≥")
                         + f"{g['ancho_transicion_px']:.1f}")
            pend = "-" if not g else f"{g['pendiente_max']:.4f}"
            par = "/".join(nombre(n) for n in clave.split("-"))
            print(f"{path.name:36} {par:18} {c['indice']:>9.3f} "
                  f"{c['clasificacion']:>6} {ancho:>8} {pend:>8}")


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
