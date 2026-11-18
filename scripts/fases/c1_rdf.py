"""Prueba del Bloque C1 de la Etapa 4: función de distribución radial.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/c1_rdf.py IMAGEN [FONDO]
    uv run python scripts/fases/c1_rdf.py CARPETA
    uv run python scripts/fases/c1_rdf.py --sintetico

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre un gráfico de g(r) por fase con la franja de lo
que da el azar.
Con una carpeta: imprime una fila por imagen y fase.
Con --sintetico: compara tres patrones conocidos (grilla, al azar y
agrupados).

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
from skimage.draw import disk  # noqa: E402

from tem import (  # noqa: E402
    extract_region_properties,
    pipeline_segmentation,
    pipeline_tem,
)
from tem.fraction import phase_names, phase_title  # noqa: E402
from tem.loader import SUPPORTED_FORMATS  # noqa: E402
from tem.spatial import compute_rdf  # noqa: E402


def preparar(path: Path, fondo: int | None) -> tuple[np.ndarray, dict]:
    """Corre las Etapas 1 y 2 y el Bloque 1 de la Etapa 3, en silencio.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).

    Returns:
        Tupla (máscara, props).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        _, names = phase_names(seg["mask"], fondo)
        props = extract_region_properties(seg["mask"], seg["processed"],
                                          phases=list(names.values()))
    return seg["mask"], props


def graficar(ax: plt.Axes, rdf: dict, titulo: str) -> None:
    """Dibuja g(r) de cada fase con la franja del azar.

    Args:
        ax: Eje donde dibujar.
        rdf: Resultado de compute_rdf.
        titulo: Título del gráfico.
    """
    for name, d in rdf.items():
        if d["g"] is None:
            continue
        line, = ax.plot(d["r"], d["g"], "o-", markersize=3,
                        label=f"{phase_title(name)} (n={d['n']}): "
                              f"{d['clasificacion']}")
        ax.fill_between(d["r"], d["g_azar_inf"], d["g_azar_sup"],
                        color=line.get_color(), alpha=0.15)
        ax.axvspan(*d["rango_corto_px"], color=line.get_color(),
                   alpha=0.07)
    ax.axhline(1, color="black", linestyle="--", linewidth=1,
               label="g = 1 (al azar)")
    ax.set_xlabel("r: distancia entre centroides (px)")
    ax.set_ylabel("g(r)")
    ax.set_title(titulo + "\n(franja: lo que da el azar; sombreado: "
                 "rango usado para clasificar)", fontsize=10)
    ax.legend(fontsize=8)


def mascara_sintetica(patron: str) -> np.ndarray:
    """Máscara de 1200 x 1200 con ~100 círculos según el patrón.

    Args:
        patron: 'grilla', 'azar' o 'agrupados'.

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    rng = np.random.default_rng(3)
    if patron == "grilla":
        centros = [(60 + 120 * i, 60 + 120 * j)
                   for i in range(10) for j in range(10)]
    elif patron == "azar":
        centros = []
        while len(centros) < 100:
            c = rng.uniform(15, 1185, 2)
            if all(np.hypot(*(c - o)) > 22 for o in centros):
                centros.append(c)
    else:  # 'agrupados': 10 grupos de 10 círculos
        semillas = rng.uniform(150, 1050, (10, 2))
        centros = []
        for s in semillas:
            while sum(np.hypot(*(np.array(c) - s)) < 80
                      for c in centros) < 10:
                c = s + rng.normal(0, 30, 2)
                if all(np.hypot(*(c - o)) > 22 for o in centros):
                    centros.append(c)
    mask = np.ones((1200, 1200), dtype=np.int32)
    for c in centros:
        rr, cc = disk(tuple(c), 10, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def probar_sintetico() -> None:
    """Muestra g(r) de tres patrones conocidos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, patron in zip(axes, ("grilla", "azar", "agrupados")):
        mask = mascara_sintetica(patron)
        with contextlib.redirect_stdout(io.StringIO()):
            props = extract_region_properties(mask, np.zeros(mask.shape))
        print(f"=== {patron}")
        graficar(ax, compute_rdf(props, mask), f"Patrón: {patron}")
    fig.suptitle("Bloque C1 — RDF de patrones sintéticos", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Calcula y grafica la RDF de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    mask, props = preparar(path, fondo)
    print(f"{path.name}\n\n=== compute_rdf ===")
    rdf = compute_rdf(props, mask, background=fondo)
    fig, ax = plt.subplots(figsize=(10, 6))
    graficar(ax, rdf, f"Bloque C1 — {path.name}")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume la RDF de todas las imágenes de una carpeta.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'g corto':>7} "
          f"{'franja azar':>12}  clasificación")
    for path in files:
        mask, props = preparar(path, None)
        with contextlib.redirect_stdout(io.StringIO()):
            rdf = compute_rdf(props, mask)
        for name, d in rdf.items():
            if d["g_corto"] is None:
                print(f"{path.name:36} {phase_title(name):7} {d['n']:>4} "
                      f"{'-':>7} {'-':>12}  {d['clasificacion']}")
                continue
            lo, hi = d["g_corto_azar"]
            print(f"{path.name:36} {phase_title(name):7} {d['n']:>4} "
                  f"{d['g_corto']:>7.2f} {lo:>5.2f}-{hi:<6.2f}  "
                  f"{d['clasificacion']}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--sintetico":
        probar_sintetico()
        sys.exit(0)
    target = Path(sys.argv[1])
    if target.is_dir():
        probar_carpeta(target)
    else:
        probar_imagen(target,
                      int(sys.argv[2]) if len(sys.argv) == 3 else None)
