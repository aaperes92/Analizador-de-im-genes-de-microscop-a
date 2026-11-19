"""Prueba del Bloque C3 de la Etapa 4: Hopkins y DBSCAN.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/c3_clustering.py IMAGEN [FONDO]
    uv run python scripts/fases/c3_clustering.py CARPETA
    uv run python scripts/fases/c3_clustering.py --sintetico

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: muestra las partículas coloreadas por grupo de DBSCAN
(aisladas en gris) con el radio de cada grupo, y H en el título.
Con una carpeta: imprime una fila por imagen y fase.
Con --sintetico: compara una grilla, partículas al azar, varios
grupos y un solo grupo.

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
from tem.spatial import _phase_points, detect_clustering  # noqa: E402


def preparar(path: Path, fondo: int | None) -> tuple[dict, dict]:
    """Corre las Etapas 1 y 2 y el Bloque 1 de la Etapa 3, en silencio.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).

    Returns:
        Tupla (resultado de pipeline_segmentation, props).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        seg = pipeline_segmentation(pipeline_tem(path, show=False),
                                    show=False)
        _, names = phase_names(seg["mask"], fondo)
        props = extract_region_properties(seg["mask"], seg["processed"],
                                          phases=list(names.values()))
    return seg, props


def graficar(ax: plt.Axes, imagen: np.ndarray, puntos: np.ndarray,
             d: dict, titulo: str) -> None:
    """Dibuja las partículas coloreadas por grupo de DBSCAN.

    Args:
        ax: Eje donde dibujar.
        imagen: Imagen de fondo.
        puntos: Centroides (n, 2) en el mismo orden que las etiquetas.
        d: Resultado de detect_clustering para una fase.
        titulo: Título del gráfico.
    """
    ax.imshow(imagen, cmap="gray")
    if d["etiquetas_dbscan"] is not None:
        etiquetas = d["etiquetas_dbscan"]
        aisladas = etiquetas == -1
        ax.plot(puntos[aisladas, 1], puntos[aisladas, 0], "o",
                color="lightgray", markersize=5, mec="black", mew=0.5)
        cmap = plt.get_cmap("tab20")
        for c in d["clusters"]:
            color = cmap((c["id"] - 1) % 20)
            sel = etiquetas == c["id"] - 1
            ax.plot(puntos[sel, 1], puntos[sel, 0], "o", color=color,
                    markersize=5, mec="black", mew=0.5)
            ax.add_patch(plt.Circle((c["centroide"][1], c["centroide"][0]),
                                    c["radio_px"], fill=False,
                                    edgecolor=color, linewidth=1.5))
        detalle = (f"Hopkins H = {d['hopkins_statistic']:.2f} → "
                   f"{d['hopkins_clasificacion']}\nDBSCAN: "
                   f"{d['n_clusters_dbscan']} grupo(s), "
                   f"{d['n_particulas_aisladas']} aislada(s) (gris)")
    else:
        detalle = f"{d['n']} partículas → {d['hopkins_clasificacion']}"
    ax.set_title(f"{titulo}\n{detalle}", fontsize=10)
    ax.axis("off")


def mascara_sintetica(patron: str) -> np.ndarray:
    """Máscara de 1200 x 1200 con 100 círculos según el patrón.

    Args:
        patron: 'grilla', 'azar', 'grupos' o 'un grupo'.

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    rng = np.random.default_rng(3)
    if patron == "grilla":
        centros = [np.array((60 + 120 * i, 60 + 120 * j), float)
                   for i in range(10) for j in range(10)]
    else:
        if patron == "azar":
            def nuevo() -> np.ndarray:
                return rng.uniform(15, 1185, 2)
        elif patron == "grupos":
            semillas = rng.uniform(150, 1050, (5, 2))

            def nuevo() -> np.ndarray:
                return semillas[rng.integers(5)] + rng.normal(0, 40, 2)
        else:  # 'un grupo'
            def nuevo() -> np.ndarray:
                return np.array([600.0, 600.0]) + rng.normal(0, 70, 2)
        centros = []
        while len(centros) < 100:
            c = nuevo()
            if (np.all((c > 12) & (c < 1188))
                    and all(np.hypot(*(c - o)) > 22 for o in centros)):
                centros.append(c)
    mask = np.ones((1200, 1200), dtype=np.int32)
    for c in centros:
        rr, cc = disk(tuple(c), 10, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def probar_sintetico() -> None:
    """Compara Hopkins y DBSCAN en cuatro patrones conocidos."""
    patrones = ("grilla", "azar", "grupos", "un grupo")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))
    for ax, patron in zip(axes, patrones):
        mask = mascara_sintetica(patron)
        with contextlib.redirect_stdout(io.StringIO()):
            props = extract_region_properties(mask, np.zeros(mask.shape))
        print(f"=== {patron}")
        d = detect_clustering(props, mask)["fase_1"]
        puntos = _phase_points(props, mask)["fase_1"]["centroides"]
        graficar(ax, np.where(mask == 0, 0.2, 0.8), puntos, d,
                 f"Patrón: {patron}")
    fig.suptitle("Bloque C3 — Hopkins y DBSCAN en patrones sintéticos",
                 fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Detecta el agrupamiento de una imagen y lo grafica.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg, props = preparar(path, fondo)
    print(f"{path.name}\n\n=== detect_clustering ===")
    res = detect_clustering(props, seg["mask"], fondo)
    puntos = _phase_points(props, seg["mask"], fondo)
    fig, axes = plt.subplots(1, len(res), figsize=(9 * len(res), 9),
                             squeeze=False)
    for ax, (name, d) in zip(axes[0], res.items()):
        graficar(ax, seg["original"], puntos[name]["centroides"], d,
                 f"{path.name} — {phase_title(name)}")
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume el agrupamiento de todas las imágenes.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    print(f"{'archivo':36} {'fase':7} {'n':>4} {'H':>5} {'Hopkins':>12} "
          f"{'grupos':>6} {'aisladas':>8}")
    for path in files:
        seg, props = preparar(path, None)
        with contextlib.redirect_stdout(io.StringIO()):
            res = detect_clustering(props, seg["mask"])
        for name, d in res.items():
            if d["hopkins_statistic"] is None:
                print(f"{path.name:36} {phase_title(name):7} {d['n']:>4}  "
                      "insuficiente")
                continue
            print(f"{path.name:36} {phase_title(name):7} {d['n']:>4} "
                  f"{d['hopkins_statistic']:>5.2f} "
                  f"{d['hopkins_clasificacion']:>12} "
                  f"{d['n_clusters_dbscan']:>6} "
                  f"{d['n_particulas_aisladas']:>8}")


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
