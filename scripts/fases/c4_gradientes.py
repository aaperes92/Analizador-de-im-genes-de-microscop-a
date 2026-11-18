"""Prueba del Bloque C4 de la Etapa 4: gradientes espaciales.

Uso (desde la raíz del proyecto):
    uv run python scripts/fases/c4_gradientes.py IMAGEN [FONDO]
    uv run python scripts/fases/c4_gradientes.py CARPETA
    uv run python scripts/fases/c4_gradientes.py --sintetico

FONDO es opcional: la etiqueta de la máscara que es el fondo (por
defecto, la fase mayoritaria).
Con una imagen: abre 6 gráficos por fase (densidad, tamaño e
intensidad en las dos direcciones) con la recta ajustada (continua si
el gradiente es significativo, punteada si no).
Con una carpeta: imprime los gradientes significativos de cada imagen.
Con --sintetico: más partículas arriba que abajo (debe dar gradiente
vertical de densidad) y una distribución sin gradiente.

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
from tem.spatial import detect_spatial_gradients  # noqa: E402

TITULOS = {"densidad": "Partículas por franja",
           "tamanio": "Área media (px)",
           "intensidad": "Intensidad media"}


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


def graficar(fig: plt.Figure, fila: list, d: dict, nombre: str) -> None:
    """Dibuja los 6 gráficos de gradiente de una fase.

    Args:
        fig: Figura.
        fila: 6 ejes (densidad, tamaño, intensidad × vertical,
            horizontal).
        d: Resultado de detect_spatial_gradients para una fase.
        nombre: Nombre de la fase.
    """
    i = 0
    for magnitud in ("densidad", "tamanio", "intensidad"):
        for direccion in ("vertical", "horizontal"):
            ax = fila[i]
            i += 1
            fit = (d[magnitud] or {}).get(direccion)
            if fit is None:
                ax.text(0.5, 0.5, "insuficiente\n(menos de 10 "
                        "partículas)", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{nombre}: {magnitud} {direccion}",
                             fontsize=9)
                ax.axis("off")
                continue
            x, y = fit["posiciones_px"], fit["valores"]
            ax.plot(x, y, "o", color="tab:blue")
            if fit["p_value"] is not None:
                recta = fit["intercepto"] + fit["pendiente_por_100px"] \
                    / 100 * x
                ax.plot(x, recta, color="tab:red",
                        linestyle="-" if fit["significativo"] else ":")
                estado = "SIGNIFICATIVO" if fit["significativo"] else "no"
                detalle = f"p={fit['p_value']:.1e} ({estado})"
            else:
                detalle = "sin datos suficientes"
            eje = "fila (px, ↓)" if direccion == "vertical" else \
                "columna (px, →)"
            ax.set_xlabel(eje, fontsize=8)
            ax.set_ylabel(TITULOS[magnitud], fontsize=8)
            ax.set_title(f"{nombre}: {magnitud} {direccion}\n{detalle}",
                         fontsize=9)


def mascara_sintetica(con_gradiente: bool) -> np.ndarray:
    """Máscara de 1200 x 1200 con 150 círculos.

    Args:
        con_gradiente: Si es True, la probabilidad de un círculo baja
            linealmente de arriba hacia abajo.

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    rng = np.random.default_rng(11)
    mask = np.ones((1200, 1200), dtype=np.int32)
    centros: list[np.ndarray] = []
    while len(centros) < 150:
        c = rng.uniform(15, 1185, 2)
        if con_gradiente and rng.random() > 1 - c[0] / 1200:
            continue  # más probable arriba (fila chica)
        if all(np.hypot(*(c - o)) > 22 for o in centros):
            centros.append(c)
    for c in centros:
        rr, cc = disk(tuple(c), 10, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def probar_sintetico() -> None:
    """Compara una distribución con gradiente vertical y una sin."""
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    for fila, (con, titulo) in zip(axes, ((True, "más arriba"),
                                          (False, "sin gradiente"))):
        mask = mascara_sintetica(con)
        image = np.where(mask == 0, 0.2, 0.8)
        with contextlib.redirect_stdout(io.StringIO()):
            props = extract_region_properties(mask, image)
        print(f"=== {titulo}")
        d = detect_spatial_gradients(props, mask, image)["fase_1"]
        graficar(fig, list(fila), d, titulo)
    fig.suptitle("Bloque C4 — gradientes sintéticos", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_imagen(path: Path, fondo: int | None) -> None:
    """Detecta y grafica los gradientes de una imagen.

    Args:
        path: Ruta a la imagen.
        fondo: Etiqueta del fondo (None: la fase mayoritaria).
    """
    seg, props = preparar(path, fondo)
    print(f"{path.name}\n\n=== detect_spatial_gradients ===")
    res = detect_spatial_gradients(props, seg["mask"], seg["processed"],
                                   fondo)
    fig, axes = plt.subplots(len(res), 6, figsize=(24, 4 * len(res)),
                             squeeze=False)
    for fila, (name, d) in zip(axes, res.items()):
        graficar(fig, list(fila), d, phase_title(name))
    fig.suptitle(f"Bloque C4 — {path.name}", fontsize=14)
    fig.tight_layout()
    plt.show()


def probar_carpeta(folder: Path) -> None:
    """Resume los gradientes significativos de todas las imágenes.

    Args:
        folder: Carpeta con imágenes.
    """
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in SUPPORTED_FORMATS)
    if not files:
        print(f"No hay imágenes soportadas en {folder}")
        return
    for path in files:
        seg, props = preparar(path, None)
        with contextlib.redirect_stdout(io.StringIO()):
            res = detect_spatial_gradients(props, seg["mask"],
                                           seg["processed"])
        for name, d in res.items():
            g = d["gradientes_significativos"]
            print(f"{path.name:36} {phase_title(name):7} "
                  + ("; ".join(g) if g else "-"))


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
