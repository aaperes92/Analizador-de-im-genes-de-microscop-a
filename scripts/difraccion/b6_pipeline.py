"""Prueba del Bloque 6 de la Etapa 4.5: pipeline completo y figura.

Uso (desde la raíz del proyecto):
    uv run python scripts/difraccion/b6_pipeline.py IMAGEN
    uv run python scripts/difraccion/b6_pipeline.py IMAGEN --difraccion
    uv run python scripts/difraccion/b6_pipeline.py --sintetico

Con una imagen: corre pipeline_diffraction completo, como lo haría
analizar_tem.py con una sola imagen: pide los metadatos que faltan,
abre la ventana para aceptar o corregir el centro y muestra la figura
de 4 paneles (se guarda en data/samples/<nombre>_difraccion.png). Con
--difraccion se analiza como patrón aunque la detección diga que no.
Con --sintetico: genera un patrón de cada tipo (anillos, puntos, NBED,
amorfo y perfil 1D) y muestra las cinco figuras, sin preguntar nada.

Para analizar carpetas enteras (y mezclar imágenes de campo claro con
patrones de difracción) usá scripts/analizar_tem.py.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Permite importar el paquete tem desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tem import pipeline_diffraction  # noqa: E402


def _normalizar(img: np.ndarray) -> np.ndarray:
    """Lleva la imagen a [0, 1] por percentiles (como load_tem_image)."""
    lo, hi = np.percentile(img, [0.1, 99.9])
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def sinteticos() -> dict[str, np.ndarray]:
    """Un patrón sintético de cada tipo."""
    rng = np.random.default_rng(0)
    side, (cy, cx) = 512, (270.0, 245.0)
    y, x = np.indices((side, side))
    r = np.hypot(y - cy, x - cx)
    halo = 1.0 / (1 + r / 30.0) + rng.normal(0, 0.002, (side, side))

    rings = halo.copy()
    for radius in (80, 92, 131, 153):  # FCC: sqrt(3, 4, 8, 11)
        rings += 0.03 * np.exp(-0.5 * ((r - radius) / 2.5) ** 2)
    rings[(np.abs(y - cy) < 12) & (x > cx - 5)] = rings.min()  # stopper

    def lattice(sigma: float, step: float, height: float) -> np.ndarray:
        img = halo.copy()
        a = np.array([0.0, step])
        b = np.array([step * np.sin(np.radians(60)),
                      step * np.cos(np.radians(60))])
        for h in range(-8, 9):
            for k in range(-8, 9):
                if h == k == 0:
                    continue
                py, px = np.array([cy, cx]) + h * a + k * b
                img += height * np.exp(
                    -0.5 * ((y - py) ** 2 + (x - px) ** 2) / sigma**2)
        return img

    amorphous = halo + 0.003 * np.exp(-0.5 * ((r - 90) / 25) ** 2)
    radial = np.arange(600, dtype=float)
    line = 1.0 / (1 + radial / 30.0)
    for p in (120, 139, 196, 230):
        line += 0.03 * np.exp(-0.5 * ((radial - p) / 3.0) ** 2)
    strip = np.tile(np.concatenate([line[::-1], line[1:]]), (20, 1))
    strip += rng.normal(0, 0.002, strip.shape)
    return {
        "sintetico_anillos": _normalizar(rings),
        "sintetico_puntos": _normalizar(lattice(1.5, 45, 0.05)),
        "sintetico_nbed": _normalizar(lattice(5.0, 70, 0.05)),
        "sintetico_amorfo": _normalizar(amorphous),
        "sintetico_perfil_1d": strip,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--sintetico"]:
        with tempfile.TemporaryDirectory() as folder:
            for name, image in sinteticos().items():
                path = Path(folder) / f"{name}.png"
                path.write_bytes(b"")  # sin metadatos: valores por defecto
                print(f"\n##### {name} #####")
                pipeline_diffraction(path, image=image, interactive=False)
    elif args and not args[0].startswith("--"):
        result = pipeline_diffraction(Path(args[0]),
                                      force="--difraccion" in args)
        if not result["es_difraccion"]:
            print("\nPara analizarla igual: agregá --difraccion.")
    else:
        print(__doc__)
        sys.exit(1)
