"""Tests de src/tem/loader.py (Etapa 1: carga y preprocesamiento).

Las imágenes se generan sintéticamente con numpy en la carpeta temporal
de pytest, así que no hace falta tener archivos TEM reales.
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
from skimage import filters, io

from tem import loader
from tem.loader import (
    PREPROCESS_METHODS,
    load_tem_image,
    pipeline_tem,
    preprocess_tem_image,
)

# --------------------------------------------------------------------
# Utilidades para generar imágenes sintéticas
# --------------------------------------------------------------------


def _rng() -> np.random.Generator:
    """Generador aleatorio con semilla fija para tests reproducibles."""
    return np.random.default_rng(1234)


def _particles(size: int = 128) -> np.ndarray:
    """Imagen sintética de partículas circulares suaves en [0, 1].

    Args:
        size: Lado de la imagen cuadrada en píxeles.

    Returns:
        Array 2D float64 que imita una imagen TEM sin ruido.
    """
    rng = _rng()
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size), 0.3)
    for _ in range(12):
        cy, cx = rng.integers(0, size, 2)
        radius = rng.integers(4, 10)
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < radius**2] = 1.0
    return filters.gaussian(image, sigma=1.5)


def _write_image(path: Path, data: np.ndarray) -> Path:
    """Guarda un array con el formato que indica la extensión.

    Args:
        path: Ruta de destino (.tif, .tiff, .png, .jpg o .mrc).
        data: Array a guardar.

    Returns:
        La misma ruta, para encadenar.
    """
    if path.suffix.lower() == ".mrc":
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
    else:
        io.imsave(path, data, check_contrast=False)
    return path


def _assert_normalized(image: np.ndarray) -> None:
    """Verifica que la imagen sea float32 y esté en [0, 1]."""
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


# Un array de ejemplo por formato, con el dtype típico de cada uno
_SAMPLE = _particles(64)
_FORMAT_DATA: dict[str, np.ndarray] = {
    ".tif": (_SAMPLE * 65535).astype(np.uint16),
    ".tiff": (_SAMPLE * 65535).astype(np.uint16),
    ".png": (_SAMPLE * 255).astype(np.uint8),
    ".jpg": (_SAMPLE * 255).astype(np.uint8),
    ".mrc": _SAMPLE.astype(np.float32) * 1000.0 - 200.0,
}


# --------------------------------------------------------------------
# load_tem_image
# --------------------------------------------------------------------


@pytest.mark.parametrize("ext", sorted(_FORMAT_DATA))
def test_load_formatos_soportados(tmp_path: Path, ext: str) -> None:
    """Cada formato se carga como 2D float32 normalizado a [0, 1]."""
    path = _write_image(tmp_path / f"muestra{ext}", _FORMAT_DATA[ext])

    image = load_tem_image(path)

    assert image.shape == _SAMPLE.shape
    _assert_normalized(image)
    # Min-max: el rango completo [0, 1] queda ocupado
    assert image.min() == pytest.approx(0.0)
    assert image.max() == pytest.approx(1.0)


def test_load_dm3_con_lector_simulado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El .dm3 se lee con ncempy (simulado: ncempy no escribe dm3)."""
    from ncempy.io import dm

    raw = (_SAMPLE * 4000).astype(np.int32)
    llamadas: list[str] = []

    def lector_falso(filename: str) -> dict[str, np.ndarray]:
        llamadas.append(filename)
        return {"data": raw}

    monkeypatch.setattr(dm, "dmReader", lector_falso)
    path = tmp_path / "muestra.dm3"
    path.touch()

    image = load_tem_image(path)

    assert llamadas == [str(path)]
    assert image.shape == raw.shape
    _assert_normalized(image)


def test_load_acepta_str_y_extension_mayuscula(tmp_path: Path) -> None:
    """Acepta la ruta como str y extensiones en mayúscula."""
    path = _write_image(tmp_path / "MUESTRA.TIF", _FORMAT_DATA[".tif"])

    image = load_tem_image(str(path))

    _assert_normalized(image)


def test_load_color_pasa_a_grises(tmp_path: Path) -> None:
    """Las imágenes RGB y RGBA se convierten a una imagen 2D."""
    rgb = (_rng().random((32, 40, 3)) * 255).astype(np.uint8)
    rgba = (_rng().random((32, 40, 4)) * 255).astype(np.uint8)
    rgba[..., 3] = 255

    for name, data in (("rgb.png", rgb), ("rgba.png", rgba)):
        image = load_tem_image(_write_image(tmp_path / name, data))
        assert image.shape == (32, 40)
        _assert_normalized(image)


def test_load_pila_usa_primer_cuadro(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """De una pila .mrc se toma el primer cuadro y se avisa."""
    stack = _rng().random((3, 20, 30)).astype(np.float32)
    path = _write_image(tmp_path / "pila.mrc", stack)

    image = load_tem_image(path)

    assert image.shape == (20, 30)
    assert "se usa el primero" in capsys.readouterr().out


def test_load_imagen_constante_da_ceros(tmp_path: Path) -> None:
    """Una imagen constante no divide por cero: devuelve ceros."""
    flat = np.full((16, 16), 7, dtype=np.uint8)
    path = _write_image(tmp_path / "plana.tif", flat)

    image = load_tem_image(path)

    _assert_normalized(image)
    assert not image.any()


def test_load_imprime_informacion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Se imprimen nombre, formato, shape y dtype."""
    path = _write_image(tmp_path / "info.png", _FORMAT_DATA[".png"])

    load_tem_image(path)

    salida = capsys.readouterr().out
    assert "info.png" in salida
    assert ".png" in salida
    assert "(64, 64)" in salida
    assert "float32" in salida


def test_load_formato_no_soportado(tmp_path: Path) -> None:
    """Un formato no soportado lanza ValueError con mensaje claro."""
    path = tmp_path / "imagen.bmp"
    path.touch()

    with pytest.raises(ValueError, match="no soportado"):
        load_tem_image(path)


def test_load_archivo_inexistente(tmp_path: Path) -> None:
    """Un archivo que no existe lanza FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tem_image(tmp_path / "no_existe.tif")


# --------------------------------------------------------------------
# preprocess_tem_image
# --------------------------------------------------------------------


@pytest.mark.parametrize("method", PREPROCESS_METHODS)
def test_preprocess_conserva_shape(method: str) -> None:
    """Cada método devuelve el mismo shape, en float32 y en [0, 1]."""
    noisy = _particles(96) + _rng().normal(0, 0.1, (96, 96))
    image = loader._normalize(noisy)

    processed = preprocess_tem_image(image, method=method)

    assert processed.shape == image.shape
    _assert_normalized(processed)
    assert np.isfinite(processed).all()


def test_preprocess_imagen_no_cuadrada() -> None:
    """También conserva el shape de imágenes rectangulares."""
    image = loader._normalize(_rng().random((50, 80)))

    for method in PREPROCESS_METHODS:
        assert preprocess_tem_image(image, method).shape == (50, 80)


def test_preprocess_metodo_invalido() -> None:
    """Un método inexistente lanza ValueError."""
    image = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="no válido"):
        preprocess_tem_image(image, method="sobel")


def test_preprocess_rechaza_imagen_no_2d() -> None:
    """Una imagen que no es 2D lanza ValueError."""
    with pytest.raises(ValueError, match="2D"):
        preprocess_tem_image(np.zeros((2, 8, 8), dtype=np.float32))


def test_auto_elige_median_con_sal_y_pimienta() -> None:
    """Con ruido impulsivo el modo auto elige la mediana."""
    image = _particles()
    mask = _rng().random(image.shape)
    image[mask < 0.01] = 0.0
    image[mask > 0.99] = 1.0

    method, _ = loader._select_method(loader._normalize(image))

    assert method == "median"


@pytest.mark.parametrize(
    ("sigma", "esperado"),
    [(0.02, "gaussian"), (0.1, "bilateral"), (0.4, "wiener")],
)
def test_auto_elige_segun_nivel_de_ruido(
    sigma: float, esperado: str
) -> None:
    """Más ruido gaussiano lleva a filtros más fuertes."""
    noisy = _particles() + _rng().normal(0, sigma, (128, 128))

    method, _ = loader._select_method(loader._normalize(noisy))

    assert method == esperado


def test_auto_imprime_justificacion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El modo auto explica por qué eligió el filtro."""
    preprocess_tem_image(loader._normalize(_particles()), "auto")

    assert "Modo auto: se elige" in capsys.readouterr().out


# --------------------------------------------------------------------
# show_preprocessing_results y pipeline_tem
# --------------------------------------------------------------------


def test_pipeline_devuelve_dict_y_guarda_figura(tmp_path: Path) -> None:
    """El pipeline encadena todo y guarda el PNG en output_dir."""
    noisy = np.clip(_particles() + _rng().normal(0, 0.1, (128, 128)), 0, 1)
    path = _write_image(
        tmp_path / "muestra.tif", (noisy * 65535).astype(np.uint16)
    )
    out_dir = tmp_path / "figuras"

    result = pipeline_tem(path, output_dir=out_dir)

    assert set(result) == {
        "original", "processed", "filepath", "method_used",
    }
    assert result["filepath"] == path
    assert result["method_used"] in PREPROCESS_METHODS
    assert result["method_used"] != "auto"
    assert result["processed"].shape == result["original"].shape
    _assert_normalized(result["processed"])
    assert (out_dir / "muestra_preprocessing.png").is_file()


def test_pipeline_metodo_explicito(tmp_path: Path) -> None:
    """Con un método explícito, method_used es ese método."""
    path = _write_image(tmp_path / "m.png", _FORMAT_DATA[".png"])

    result = pipeline_tem(path, method="median", output_dir=tmp_path)

    assert result["method_used"] == "median"


def test_pipeline_metodo_invalido_no_carga(tmp_path: Path) -> None:
    """Un método inválido falla antes de leer el archivo."""
    with pytest.raises(ValueError, match="no válido"):
        pipeline_tem(tmp_path / "no_existe.tif", method="foo")
