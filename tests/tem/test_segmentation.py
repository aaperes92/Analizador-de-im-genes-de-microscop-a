"""Tests de src/tem/segmentation.py (Etapa 2: segmentación de fases).

Todas las imágenes son sintéticas (numpy), así que no hace falta tener
archivos TEM reales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import filters, measure

from tem import segmentation as seg
from tem.segmentation import (
    INTERFACE_LABEL,
    analyze_histogram,
    apply_segmentation,
    clean_mask,
    compute_phase_stats,
    detect_interfaces,
    detect_n_phases,
    pipeline_segmentation,
    select_threshold_method,
    separate_touching_regions,
    show_segmentation_results,
    histogram_skewness,
    threshold_adaptive_local,
    threshold_gmm,
    threshold_otsu_multi,
    threshold_triangle_method,
)

# --------------------------------------------------------------------
# Imágenes sintéticas
# --------------------------------------------------------------------


def _rng() -> np.random.Generator:
    """Generador aleatorio con semilla fija."""
    return np.random.default_rng(42)


def _niveles(niveles: tuple[float, ...], size: int = 240,
             ruido: float = 0.03) -> np.ndarray:
    """Imagen con franjas verticales de intensidades dadas más ruido.

    Args:
        niveles: Intensidad de cada franja (una franja por fase).
        size: Lado de la imagen.
        ruido: Desvío del ruido gaussiano.

    Returns:
        Imagen 2D float32 recortada a [0, 1].
    """
    image = np.zeros((size, size))
    for i, nivel in enumerate(niveles):
        image[:, i * size // len(niveles):(i + 1) * size // len(niveles)] \
            = nivel
    image += _rng().normal(0, ruido, image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def _dos_circulos(size: int = 400, radio: int = 30,
                  separacion: int = 55) -> np.ndarray:
    """Máscara con dos círculos (fase 0) tocándose sobre fondo (fase 1).

    Args:
        size: Lado de la imagen.
        radio: Radio de cada círculo.
        separacion: Distancia entre centros (< 2 * radio: se tocan).

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    yy, xx = np.mgrid[:size, :size]
    centro = size // 2
    mask = np.ones((size, size), dtype=np.int32)
    for cx in (centro - separacion // 2, centro + separacion // 2):
        mask[(yy - centro) ** 2 + (xx - cx) ** 2 < radio**2] = 0
    return mask


def _particulas(size: int = 300) -> np.ndarray:
    """Imagen tipo campo claro: partículas oscuras sobre fondo claro.

    Args:
        size: Lado de la imagen.

    Returns:
        Imagen 2D float32 en [0, 1].
    """
    rng = _rng()
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size), 0.75)
    for _ in range(10):
        cy, cx = rng.integers(20, size - 20, 2)
        r = rng.integers(8, 18)
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 0.25
    image = filters.gaussian(image, sigma=1) + rng.normal(0, 0.02,
                                                          image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def _campo_oscuro(size: int = 300) -> np.ndarray:
    """Imagen tipo campo oscuro: fondo oscuro y partículas brillantes.

    Una partícula saturada y varias tenues, como en las imágenes
    reales: con Otsu, la saturada se lleva el umbral.

    Args:
        size: Lado de la imagen.

    Returns:
        Imagen 2D float32 en [0, 1].
    """
    rng = _rng()
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size), 0.05)
    centros = [(60, 60), (60, 200), (150, 120), (230, 70), (220, 220)]
    for cy, cx in centros:
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < 12**2] = 0.2  # tenues
    image[(yy - 270) ** 2 + (xx - 270) ** 2 < 15**2] = 1.0  # saturada
    image = filters.gaussian(image, sigma=1) + rng.normal(0, 0.01,
                                                          image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


# --------------------------------------------------------------------
# Bloque 1 — Detección de fases
# --------------------------------------------------------------------


def test_analyze_histogram_devuelve_claves() -> None:
    """El diccionario trae las claves de la especificación."""
    info = analyze_histogram(_niveles((0.2, 0.8)))

    for clave in ("n_peaks", "peaks_positions", "valleys_positions",
                  "histogram", "bin_edges"):
        assert clave in info
    assert info["histogram"].size == 256
    assert info["bin_edges"].size == 257
    assert len(info["valleys_positions"]) == max(0, info["n_peaks"] - 1)


def test_detect_n_phases_imagen_uniforme_da_minimo_2() -> None:
    """Una imagen uniforme (sin picos separados) da 2 fases."""
    assert detect_n_phases(np.full((64, 64), 0.5, np.float32)) == 2


def test_detect_n_phases_ruido_uniforme_da_minimo_2() -> None:
    """Ruido sin estructura también da el mínimo de 2 fases."""
    image = _rng().random((128, 128)).astype(np.float32)
    assert detect_n_phases(image) >= 2


@pytest.mark.parametrize("niveles", [(0.2, 0.8), (0.15, 0.5, 0.85)])
def test_detect_n_phases_cuenta_niveles(niveles: tuple[float, ...]) -> None:
    """Con niveles bien separados detecta un fase por nivel."""
    assert detect_n_phases(_niveles(niveles)) == len(niveles)


def test_detect_n_phases_fusiona_pico_sin_masa() -> None:
    """Un nivel con menos del 5 % de los píxeles no es una fase."""
    image = _niveles((0.2, 0.8))
    image[:5, :5] = 0.5  # 25 píxeles de 57 600: muy poca masa
    assert detect_n_phases(image) == 2


# --------------------------------------------------------------------
# Bloque 2 — Umbralización
# --------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_otsu_y_gmm_devuelven_n_menos_1_umbrales(n: int) -> None:
    """Otsu y GMM dan n - 1 umbrales ordenados en [0, 1]."""
    image = _niveles(tuple(np.linspace(0.1, 0.9, n)))

    for umbrales in (threshold_otsu_multi(image, n),
                     threshold_gmm(image, n)):
        assert umbrales.size == n - 1
        assert np.all(np.diff(umbrales) > 0)
        assert umbrales.min() > 0 and umbrales.max() < 1


def test_gmm_es_reproducible() -> None:
    """Con la semilla fija, la misma imagen da los mismos umbrales."""
    image = _niveles((0.2, 0.5, 0.8))
    np.testing.assert_array_equal(threshold_gmm(image, 3),
                                  threshold_gmm(image, 3))


def test_adaptive_local_devuelve_mascara_y_bloque_impar(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El umbral local da una máscara booleana con bloque impar."""
    image = _particulas(200)

    mask = threshold_adaptive_local(image)

    assert mask.dtype == bool and mask.shape == image.shape
    # 10 % de 200 = 20 → impar más cercano: 19 o 21
    assert "block_size = 21" in capsys.readouterr().out


def test_adaptive_local_rechaza_bloque_par() -> None:
    """Un block_size par lanza ValueError."""
    with pytest.raises(ValueError):
        threshold_adaptive_local(_particulas(100), block_size=20)


@pytest.mark.parametrize(("n", "esperado"), [(2, "otsu"), (3, "otsu"),
                                             (4, "gmm")])
def test_select_threshold_method(n: int, esperado: str) -> None:
    """Hasta 3 fases elige Otsu; con más, GMM."""
    image = _niveles(tuple(np.linspace(0.1, 0.9, n)))

    metodo, umbrales = select_threshold_method(image, n)

    assert metodo == esperado
    assert umbrales.size == n - 1


def test_triangle_devuelve_un_umbral() -> None:
    """El método del triángulo da un único umbral en [0, 1]."""
    umbral = threshold_triangle_method(_campo_oscuro())

    assert umbral.shape == (1,)
    assert 0 < umbral[0] < 1


def test_asimetria_separa_campo_oscuro_de_campo_claro() -> None:
    """Campo oscuro: asimetría alta; campo claro: baja o negativa."""
    assert histogram_skewness(_campo_oscuro()) > seg.DARKFIELD_SKEW
    assert histogram_skewness(_particulas()) < seg.DARKFIELD_SKEW
    assert histogram_skewness(np.full((10, 10), 0.5)) == 0.0


def test_select_elige_triangle_en_campo_oscuro() -> None:
    """Con 2 fases y asimetría alta, el modo auto usa el triángulo."""
    metodo, umbrales = select_threshold_method(_campo_oscuro(), 2)

    assert metodo == "triangle"
    assert umbrales.size == 1


def test_triangle_encuentra_las_particulas_tenues() -> None:
    """El triángulo detecta las partículas tenues que Otsu pierde."""
    image = _campo_oscuro()
    yy, xx = np.mgrid[:300, :300]
    tenue = (yy - 150) ** 2 + (xx - 120) ** 2 < 8**2

    otsu = image > threshold_otsu_multi(image, 2)[0]
    triangulo = image > threshold_triangle_method(image)[0]

    assert otsu[tenue].mean() < 0.5        # Otsu la pierde
    assert triangulo[tenue].mean() > 0.9   # el triángulo la encuentra


def test_select_no_usa_triangle_con_mas_de_2_fases() -> None:
    """Con 3 fases, aunque la asimetría sea alta, no usa el triángulo."""
    metodo, _ = select_threshold_method(_campo_oscuro(), 3)

    assert metodo == "otsu"


# --------------------------------------------------------------------
# Bloque 3 — Separación y limpieza
# --------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_apply_segmentation_etiquetas_en_rango(n: int) -> None:
    """Las etiquetas van de 0 a n_phases - 1 y conservan el shape."""
    image = _rng().random((100, 120)).astype(np.float32)
    umbrales = np.linspace(0, 1, n + 1)[1:-1]

    mask = apply_segmentation(image, umbrales)

    assert mask.shape == image.shape
    assert mask.min() == 0 and mask.max() == n - 1
    assert set(np.unique(mask)) == set(range(n))


def test_apply_segmentation_con_mascara_booleana() -> None:
    """Una máscara booleana pasa a etiquetas 0 y 1."""
    booleana = _rng().random((50, 50)) > 0.5

    mask = apply_segmentation(np.zeros((50, 50)), booleana)

    np.testing.assert_array_equal(mask, booleana.astype(int))


def test_clean_mask_no_cambia_el_numero_de_fases() -> None:
    """La limpieza conserva todas las fases presentes."""
    image = _niveles((0.15, 0.5, 0.85), ruido=0.12)
    mask = apply_segmentation(image, threshold_otsu_multi(image, 3))

    limpia = clean_mask(mask)

    assert set(np.unique(limpia)) == set(np.unique(mask))
    assert (limpia >= 0).all()  # ningún píxel queda sin fase


def test_clean_mask_conserva_fase_de_solo_regiones_chicas() -> None:
    """Una fase hecha solo de manchas chicas no desaparece."""
    mask = np.zeros((200, 200), dtype=np.int32)
    mask[10:13, 10:13] = 1  # 9 píxeles, mucho menos que el mínimo
    mask[50:53, 80:83] = 1

    limpia = clean_mask(mask, min_size=100)

    assert set(np.unique(limpia)) == {0, 1}


def test_clean_mask_elimina_regiones_chicas() -> None:
    """Las regiones más chicas que min_size pasan a la fase vecina."""
    mask = np.zeros((200, 200), dtype=np.int32)
    mask[50:150, 50:150] = 1  # región grande
    mask[5:8, 5:8] = 1        # mancha chica, aislada

    limpia = clean_mask(mask, min_size=100)

    assert limpia[6, 6] == 0
    assert limpia[100, 100] == 1


def test_separate_touching_regions_separa_dos_circulos() -> None:
    """Dos círculos tocándose quedan como dos regiones separadas."""
    mask = _dos_circulos()
    antes = measure.label(mask == 0, connectivity=1).max()

    separada = separate_touching_regions(mask)
    despues = measure.label(separada == 0, connectivity=1).max()

    assert antes == 1
    assert despues == 2
    assert (separada == INTERFACE_LABEL).any()  # hay línea de corte


def test_separate_touching_regions_sin_contacto_no_cambia() -> None:
    """Si los círculos no se tocan, la máscara no cambia."""
    mask = _dos_circulos(separacion=100)

    np.testing.assert_array_equal(separate_touching_regions(mask), mask)


def test_separate_touching_regions_no_corta_regiones_grandes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Las regiones de más de WATERSHED_MAX_REGION no se cortan."""
    monkeypatch.setattr(seg, "WATERSHED_MAX_REGION", 0.01)
    mask = _dos_circulos()  # ~3.5 % de la imagen: ahora es "grande"

    np.testing.assert_array_equal(separate_touching_regions(mask), mask)


def test_detect_interfaces_marca_borde_de_2_pixeles() -> None:
    """El borde entre dos fases queda marcado con 2 píxeles de ancho."""
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[:, 10:] = 1

    out = detect_interfaces(mask)

    np.testing.assert_array_equal(
        np.flatnonzero(out[5] == INTERFACE_LABEL), [9, 10]
    )
    assert set(np.unique(out)) == {INTERFACE_LABEL, 0, 1}


# --------------------------------------------------------------------
# Bloque 4 — Estadísticas
# --------------------------------------------------------------------


def test_compute_phase_stats_porcentajes_suman_100() -> None:
    """Los porcentajes de las fases (sin interfaces) suman 100 %."""
    image = _particulas()
    mask = detect_interfaces(
        clean_mask(apply_segmentation(image, [0.5]))
    )

    stats = compute_phase_stats(image, mask)

    fases = [k for k in stats if k != "interfaces"]
    assert sum(stats[k]["area_pct"] for k in fases) == pytest.approx(100)


def test_compute_phase_stats_estructura_y_valores() -> None:
    """Cada fase trae todas las claves y valores coherentes."""
    image = _particulas()
    mask = detect_interfaces(apply_segmentation(image, [0.5]))

    stats = compute_phase_stats(image, mask)

    claves = {"area_px", "area_pct", "intensidad_media",
              "intensidad_std", "n_regiones", "tamanio_medio_px",
              "tamanio_min_px", "tamanio_max_px", "n_fragmentos"}
    assert set(stats) == {"fase_0", "fase_1", "interfaces"}
    for k in ("fase_0", "fase_1"):
        assert set(stats[k]) == claves
        assert (stats[k]["tamanio_min_px"] <= stats[k]["tamanio_medio_px"]
                <= stats[k]["tamanio_max_px"])
    # Las partículas son la fase oscura
    assert (stats["fase_0"]["intensidad_media"]
            < stats["fase_1"]["intensidad_media"])
    assert stats["interfaces"]["area_px"] == int((mask == -1).sum())


def test_compute_phase_stats_rechaza_shapes_distintos() -> None:
    """image y mask con shapes distintos lanzan ValueError."""
    with pytest.raises(ValueError):
        compute_phase_stats(np.zeros((10, 10)), np.zeros((5, 5), int))


# --------------------------------------------------------------------
# Bloques 5 y 6 — Figura y pipeline
# --------------------------------------------------------------------


def test_show_segmentation_results_guarda_png(tmp_path: Path) -> None:
    """La figura se guarda con el nombre esperado."""
    image = _particulas()
    mask = detect_interfaces(apply_segmentation(image, [0.5]))
    stats = compute_phase_stats(image, mask)

    out = show_segmentation_results(
        image, mask, stats, "muestra.dm3", thresholds=np.array([0.5]),
        output_dir=tmp_path, show=False,
    )

    assert out == tmp_path / "muestra_segmentation.png"
    assert out.is_file()


def _dict_preprocesado(path: str = "sintetica.tif") -> dict:
    """Imita el diccionario que devuelve pipeline_tem."""
    image = _particulas()
    return {"original": image, "processed": image, "filepath": path,
            "method_used": "gaussian"}


def test_pipeline_segmentation_devuelve_claves(tmp_path: Path) -> None:
    """El resultado trae todas las claves de la especificación."""
    result = pipeline_segmentation(
        _dict_preprocesado(), output_dir=tmp_path, show=False
    )

    esperadas = {"original", "processed", "mask", "n_phases",
                 "threshold_method", "stats", "filepath"}
    assert esperadas <= set(result)
    assert result["threshold_method"] in ("otsu", "gmm")
    assert result["n_phases"] >= 2
    assert result["mask"].shape == result["processed"].shape
    assert (tmp_path / "sintetica_segmentation.png").is_file()


@pytest.mark.parametrize("method", ["otsu", "gmm", "triangle", "adaptive"])
def test_pipeline_segmentation_metodo_explicito(
    tmp_path: Path, method: str
) -> None:
    """Con un método explícito, threshold_method es ese método."""
    result = pipeline_segmentation(
        _dict_preprocesado(), method=method, output_dir=tmp_path,
        show=False,
    )

    assert result["threshold_method"] == method
    if method == "adaptive":
        assert result["thresholds"] is None
        assert result["n_phases"] == 2


def test_pipeline_segmentation_valida_entradas() -> None:
    """Método inválido o diccionario incompleto lanzan ValueError."""
    with pytest.raises(ValueError, match="no válido"):
        pipeline_segmentation(_dict_preprocesado(), method="kmeans")
    with pytest.raises(ValueError, match="falta"):
        pipeline_segmentation({"original": np.zeros((5, 5))})
