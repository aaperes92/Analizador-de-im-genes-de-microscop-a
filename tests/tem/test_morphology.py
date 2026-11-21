"""Tests de src/tem/morphology.py (Etapa 3: análisis morfológico).

Todas las máscaras son sintéticas (numpy / skimage.draw), así que no
hace falta tener archivos TEM reales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.draw import disk, ellipse

from tem import morphology as morph
from tem.morphology import (
    analyze_orientation,
    analyze_shape,
    analyze_size_distribution,
    extract_region_properties,
    pipeline_morphology,
    plot_orientation_rose,
    show_morphology_results,
)

# --------------------------------------------------------------------
# Máscaras sintéticas (fase 0 = objetos, fase 1 = fondo mayoritario)
# --------------------------------------------------------------------

RADIOS = (20, 30, 40)


def _tres_circulos(size: int = 400) -> np.ndarray:
    """Máscara con 3 círculos de radios conocidos sobre fondo."""
    mask = np.ones((size, size), dtype=np.int32)
    for (cy, cx), r in zip([(80, 80), (200, 250), (310, 110)], RADIOS):
        rr, cc = disk((cy, cx), r, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def _elipses(angulos_deg: list[float], size: int = 600) -> np.ndarray:
    """Elipses (fase 0) con la orientación dada, en una grilla.

    Args:
        angulos_deg: Ángulo de cada elipse respecto de la horizontal,
            medido contra reloj como se ve la imagen.
        size: Lado de la imagen.

    Returns:
        Máscara int32.
    """
    mask = np.ones((size, size), dtype=np.int32)
    lado = int(np.ceil(np.sqrt(len(angulos_deg))))
    paso = size // lado
    for i, ang in enumerate(angulos_deg):
        cy = paso // 2 + (i // lado) * paso
        cx = paso // 2 + (i % lado) * paso
        rr, cc = ellipse(cy, cx, 8, 30, shape=mask.shape,
                         rotation=np.radians(ang))
        mask[rr, cc] = 0
    return mask


def _seg_dict(mask: np.ndarray) -> dict:
    """Imita el diccionario que devuelve pipeline_segmentation."""
    image = np.where(mask == 0, 0.3, 0.8).astype(np.float32)
    return {"original": image, "processed": image, "mask": mask,
            "n_phases": 2, "threshold_method": "otsu", "stats": {},
            "filepath": "sintetica.tif", "thresholds": np.array([0.5])}


# --------------------------------------------------------------------
# Bloque 1 — Propiedades por región
# --------------------------------------------------------------------


def test_area_de_circulos_coincide_con_pi_r2() -> None:
    """El área de cada círculo es π·r² con menos del 5 % de error."""
    mask = _tres_circulos()

    props = extract_region_properties(mask, np.zeros(mask.shape))

    areas = sorted(r["area_px"] for r in props["fase_0"])
    for area, r in zip(areas, RADIOS):
        assert area == pytest.approx(np.pi * r**2, rel=0.05)


def test_regiones_traen_todas_las_claves() -> None:
    """Cada región trae las claves de la especificación."""
    props = extract_region_properties(_tres_circulos(), np.zeros((400, 400)))

    claves = {"id", "area_px", "perimetro_px", "eje_mayor_px",
              "eje_menor_px", "orientacion_deg", "centroide", "bbox",
              "solidez", "excentricidad", "toca_borde"}
    for region in props["fase_0"]:
        assert set(region) == claves
        assert 0 <= region["orientacion_deg"] < 180


def test_excluye_fase_mayoritaria_y_respeta_phases() -> None:
    """Por defecto no analiza el fondo; con phases, la fase pedida."""
    mask = _tres_circulos()

    assert list(extract_region_properties(mask, np.zeros(mask.shape))) \
        == ["fase_0"]
    assert list(extract_region_properties(mask, np.zeros(mask.shape),
                                          phases=[1])) == ["fase_1"]


def test_marca_regiones_que_tocan_el_borde() -> None:
    """Una región pegada al borde queda con toca_borde = True."""
    mask = _tres_circulos()
    mask[0:30, 360:400] = 0  # esquina superior derecha

    props = extract_region_properties(mask, np.zeros(mask.shape))

    borde = [r for r in props["fase_0"] if r["toca_borde"]]
    assert len(borde) == 1
    assert borde[0]["bbox"][0] == 0


def test_ignora_fragmentos_chicos() -> None:
    """Las regiones de menos de CLEAN_MIN_FRACTION no se cuentan."""
    mask = _tres_circulos()
    mask[150:152, 150:152] = 0  # 4 px, por debajo del mínimo (16 px)

    props = extract_region_properties(mask, np.zeros(mask.shape))

    assert len(props["fase_0"]) == 3


@pytest.mark.parametrize("angulo", [0, 30, 90, 135])
def test_orientacion_desde_la_horizontal(angulo: float) -> None:
    """El ángulo se mide desde la horizontal, contra reloj."""
    props = extract_region_properties(_elipses([angulo]),
                                      np.zeros((600, 600)))

    medido = props["fase_0"][0]["orientacion_deg"]
    diferencia = abs((medido - angulo + 90) % 180 - 90)
    assert diferencia < 2


def test_rechaza_shapes_distintos() -> None:
    """mask e image con shapes distintos lanzan ValueError."""
    with pytest.raises(ValueError):
        extract_region_properties(np.zeros((10, 10), int),
                                  np.zeros((5, 5)))


# --------------------------------------------------------------------
# Bloque 2 — Tamaño y forma
# --------------------------------------------------------------------


def test_size_distribution_claves_y_n_regiones() -> None:
    """Trae todas las claves y n_regiones es correcto."""
    mask = np.ones((600, 600), dtype=np.int32)
    for i in range(9):  # 9 círculos de radios distintos
        rr, cc = disk((100 + 200 * (i // 3), 100 + 200 * (i % 3)),
                      10 + 4 * i, shape=mask.shape)
        mask[rr, cc] = 0
    props = extract_region_properties(mask, np.zeros(mask.shape))

    stats = analyze_size_distribution(props)["fase_0"]

    claves = {"n_regiones", "n_excluidas_borde", "area_media_px",
              "area_mediana_px", "area_std_px", "area_min_px",
              "area_max_px", "iqr_px", "percentiles", "ajuste_lognormal"}
    assert set(stats) == claves
    assert stats["n_regiones"] == 9
    assert set(stats["percentiles"]) == {"p10", "p25", "p75", "p90"}
    assert set(stats["ajuste_lognormal"]) == {
        "shape", "loc", "scale", "ks_statistic", "ks_pvalue",
        "buen_ajuste",
    }
    assert stats["ajuste_lognormal"]["loc"] == 0


def test_size_distribution_sin_ajuste_con_pocas_regiones() -> None:
    """Con menos de LOGNORM_MIN_REGIONS no se ajusta la log-normal."""
    props = extract_region_properties(_tres_circulos(),
                                      np.zeros((400, 400)))

    stats = analyze_size_distribution(props)["fase_0"]

    assert stats["n_regiones"] == 3
    assert stats["ajuste_lognormal"] is None


def test_size_distribution_lognormal_con_datos_lognormales() -> None:
    """Con áreas log-normales el ajuste se considera bueno."""
    rng = np.random.default_rng(0)
    areas = rng.lognormal(mean=7, sigma=0.5, size=200)
    props = {"fase_0": [{"area_px": float(a), "toca_borde": False}
                        for a in areas]}

    fit = analyze_size_distribution(props)["fase_0"]["ajuste_lognormal"]

    assert fit["buen_ajuste"]
    assert fit["shape"] == pytest.approx(0.5, abs=0.1)


def test_fase_sin_regiones_no_falla() -> None:
    """Una fase vacía devuelve n_regiones = 0 y valores None."""
    vacia = {"fase_0": []}

    assert analyze_size_distribution(vacia)["fase_0"]["n_regiones"] == 0
    assert analyze_shape(vacia)["fase_0"]["forma_predominante"] is None
    assert analyze_orientation(vacia)["fase_0"]["indice_orden_S"] is None


def test_circulo_tiene_circularidad_alta() -> None:
    """Un círculo sintético tiene circularidad > 0.90."""
    props = extract_region_properties(_tres_circulos(),
                                      np.zeros((400, 400)))

    shape = analyze_shape(props)["fase_0"]

    assert shape["circularidad_media"] > 0.90
    assert shape["forma_predominante"] == "circular"
    assert shape["conteo_formas"]["circular"] == 3


def test_elipses_largas_son_elongadas() -> None:
    """Elipses de relación de aspecto ~3.7 se clasifican elongadas."""
    mask = _elipses([0.0, 45.0, 90.0, 135.0])
    props = extract_region_properties(mask, np.zeros(mask.shape))

    shape = analyze_shape(props)["fase_0"]

    assert shape["relacion_aspecto_media"] > 2.0
    assert shape["forma_predominante"] == "elongada"


# --------------------------------------------------------------------
# Bloque 3 — Orientación
# --------------------------------------------------------------------


def test_rectangulos_horizontales_angulo_cero() -> None:
    """Rectángulos horizontales: ángulo medio ≈ 0° y S ≈ 1."""
    mask = np.ones((500, 500), dtype=np.int32)
    for i in range(12):
        fila = 20 + 40 * i
        mask[fila:fila + 8, 100:300] = 0

    props = extract_region_properties(mask, np.zeros(mask.shape))
    orient = analyze_orientation(props)["fase_0"]

    diferencia = abs((orient["angulo_medio_deg"] + 90) % 180 - 90)
    assert diferencia < 1
    assert orient["indice_orden_S"] == pytest.approx(1, abs=0.01)
    assert orient["clasificacion"] == "orientado"


def test_angulo_medio_respeta_periodo_180() -> None:
    """170° y 10° promedian 0°, no 90° (los ángulos son ejes)."""
    mask = _elipses([170.0, 10.0] * 6)
    props = extract_region_properties(mask, np.zeros(mask.shape))

    medio = analyze_orientation(props)["fase_0"]["angulo_medio_deg"]

    assert abs((medio + 90) % 180 - 90) < 2


def test_orientaciones_al_azar_son_aleatorias() -> None:
    """Con ángulos repartidos uniformemente, S es chico."""
    mask = _elipses(list(np.linspace(0, 180, 36, endpoint=False)))
    props = extract_region_properties(mask, np.zeros(mask.shape))

    orient = analyze_orientation(props)["fase_0"]

    assert orient["indice_orden_S"] < 0.1
    assert orient["clasificacion"] == "aleatorio"


def test_pocas_regiones_son_insuficientes() -> None:
    """Con menos de ORIENTATION_MIN_REGIONS no se clasifica."""
    mask = _elipses([0.0, 0.0, 0.0])
    props = extract_region_properties(mask, np.zeros(mask.shape))

    orient = analyze_orientation(props)["fase_0"]

    assert orient["indice_orden_S"] == pytest.approx(1, abs=0.01)
    assert orient["clasificacion"] == "insuficiente"


def test_circulos_no_cuentan_para_la_orientacion() -> None:
    """Las regiones casi circulares se excluyen de la orientación."""
    props = extract_region_properties(_tres_circulos(),
                                      np.zeros((400, 400)))

    orient = analyze_orientation(props)["fase_0"]

    assert orient["n_regiones"] == 0
    assert orient["n_excluidas_isotropas"] == 3


# --------------------------------------------------------------------
# Bloques 4 y 5 — Figuras y pipeline
# --------------------------------------------------------------------


def test_figuras_se_guardan(tmp_path: Path) -> None:
    """La rosa y la figura morfológica se guardan con su nombre."""
    mask = _elipses([0.0, 30.0, 60.0, 90.0])
    image = np.where(mask == 0, 0.3, 0.8)
    props = extract_region_properties(mask, image)

    rosa = plot_orientation_rose(props, "muestra.dm3",
                                 output_dir=tmp_path, show=False)
    figura = show_morphology_results(
        image, mask, props, analyze_size_distribution(props),
        analyze_shape(props), "muestra.dm3", output_dir=tmp_path,
        show=False,
    )

    assert rosa == tmp_path / "muestra_orientacion.png"
    assert figura == tmp_path / "muestra_morfologia.png"
    assert rosa.is_file() and figura.is_file()


def test_pipeline_morphology_devuelve_claves(tmp_path: Path) -> None:
    """El resultado trae todas las claves de la especificación."""
    result = pipeline_morphology(_seg_dict(_tres_circulos()),
                                 output_dir=tmp_path, show=False)

    assert set(result) == {
        "original", "processed", "mask", "props", "size_stats",
        "shape_stats", "orientation_stats", "filepath", "n_phases",
    }
    assert result["n_phases"] == 2
    assert len(result["props"]["fase_0"]) == 3
    assert (tmp_path / "sintetica_morfologia.png").is_file()
    assert (tmp_path / "sintetica_orientacion.png").is_file()


def test_pipeline_morphology_valida_entrada() -> None:
    """Un diccionario incompleto lanza ValueError."""
    with pytest.raises(ValueError, match="falta"):
        pipeline_morphology({"mask": np.zeros((5, 5), int)})


def test_pipeline_morphology_con_phases(tmp_path: Path) -> None:
    """El parámetro phases elige qué fases analizar."""
    result = pipeline_morphology(_seg_dict(_tres_circulos()), phases=[1],
                                 output_dir=tmp_path, show=False)

    assert list(result["props"]) == ["fase_1"]


def test_random_s_decrece_con_n() -> None:
    """El S esperado al azar baja como 1/sqrt(n)."""
    assert morph._random_s(4) == pytest.approx(2 * morph._random_s(16))
    assert morph._random_s(0) == 0.0
