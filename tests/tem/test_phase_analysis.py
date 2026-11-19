"""Tests de la Etapa 4: análisis de fases.

Módulo A (fraction.py), Módulo B (compositional.py) y Módulo C
(spatial.py). Todas las máscaras e imágenes son sintéticas.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from skimage.draw import disk

from tem import fraction as frac
from tem.compositional import (
    build_phase_intensity_map,
    compute_intensity_gradients,
    compute_intensity_profiles,
    detect_inhomogeneities,
    pipeline_compositional,
    show_compositional_results,
)
from tem.fraction import (
    compute_area_fractions,
    compute_interface_fraction,
    compute_number_density,
    estimate_volume_fractions,
    phase_names,
    pipeline_fraction,
    show_fraction_results,
)
from tem.morphology import extract_region_properties
from tem.segmentation import detect_interfaces
from tem.spatial import (
    compute_density_map,
    compute_rdf,
    detect_clustering,
    detect_spatial_gradients,
    pipeline_spatial,
    show_spatial_results,
)

# --------------------------------------------------------------------
# Máscaras sintéticas
# --------------------------------------------------------------------


def _silencio(func, *args, **kwargs):
    """Llama a func sin imprimir en pantalla."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _circulos(centros: list[tuple[int, int]], radio: int = 10,
              size: int = 600) -> np.ndarray:
    """Círculos de fase 0 sobre fondo mayoritario (fase 1)."""
    mask = np.ones((size, size), dtype=np.int32)
    for c in centros:
        rr, cc = disk(c, radio, shape=mask.shape)
        mask[rr, cc] = 0
    return mask


def _grilla(n_lado: int = 10, paso: int = 60, radio: int = 8) -> np.ndarray:
    """Círculos en una grilla regular (n_lado × n_lado)."""
    centros = [(paso // 2 + i * paso, paso // 2 + j * paso)
               for i in range(n_lado) for j in range(n_lado)]
    return _circulos(centros, radio, size=n_lado * paso)


def _franjas() -> np.ndarray:
    """Tres franjas verticales con interfaces: fondo | 0 | 2.

    El fondo (etiqueta 1) es la franja más ancha; la etiqueta 0 pasa a
    ser Fase 1 y la 2, Fase 2.
    """
    mask = np.ones((200, 400), dtype=np.int32)
    mask[:, 200:300] = 0
    mask[:, 300:] = 2
    return _silencio(detect_interfaces, mask)


def _props(mask: np.ndarray) -> dict:
    """Propiedades de la Etapa 3 de todas las fases menos el fondo."""
    return _silencio(extract_region_properties, mask, np.zeros(mask.shape))


# ====================================================================
# MÓDULO A — fraction.py
# ====================================================================

# --- Nombres de fases ------------------------------------------------


def test_nombres_fondo_y_fases_desde_1() -> None:
    """El fondo es la mayoritaria; el resto se numera desde 1."""
    bg, names = phase_names(_franjas())

    assert bg == 1
    assert names == {"fase_1": 0, "fase_2": 2}


def test_background_elegido_a_mano() -> None:
    """Con background se elige el fondo; uno inexistente da error."""
    bg, names = phase_names(_franjas(), background=0)

    assert bg == 0
    assert names == {"fase_1": 1, "fase_2": 2}
    with pytest.raises(ValueError, match="no es una fase"):
        phase_names(_franjas(), background=7)


# --- A1: fracciones de área y volumen --------------------------------


def test_fracciones_suman_100() -> None:
    """Fases + interfaces + fondo = 100 %; fases sin fondo = 100 %."""
    areas = _silencio(compute_area_fractions, _franjas())

    fases = [k for k in areas if k.startswith("fase_")]
    total = (sum(areas[k]["area_pct_con_fondo"] for k in fases)
             + areas["interfaces"]["area_pct"] + areas["fondo"]["area_pct"])
    assert total == pytest.approx(100, abs=0.01)
    assert sum(areas[k]["area_pct"] for k in fases) == pytest.approx(
        100, abs=0.01)
    assert areas["total_px"] == 200 * 400


def test_fracciones_exactas_sin_interfaces() -> None:
    """Sin interfaces, las fracciones son las áreas exactas."""
    mask = np.ones((100, 100), dtype=np.int32)
    mask[:, :25] = 0   # 25 % → Fase 1
    mask[:, 25:40] = 2  # 15 % → Fase 2

    areas = _silencio(compute_area_fractions, mask)

    assert areas["fondo"]["area_pct"] == pytest.approx(60)
    assert areas["fase_1"]["area_pct_con_fondo"] == pytest.approx(25)
    assert areas["fase_1"]["area_pct"] == pytest.approx(25 / 40 * 100)
    assert areas["interfaces"]["area_px"] == 0


def test_fracciones_que_no_suman_dan_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si las sumas se apartan de 100 % se lanza ValueError."""
    # Con tolerancia negativa ninguna suma puede pasar la verificación
    monkeypatch.setattr(frac, "SUM_TOLERANCE_PCT", -1.0)

    with pytest.raises(ValueError, match="no suman 100"):
        _silencio(compute_area_fractions, _franjas())


def test_una_sola_fase_no_falla() -> None:
    """Una máscara de una sola fase es todo fondo, sin error."""
    areas = _silencio(compute_area_fractions, np.zeros((10, 10), int))

    assert areas["fondo"]["area_pct"] == pytest.approx(100)
    assert not any(k.startswith("fase_") for k in areas)


def test_volumen_igual_a_area_sobre_100() -> None:
    """vol_frac_estimada ≈ area_pct / 100 en cada fase."""
    areas = _silencio(compute_area_fractions, _franjas())

    vols = _silencio(estimate_volume_fractions, areas)

    for name in ("fase_1", "fase_2"):
        v = vols[name]
        assert v["vol_frac_estimada"] == pytest.approx(
            areas[name]["area_pct"] / 100)
        assert v["vol_frac_estimada_con_fondo"] == pytest.approx(
            areas[name]["area_pct_con_fondo"] / 100)
        assert "proyección" in v["advertencia"]


def test_volumen_no_modifica_la_entrada() -> None:
    """estimate_volume_fractions devuelve una copia."""
    areas = _silencio(compute_area_fractions, _franjas())

    _silencio(estimate_volume_fractions, areas)

    assert "vol_frac_estimada" not in areas["fase_1"]


def test_volumen_avisa_con_una_sola_fase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con una sola fase fuera del fondo, avisa que 'sin fondo' = 100 %."""
    areas = _silencio(compute_area_fractions, _grilla())

    vols = estimate_volume_fractions(areas)

    assert vols["fase_1"]["vol_frac_pct"] == pytest.approx(100)
    assert "una sola fase" in capsys.readouterr().out


# --- A2: densidad numérica -------------------------------------------


@pytest.mark.parametrize("n", [1, 5, 25])
def test_densidad_cuenta_n_circulos(n: int) -> None:
    """Con N círculos conocidos, n_total == N."""
    rng = np.random.default_rng(n)
    centros = []
    while len(centros) < n:  # círculos sin superponerse
        c = tuple(int(v) for v in rng.integers(30, 570, 2))
        if all(np.hypot(c[0] - a, c[1] - b) > 30 for a, b in centros):
            centros.append(c)
    mask = _circulos(centros)

    dens = _silencio(compute_number_density, _props(mask), mask)

    assert dens["fase_1"]["n_total"] == n
    assert dens["fase_1"]["densidad_por_1000px2"] == pytest.approx(
        1000 * n / mask.size)


def test_densidad_una_region_sin_separacion() -> None:
    """Con una sola región, la separación queda en None."""
    mask = _circulos([(300, 300)])

    d = _silencio(compute_number_density, _props(mask), mask)["fase_1"]

    assert d["n_total"] == 1
    assert d["separacion_media_px"] is None
    assert d["indice_clark_evans"] is None


def test_clark_evans_grilla_regular() -> None:
    """En una grilla cuadrada, separación = paso y R = 2."""
    mask = _grilla(paso=60)

    d = _silencio(compute_number_density, _props(mask), mask)["fase_1"]

    assert d["separacion_media_px"] == pytest.approx(60, abs=0.5)
    assert d["separacion_std_px"] == pytest.approx(0, abs=0.5)
    assert d["indice_clark_evans"] == pytest.approx(2.0, abs=0.05)


def test_clark_evans_agrupadas_menor_que_1() -> None:
    """Partículas en un solo grupo: R muy por debajo de 1."""
    centros = [(280 + 25 * i, 280 + 25 * j)
               for i in range(3) for j in range(3)]
    mask = _circulos(centros, radio=8, size=1200)

    d = _silencio(compute_number_density, _props(mask), mask)["fase_1"]

    assert d["indice_clark_evans"] < 0.5


def test_densidad_sin_props_de_la_fase() -> None:
    """Si props no tiene la fase, sus valores quedan en None."""
    mask = _franjas()

    dens = _silencio(compute_number_density, {}, mask)

    assert dens["fase_1"]["n_total"] is None


# --- A3: interfaces --------------------------------------------------


def test_interfaces_pares_y_longitudes() -> None:
    """Cada borde de 200 px mide 200 px y la fracción es 0.5."""
    r = _silencio(compute_interface_fraction, _franjas())

    pares = r["pares_adyacentes"]
    assert set(pares) == {"fondo-fase_1", "fase_1-fase_2"}
    for p in pares.values():
        assert p["longitud_px"] == pytest.approx(200)
        assert p["fraccion"] == pytest.approx(0.5)
    assert r["fraccion_total_interfaces"] == pytest.approx(
        800 / (200 * 400))


def test_contactos_con_y_sin_lineas_watershed() -> None:
    """Con las líneas se miden los contactos; sin ellas quedan None."""
    mask = np.ones((200, 400), dtype=np.int32)
    mask[:, 200:300] = 0
    mask[:, 300:] = 2
    mask[80, 200:300] = -1  # línea de corte dentro de la etiqueta 0
    lineas = mask == -1
    final = _silencio(detect_interfaces, mask)

    con = _silencio(compute_interface_fraction, final,
                    watershed_lines=lineas)
    sin = _silencio(compute_interface_fraction, final)

    assert con["contactos_misma_fase"]["fase_1"]["longitud_px"] == 100
    assert con["contactos_misma_fase"]["fase_2"]["longitud_px"] == 0
    assert sin["contactos_misma_fase"] is None


def test_sin_interfaces() -> None:
    """Una máscara sin -1 no tiene pares ni fracción de interfaces."""
    r = _silencio(compute_interface_fraction, np.zeros((20, 20), int))

    assert r["fraccion_total_interfaces"] == 0
    assert r["pares_adyacentes"] == {}


# --- A4: figura y pipeline -------------------------------------------


def _morph_dict(mask: np.ndarray) -> dict:
    """Imita el diccionario de pipeline_morphology."""
    image = np.where(mask == 0, 0.3, 0.8).astype(np.float32)
    return {"original": image, "processed": image, "mask": mask,
            "props": _props(mask), "filepath": "sintetica.tif",
            "n_phases": 2, "lineas_watershed": np.zeros(mask.shape, bool)}


def test_figura_de_fracciones_se_guarda(tmp_path: Path) -> None:
    """La figura se guarda como '<nombre>_fracciones.png'."""
    mask = _grilla()
    areas = _silencio(compute_area_fractions, mask)

    out = _silencio(
        show_fraction_results, mask, areas,
        _silencio(estimate_volume_fractions, areas),
        _silencio(compute_number_density, _props(mask), mask),
        "muestra.dm3", output_dir=tmp_path, show=False,
    )

    assert out == tmp_path / "muestra_fracciones.png"
    assert out.is_file()


def test_pipeline_fraction_agrega_claves(tmp_path: Path) -> None:
    """Devuelve la entrada más las métricas del Módulo A."""
    entrada = _morph_dict(_grilla())

    r = _silencio(pipeline_fraction, entrada, output_dir=tmp_path,
                  show=False)

    for clave in entrada:
        assert clave in r
    for clave in ("area_fractions", "volume_fractions", "number_density",
                  "interface_fraction", "background"):
        assert clave in r
    assert r["number_density"]["fase_1"]["n_total"] == 100
    assert r["interface_fraction"]["contactos_misma_fase"] is not None
    assert (tmp_path / "sintetica_fracciones.png").is_file()


def test_pipeline_fraction_valida_entrada() -> None:
    """Un diccionario incompleto lanza ValueError."""
    with pytest.raises(ValueError, match="falta"):
        pipeline_fraction({"mask": np.zeros((5, 5), int)})


# ====================================================================
# MÓDULO B — compositional.py
# ====================================================================


def _dos_fases(sigma_borde: float = 0.0, ruido: float = 0.02,
               size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Imagen de 2 fases (0.2 a la izquierda, 0.8 a la derecha).

    Args:
        sigma_borde: Suavizado gaussiano del borde (px).
        ruido: Desvío del ruido gaussiano.
        size: Lado de la imagen.

    Returns:
        Tupla (imagen, máscara con interfaces). La fase clara (0.8)
        ocupa más área: es el fondo; la oscura (0.2) es Fase 1.
    """
    mask = np.ones((size, size), dtype=np.int32)
    mask[:, : size * 2 // 5] = 0
    image = np.where(mask == 0, 0.2, 0.8).astype(np.float64)
    if sigma_borde:
        image = ndimage.gaussian_filter(image, sigma_borde)
    image = image + np.random.default_rng(0).normal(0, ruido, image.shape)
    return image, _silencio(detect_interfaces, mask)


# --- B1: perfiles de intensidad --------------------------------------


def test_perfiles_medias_coinciden() -> None:
    """Con fases de 0.2 y 0.8, las medias coinciden ±0.05."""
    image, mask = _dos_fases()

    r = _silencio(compute_intensity_profiles, image, mask)

    assert r["perfiles"]["fase_1"]["media"] == pytest.approx(0.2, abs=0.05)
    assert r["perfiles"]["fondo"]["media"] == pytest.approx(0.8, abs=0.05)
    for p in r["perfiles"].values():
        assert p["histograma"].size == 100
        assert p["iqr"] > 0


def test_contraste_y_clasificacion() -> None:
    """|0.8 - 0.2| / (0.8 + 0.2) = 0.6 → contraste alto."""
    image, mask = _dos_fases()

    c = _silencio(compute_intensity_profiles, image,
                  mask)["contraste"]["fondo-fase_1"]

    assert c["indice"] == pytest.approx(0.6, abs=0.02)
    assert c["clasificacion"] == "alto"


@pytest.mark.parametrize(("oscura", "clara", "indice", "clase"),
                         [(0.47, 0.53, 0.06, "bajo"),
                          (0.4, 0.6, 0.2, "medio")])
def test_contraste_bajo_y_medio(oscura: float, clara: float,
                                indice: float, clase: str) -> None:
    """Contraste 0.06 → bajo (< 0.1); 0.2 → medio (0.1-0.3)."""
    mask = np.ones((100, 100), dtype=np.int32)
    mask[:, :40] = 0
    image = np.where(mask == 0, oscura, clara)

    c = _silencio(compute_intensity_profiles, image,
                  mask)["contraste"]["fondo-fase_1"]

    assert c["indice"] == pytest.approx(indice)
    assert c["clasificacion"] == clase


@pytest.mark.parametrize("sigma", [2, 3, 6])
def test_ancho_de_transicion_teorico(sigma: float) -> None:
    """El ancho 10-90 % coincide con 2.56·σ dentro de ~1 px."""
    image, mask = _dos_fases(sigma_borde=sigma, ruido=0.0)

    g = _silencio(compute_intensity_gradients, image,
                  mask)["fondo-fase_1"]

    assert g["meseta_alcanzada"]
    assert g["ancho_transicion_px"] == pytest.approx(2.563 * sigma, abs=1.2)
    assert g["distancias_px"].size == g["perfil"].size == 41


def test_borde_mas_ancho_que_la_ventana_sin_meseta() -> None:
    """Un borde más ancho que la ventana se marca sin meseta."""
    image, mask = _dos_fases(sigma_borde=14, ruido=0.0)

    g = _silencio(compute_intensity_gradients, image,
                  mask)["fondo-fase_1"]

    assert not g["meseta_alcanzada"]


def test_sin_cambio_de_intensidad_no_hay_ancho() -> None:
    """Sin escalón de intensidad, el ancho queda en None."""
    _, mask = _dos_fases()

    g = _silencio(compute_intensity_gradients, np.full(mask.shape, 0.5),
                  mask)["fondo-fase_1"]

    assert g["ancho_transicion_px"] is None


# --- B2: mapas e inhomogeneidades ------------------------------------


def test_zscore_global_media_0_desvio_1() -> None:
    """Respecto de cada fase, el z-score tiene media 0 y desvío 1."""
    image, mask = _dos_fases(ruido=0.05)

    maps = build_phase_intensity_map(image, mask)

    z = maps["zscore_global"]
    assert z.dtype == np.float32
    assert np.isnan(z[mask == -1]).all()
    assert float(np.nanmean(z)) == pytest.approx(0, abs=1e-3)
    assert float(np.nanstd(z)) == pytest.approx(1, abs=1e-3)
    assert np.isnan(maps["fase_1"]["intensidad"][mask != 0]).all()


def test_imagen_uniforme_sin_inhomogeneidades() -> None:
    """Una imagen uniforme no tiene inhomogeneidades."""
    _, mask = _dos_fases()

    r = _silencio(detect_inhomogeneities, np.full(mask.shape, 0.5), mask)

    for d in r.values():
        assert d["n_alta"] == d["n_baja"] == 0


def test_ruido_puro_sin_falsos_positivos() -> None:
    """Con ruido gaussiano y sin anomalías, no se marca nada."""
    image, mask = _dos_fases(ruido=0.05, size=800)

    r = _silencio(detect_inhomogeneities, image, mask)

    assert sum(d["n_alta"] + d["n_baja"] for d in r.values()) == 0


def test_detecta_mancha_clara_y_oscura() -> None:
    """Una mancha clara en la fase y una oscura en el fondo."""
    image, mask = _dos_fases(ruido=0.05, size=800)
    image[100:140, 100:140] += 0.25   # dentro de la Fase 1 (x < 320)
    image[500:550, 600:650] -= 0.25   # dentro del fondo

    r = _silencio(detect_inhomogeneities, image, mask)

    assert r["fase_1"]["n_alta"] == 1 and r["fase_1"]["n_baja"] == 0
    assert r["fondo"]["n_baja"] == 1 and r["fondo"]["n_alta"] == 0
    zona = r["fase_1"]["alta"][0]
    assert zona["centroide"][0] == pytest.approx(119.5, abs=3)
    assert zona["z_medio"] > 2.5


# --- B3: figura y pipeline -------------------------------------------


def _fraction_dict() -> dict:
    """Imita el diccionario de pipeline_fraction."""
    image, mask = _dos_fases(sigma_borde=2)
    return {"original": image, "processed": image, "mask": mask,
            "filepath": "sintetica.tif", "background": 1,
            "lineas_watershed": np.zeros(mask.shape, bool)}


def test_figura_composicional_se_guarda(tmp_path: Path) -> None:
    """La figura se guarda como '<nombre>_composicional.png'."""
    image, mask = _dos_fases(sigma_borde=2)

    out = _silencio(
        show_compositional_results, image, mask,
        _silencio(compute_intensity_profiles, image, mask),
        _silencio(compute_intensity_gradients, image, mask),
        _silencio(detect_inhomogeneities, image, mask),
        "muestra.dm3", output_dir=tmp_path, show=False,
    )

    assert out == tmp_path / "muestra_composicional.png"
    assert out.is_file()


def test_pipeline_compositional_agrega_claves(tmp_path: Path) -> None:
    """Devuelve la entrada más las métricas del Módulo B."""
    entrada = _fraction_dict()

    r = _silencio(pipeline_compositional, entrada, output_dir=tmp_path,
                  show=False)

    for clave in entrada:
        assert clave in r
    for clave in ("intensity_profiles", "interface_gradients",
                  "phase_maps", "inhomogeneities"):
        assert clave in r
    assert "fondo-fase_1" in r["interface_gradients"]
    assert (tmp_path / "sintetica_composicional.png").is_file()


def test_pipeline_compositional_valida_entrada() -> None:
    """Un diccionario incompleto lanza ValueError."""
    with pytest.raises(ValueError, match="falta"):
        pipeline_compositional({"mask": np.zeros((5, 5), int)})


# ====================================================================
# MÓDULO C — spatial.py
# ====================================================================

LADO = 1200  # lado de las imágenes sintéticas del Módulo C


def _patron(nombre: str, n: int = 100) -> np.ndarray:
    """Máscara de LADO × LADO con n círculos de radio 10 (fase 0).

    Args:
        nombre: 'grilla' (10 × 10, paso 120 px), 'azar', 'grupos' (5
            grupos), 'un grupo' o 'arriba' (más probable arriba).
        n: Cantidad de círculos (no se usa en 'grilla').

    Returns:
        Máscara int32: 0 en los círculos, 1 en el fondo.
    """
    rng = np.random.default_rng(3)
    if nombre == "grilla":
        centros = [np.array((60 + 120 * i, 60 + 120 * j), float)
                   for i in range(10) for j in range(10)]
    else:
        semillas = rng.uniform(150, 1050, (5, 2))
        centros = []
        while len(centros) < n:
            if nombre == "azar":
                c = rng.uniform(15, LADO - 15, 2)
            elif nombre == "grupos":
                c = semillas[rng.integers(5)] + rng.normal(0, 40, 2)
            elif nombre == "un grupo":
                c = np.array([600.0, 600.0]) + rng.normal(0, 70, 2)
            else:  # 'arriba': la probabilidad baja de arriba hacia abajo
                c = rng.uniform(15, LADO - 15, 2)
                if rng.random() > 1 - c[0] / LADO:
                    continue
            if (np.all((c > 12) & (c < LADO - 12))
                    and all(np.hypot(*(c - o)) > 22 for o in centros)):
                centros.append(c)
    return _circulos([tuple(c) for c in centros], radio=10, size=LADO)


def _espacial(nombre: str, n: int = 100) -> tuple[np.ndarray, dict]:
    """Máscara de un patrón y sus propiedades de la Etapa 3."""
    mask = _patron(nombre, n)
    return mask, _props(mask)


# --- C1: RDF ---------------------------------------------------------


def test_rdf_grilla_picos_en_las_distancias_de_la_grilla() -> None:
    """En una grilla de paso a, g(r) tiene picos en a y a·√2."""
    mask, props = _espacial("grilla")

    r = _silencio(compute_rdf, props, mask)["fase_1"]

    paso = r["r"][1] - r["r"][0]

    def g_en(dist: float) -> float:
        # Máximo en ±1 intervalo (dist puede caer en el borde de dos)
        cerca = np.abs(r["r"] - dist) <= paso
        return float(np.max(r["g"][cerca]))

    assert g_en(120) > 3       # paso de la grilla
    assert g_en(120 * np.sqrt(2)) > 3
    assert g_en(90) == 0       # entre picos no hay pares
    assert r["clasificacion"] == "exclusion"


@pytest.mark.parametrize(("patron", "clase"),
                         [("azar", "aleatorio"), ("grupos", "clustering")])
def test_rdf_clasifica_azar_y_grupos(patron: str, clase: str) -> None:
    """Al azar → 'aleatorio'; agrupadas → 'clustering'."""
    mask, props = _espacial(patron)

    r = _silencio(compute_rdf, props, mask)["fase_1"]

    assert r["clasificacion"] == clase
    assert r["diametro_tipico_px"] == pytest.approx(20, abs=1.5)


def test_rdf_pocas_particulas_insuficiente() -> None:
    """Con menos de 10 partículas no se calcula."""
    mask, props = _espacial("azar", n=5)

    r = _silencio(compute_rdf, props, mask)["fase_1"]

    assert r["clasificacion"] == "insuficiente"
    assert r["g"] is None


# --- C2: mapa de densidad --------------------------------------------


def test_densidad_azar_compatible_y_concentrada_heterogenea() -> None:
    """Al azar el CV no supera el del azar; arriba sí."""
    mask_a, props_a = _espacial("azar", n=150)
    mask_b, props_b = _espacial("arriba", n=150)

    a = _silencio(compute_density_map, props_a, mask_a)["fase_1"]
    b = _silencio(compute_density_map, props_b, mask_b)["fase_1"]

    assert a["clasificacion"] == "uniforme (compatible con azar)"
    assert not a["heterogeneidad_significativa"]
    assert b["heterogeneidad_significativa"]
    assert b["posicion_max_px"][0] < LADO / 2  # el máximo está arriba
    assert a["mapa_conteos"].sum() == 150


def test_densidad_grilla_fija_y_validacion() -> None:
    """grid_size entero se respeta; 0 da error."""
    mask, props = _espacial("azar")

    d = _silencio(compute_density_map, props, mask, grid_size=8)["fase_1"]

    assert d["grid_size"] == 8
    assert d["mapa_suavizado"].shape == (8, 8)
    with pytest.raises(ValueError):
        compute_density_map(props, mask, grid_size=0)


# --- C3: Hopkins y DBSCAN --------------------------------------------


def test_un_solo_grupo_hopkins_mayor_que_075() -> None:
    """Puntos en un solo grupo → H > 0.75 (test de la especificación)."""
    mask, props = _espacial("un grupo")

    c = _silencio(detect_clustering, props, mask)["fase_1"]

    assert c["hopkins_statistic"] > 0.75
    assert c["hopkins_clasificacion"] == "clustering"
    assert c["n_clusters_dbscan"] == 1


@pytest.mark.parametrize(("patron", "clase"),
                         [("grilla", "regular"), ("azar", "aleatorio")])
def test_hopkins_grilla_y_azar(patron: str, clase: str) -> None:
    """Grilla → 'regular'; al azar → 'aleatorio', sin grupos DBSCAN."""
    mask, props = _espacial(patron)

    c = _silencio(detect_clustering, props, mask)["fase_1"]

    assert c["hopkins_clasificacion"] == clase
    assert c["n_clusters_dbscan"] == 0
    assert c["n_particulas_aisladas"] == 100


def test_dbscan_encuentra_grupos() -> None:
    """Con 5 grupos, DBSCAN encuentra grupos y pocas aisladas."""
    mask, props = _espacial("grupos")

    c = _silencio(detect_clustering, props, mask)["fase_1"]

    assert c["n_clusters_dbscan"] >= 2
    assert c["n_particulas_aisladas"] < 10
    assert (c["n_particulas_en_clusters"] + c["n_particulas_aisladas"]
            == 100)
    for cl in c["clusters"]:
        assert set(cl) == {"id", "size", "centroide", "radio_px"}


def test_clustering_pocas_particulas() -> None:
    """Con menos de 10 partículas, 'insuficiente'."""
    mask, props = _espacial("azar", n=5)

    c = _silencio(detect_clustering, props, mask)["fase_1"]

    assert c["hopkins_clasificacion"] == "insuficiente"


# --- C4: gradientes --------------------------------------------------


def test_mas_particulas_arriba_gradiente_vertical() -> None:
    """Más partículas arriba → gradiente vertical (p < 0.05)."""
    mask, props = _espacial("arriba", n=150)
    image = np.where(mask == 0, 0.2, 0.8)

    g = _silencio(detect_spatial_gradients, props, mask, image)["fase_1"]

    vertical = g["densidad"]["vertical"]
    assert vertical["p_value"] < 0.05
    assert vertical["significativo"]
    assert vertical["pendiente_por_100px"] < 0  # disminuye hacia abajo
    assert any("densidad vertical: disminuye hacia abajo" in s
               for s in g["gradientes_significativos"])


def test_sin_gradiente_y_sin_falsos_por_redondeo() -> None:
    """Al azar y con intensidad constante no hay gradientes."""
    mask, props = _espacial("azar", n=150)
    image = np.where(mask == 0, 0.2, 0.8)  # constante dentro de la fase

    g = _silencio(detect_spatial_gradients, props, mask, image)["fase_1"]

    assert g["gradientes_significativos"] == []
    assert g["intensidad"]["vertical"]["p_value"] is None
    assert g["umbral_p"] == pytest.approx(0.05 / 6)


def test_gradiente_de_intensidad() -> None:
    """Una fase que se aclara hacia la derecha da gradiente."""
    mask, props = _espacial("azar")
    rampa = np.linspace(0.1, 0.4, LADO)[None, :] * np.ones((LADO, 1))
    image = np.where(mask == 0, rampa, 0.8)

    g = _silencio(detect_spatial_gradients, props, mask, image)["fase_1"]

    assert g["intensidad"]["horizontal"]["significativo"]
    assert g["intensidad"]["horizontal"]["pendiente_por_100px"] > 0


def test_gradientes_validan_shape() -> None:
    """image y mask con shapes distintos lanzan ValueError."""
    with pytest.raises(ValueError):
        detect_spatial_gradients({}, np.zeros((5, 5), int),
                                 np.zeros((4, 4)))


# --- C5: figura y pipeline -------------------------------------------


def test_figura_espacial_se_guarda(tmp_path: Path) -> None:
    """La figura se guarda como '<nombre>_espacial.png'."""
    mask, props = _espacial("grupos")
    image = np.where(mask == 0, 0.2, 0.8)

    out = _silencio(
        show_spatial_results, image, mask,
        _silencio(compute_rdf, props, mask),
        _silencio(compute_density_map, props, mask),
        _silencio(detect_clustering, props, mask),
        _silencio(detect_spatial_gradients, props, mask, image),
        "muestra.dm3", props=props, output_dir=tmp_path, show=False,
    )

    assert out == tmp_path / "muestra_espacial.png"
    assert out.is_file()


def test_pipeline_spatial_agrega_claves(tmp_path: Path) -> None:
    """Devuelve la entrada más las métricas del Módulo C."""
    mask, props = _espacial("grupos")
    image = np.where(mask == 0, 0.2, 0.8)
    entrada = {"original": image, "processed": image, "mask": mask,
               "props": props, "filepath": "sintetica.tif",
               "background": 1}

    r = _silencio(pipeline_spatial, entrada, output_dir=tmp_path,
                  show=False)

    for clave in entrada:
        assert clave in r
    for clave in ("rdf", "density_map", "clustering",
                  "spatial_gradients"):
        assert "fase_1" in r[clave]
    assert (tmp_path / "sintetica_espacial.png").is_file()


def test_pipeline_spatial_valida_entrada() -> None:
    """Un diccionario incompleto lanza ValueError."""
    with pytest.raises(ValueError, match="falta"):
        pipeline_spatial({"mask": np.zeros((5, 5), int)})
