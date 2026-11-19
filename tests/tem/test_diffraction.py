"""Tests de la Etapa 4.5: análisis de patrones de difracción.

Los patrones son sintéticos: halo que decae con el radio, anillos o
spots gaussianos, beam stopper opcional y ruido.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from tem import diffraction as dif

LADO = 512
CENTRO = (270.0, 245.0)


def _silencio(func, *args, **kwargs):
    """Llama a func sin imprimir nada."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _normalizar(img: np.ndarray) -> np.ndarray:
    """Lleva la imagen a [0, 1] (como load_tem_image)."""
    lo, hi = np.percentile(img, [0.1, 99.9])
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def _halo(lado: int = LADO, centro=CENTRO, seed: int = 0) -> np.ndarray:
    """Halo central que decae con el radio en todo el campo (como en los
    patrones reales), más ruido."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((lado, lado))
    r = np.hypot(y - centro[0], x - centro[1])
    img = 1.0 / (1 + r / 30.0)
    return img + rng.normal(0, 0.002, img.shape)


def _stopper(img: np.ndarray, centro=CENTRO) -> np.ndarray:
    """Agrega un beam stopper: franja oscura desde el borde derecho
    hasta el centro."""
    out = img.copy()
    y, x = np.indices(img.shape)
    out[(np.abs(y - centro[0]) < 12) & (x > centro[1] - 5)] = img.min()
    return out


def _anillos(radios=(60, 100, 150), stopper: bool = False) -> np.ndarray:
    """Patrón policristalino: halo más anillos gaussianos."""
    img = _halo()
    y, x = np.indices(img.shape)
    r = np.hypot(y - CENTRO[0], x - CENTRO[1])
    for k, radio in enumerate(radios):
        img += 0.02 / (k + 1) * np.exp(-0.5 * ((r - radio) / 1.5) ** 2)
    if stopper:
        img = _stopper(img)
    return _normalizar(img)


def _red(paso: float = 45.0, sigma: float = 1.2,
         alto: float = 0.03) -> np.ndarray:
    """Patrón de puntos: red hexagonal centrada, simétrica."""
    img = _halo()
    y, x = np.indices(img.shape)
    a = np.array([paso, 0.0])
    b = np.array([paso / 2, paso * np.sqrt(3) / 2])
    for i in range(-6, 7):
        for j in range(-6, 7):
            if i == j == 0:
                continue
            dy, dx = i * a + j * b
            cy, cx = CENTRO[0] + dy, CENTRO[1] + dx
            if 10 < cy < LADO - 10 and 10 < cx < LADO - 10:
                img += alto * np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2)
                                     / sigma**2)
    return _normalizar(img)


def _campo_claro(seed: int = 1) -> np.ndarray:
    """Imagen de campo claro: gradiente suave más partículas oscuras."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((LADO, LADO))
    img = 0.6 + 0.2 * x / LADO
    for _ in range(40):
        cy, cx = rng.uniform(20, LADO - 20, 2)
        img -= 0.3 * (np.hypot(y - cy, x - cx) < rng.uniform(5, 15))
    img += rng.normal(0, 0.02, img.shape)
    return _normalizar(img)


# --------------------------------------------------------------------
# is_diffraction_pattern
# --------------------------------------------------------------------


def test_anillos_es_difraccion():
    assert _silencio(dif.is_diffraction_pattern, _anillos()) is True


def test_anillos_con_stopper_es_difraccion():
    img = _anillos(stopper=True)
    assert _silencio(dif.is_diffraction_pattern, img) is True


def test_red_de_puntos_es_difraccion():
    assert _silencio(dif.is_diffraction_pattern, _red()) is True


def test_gradiente_suave_no_es_difraccion():
    y, x = np.indices((LADO, LADO))
    img = (0.3 + 0.5 * x / LADO).astype(np.float32)
    assert _silencio(dif.is_diffraction_pattern, img) is False


def test_campo_claro_no_es_difraccion():
    assert _silencio(dif.is_diffraction_pattern, _campo_claro()) is False


def test_particula_grande_no_es_difraccion():
    # Campo oscuro: una partícula brillante irregular sobre fondo oscuro
    rng = np.random.default_rng(2)
    y, x = np.indices((LADO, LADO))
    ang = np.arctan2(y - 256, x - 256)
    borde = 120 + 30 * np.sin(3 * ang) + 15 * np.cos(5 * ang)
    img = (np.hypot(y - 256, x - 256) < borde) * (
        0.7 + 0.2 * rng.random((LADO, LADO)))
    img = img + rng.normal(0, 0.02, img.shape)
    assert _silencio(dif.is_diffraction_pattern, _normalizar(img)) is False


def test_metadatos_en_modo_difraccion_deciden():
    img = _campo_claro()
    meta = {"modo_operacion": "DIFFRACTION"}
    assert _silencio(dif.is_diffraction_pattern, img, meta) is True


def test_perfil_1d_es_difraccion():
    perfil = np.sin(np.linspace(0, 20, 800)) ** 2
    img = np.tile(perfil, (20, 1)).astype(np.float32)
    assert _silencio(dif.is_diffraction_pattern, img) is True


def test_imprime_los_tres_votos(capsys):
    dif.is_diffraction_pattern(_anillos())
    salida = capsys.readouterr().out
    for texto in ("Simetría", "Decaimiento", "Uniformidad"):
        assert texto in salida


# --------------------------------------------------------------------
# detect_pattern_type
# --------------------------------------------------------------------


def test_tipo_anillos():
    assert _silencio(dif.detect_pattern_type, _anillos()) == "anillos"


def test_tipo_anillos_con_stopper():
    img = _anillos(stopper=True)
    assert _silencio(dif.detect_pattern_type, img) == "anillos"


def test_tipo_puntos():
    assert _silencio(dif.detect_pattern_type, _red()) == "puntos"


def test_tipo_nbed_spots_anchos():
    img = _red(paso=70, sigma=5, alto=0.05)
    assert _silencio(dif.detect_pattern_type, img) == "nbed"


def test_tipo_amorfo():
    # Halo con un anillo muy ancho y difuso, sin spots
    img = _halo()
    y, x = np.indices(img.shape)
    r = np.hypot(y - CENTRO[0], x - CENTRO[1])
    img += 0.003 * np.exp(-0.5 * ((r - 90) / 25) ** 2)
    assert _silencio(dif.detect_pattern_type, _normalizar(img)) == "amorfo"


def test_tipo_perfil_1d_por_aspecto():
    perfil = np.sin(np.linspace(0, 20, 900)) ** 2
    img = np.tile(perfil, (30, 1)).astype(np.float32)
    assert _silencio(dif.detect_pattern_type, img) == "perfil_1d"


def test_tipo_perfil_1d_vector():
    perfil = np.sin(np.linspace(0, 20, 900)) ** 2
    assert _silencio(dif.detect_pattern_type, perfil) == "perfil_1d"


def test_tipo_encuentra_el_centro():
    info = dif._pattern_features(_anillos(stopper=True), None)
    assert np.hypot(info["centro"][0] - CENTRO[0],
                    info["centro"][1] - CENTRO[1]) < 2


def test_tipo_radios_de_los_anillos():
    info = dif._pattern_features(_anillos(), CENTRO)
    encontrados = [r for r, c in zip(info["radios_anillos"],
                                     info["coberturas"]) if c >= 0.5]
    assert len(encontrados) == 3
    assert np.allclose(sorted(encontrados), [60, 100, 150], atol=2)


# --------------------------------------------------------------------
# read_camera_metadata y longitud de onda
# --------------------------------------------------------------------


@pytest.mark.parametrize("kv, esperado", [
    (100, 0.03701), (200, 0.02508), (300, 0.01969),
])
def test_longitud_de_onda(kv, esperado):
    assert dif.electron_wavelength(kv) == pytest.approx(esperado, abs=1e-5)


def test_metadatos_por_defecto(tmp_path):
    from skimage import io as skio

    archivo = tmp_path / "patron.png"
    skio.imsave(archivo, (_anillos() * 255).astype(np.uint8))
    meta = _silencio(dif.read_camera_metadata, archivo, interactive=False)
    lam = dif.electron_wavelength(dif.DEFAULT_VOLTAGE_KV)
    assert meta["fuente"] == "por_defecto"
    assert meta["metodo_calibracion"] == "lambda_L"
    assert meta["lambda_angs"] == pytest.approx(lam)
    assert meta["lambda_L"] == pytest.approx(
        lam * dif.DEFAULT_CAMERA_LENGTH_MM)
    assert meta["calibracion_inv_angs_px"] == pytest.approx(
        dif.DEFAULT_DETECTOR_PIXEL_UM * 1e-3 / meta["lambda_L"])


def test_metadatos_pide_al_usuario(tmp_path, monkeypatch):
    archivo = tmp_path / "patron.png"
    archivo.write_bytes(b"")
    respuestas = iter(["300", "500", ""])  # kV, L (mm), píxel (Enter)
    monkeypatch.setattr("builtins.input", lambda _: next(respuestas))
    meta = _silencio(dif.read_camera_metadata, archivo, interactive=True)
    assert meta["voltage_kV"] == 300
    assert meta["camera_length_mm"] == 500
    assert meta["pixel_detector_um"] == dif.DEFAULT_DETECTOR_PIXEL_UM
    assert meta["fuentes"]["voltage_kV"] == "usuario"
    assert meta["fuente"] == "usuario"


def test_buscar_tag_respeta_prioridad_y_exclusiones():
    tags = {
        ".ImageList.2.ImageTags.Microscope Info.STEM Camera Length": "700",
        ".ImageList.2.ImageTags.Microscope Info.Magnification": "1",
        ".ImageList.2.ImageTags.Microscope Info.Actual Magnification": "137",
        ".ImageList.1.ImageTags.Microscope Info.Actual Magnification": "5",
    }
    assert dif._find_tag(tags, ("Camera Length",), exclude=("STEM",)) is None
    valor = dif._find_tag(tags, ("Actual Magnification", "Magnification"))
    assert valor == "137"  # variante prioritaria y última ImageList


def test_numero_de_texto():
    assert dif._number("200kV") == 200
    assert dif._number("1.5e3 V") == 1500
    assert dif._number("sin número") is None


# --------------------------------------------------------------------
# Bloque 2: centro del haz, beam stopper y preprocesamiento
# --------------------------------------------------------------------


def _haz_visible(centro=(300.3, 211.7)) -> np.ndarray:
    """Halo con el haz directo visible (gaussiana) en posición conocida."""
    img = _halo(centro=centro)
    y, x = np.indices(img.shape)
    img += 0.5 * np.exp(-0.5 * ((y - centro[0]) ** 2 + (x - centro[1]) ** 2)
                        / 4.0**2)
    return _normalizar(img)


def _distancia(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_conversion_entre_escalas_es_inversa():
    punto = (123.4, 56.7)
    for factor in (1, 2, 4):
        vuelta = dif._to_full(dif._to_small(punto, factor), factor)
        assert np.allclose(vuelta, punto)
    # El píxel 0 de una copia reducida x2 promedia los píxeles 0 y 1
    assert dif._to_full((0, 0), 2) == (0.5, 0.5)


def test_centro_gaussiana_en_posicion_conocida():
    centro = (300.3, 211.7)
    encontrado = _silencio(dif.find_beam_center, _haz_visible(centro))
    assert _distancia(encontrado, centro) < 2


def test_centro_con_beam_stopper():
    encontrado = _silencio(dif.find_beam_center, _anillos(stopper=True))
    assert _distancia(encontrado, CENTRO) < 2


def test_centro_red_de_puntos():
    encontrado = _silencio(dif.find_beam_center, _red())
    assert _distancia(encontrado, CENTRO) < 2


def test_centro_imagen_reducida():
    # Imagen de 1200 px: se trabaja sobre copias reducidas y el centro
    # tiene que volver bien a la escala completa
    centro = (640.0, 571.0)
    img = _halo(lado=1200, centro=centro)
    y, x = np.indices(img.shape)
    r = np.hypot(y - centro[0], x - centro[1])
    for radio in (150, 260, 380):
        img += 0.02 * np.exp(-0.5 * ((r - radio) / 2.0) ** 2)
    encontrado = _silencio(dif.find_beam_center, _normalizar(img))
    assert _distancia(encontrado, centro) < 2


def test_centro_imprime_metodos(capsys):
    dif.find_beam_center(_anillos(stopper=True))
    salida = capsys.readouterr().out
    for texto in ("Simetría del halo", "Nitidez de anillos",
                  "Centro de masa", "Gaussiana 2D"):
        assert texto in salida


def test_centro_de_perfil_1d_da_error():
    perfil = np.sin(np.linspace(0, 20, 600)) ** 2
    with pytest.raises(ValueError):
        dif.find_beam_center(np.tile(perfil, (10, 1)))


def test_stopper_detectado():
    img = _anillos(stopper=True)
    mascara = dif.detect_beam_stopper(img, CENTRO)
    y, x = np.indices(img.shape)
    real = (np.abs(y - CENTRO[0]) < 12) & (x > CENTRO[1] - 5)
    assert mascara.shape == img.shape
    assert mascara[real].mean() > 0.95
    # Fuera del stopper (con 6 px de margen) casi no marca nada
    lejos = ~((np.abs(y - CENTRO[0]) < 18) & (x > CENTRO[1] - 25))
    assert mascara[lejos].mean() < 0.01


def test_sin_stopper_no_marca_nada():
    assert not dif.detect_beam_stopper(_red(), CENTRO).any()


def test_preprocesamiento_enmascara_haz_y_stopper():
    img = _anillos(stopper=True)
    prep = _silencio(dif.preprocess_diffraction, img, CENTRO)
    assert prep.dtype == np.float32 and prep.shape == img.shape
    # Haz directo: disco de 2 % del lado
    assert np.isnan(prep[int(CENTRO[0]), int(CENTRO[1])])
    # Stopper (lejos del centro, hacia el borde derecho)
    assert np.isnan(prep[int(CENTRO[0]), LADO - 5])
    # El resto queda con valores finitos (esquina y arriba del centro)
    assert np.isfinite(prep[50, 50])
    assert np.isfinite(prep[int(CENTRO[0]) - 100, int(CENTRO[1])])


def test_preprocesamiento_es_logaritmico():
    img = _red()
    prep = _silencio(dif.preprocess_diffraction, img, CENTRO)
    ok = np.isfinite(prep)
    # Monótono: más intensidad, mayor valor
    orden = np.argsort(img[ok])
    assert np.all(np.diff(np.sort(prep[ok])) >= 0)
    assert np.corrcoef(img[ok][orden], prep[ok][orden])[0, 1] > 0.8
    # Comprime: el rango relativo baja respecto de la imagen lineal
    assert prep[ok].max() / np.median(prep[ok]) < img[ok].max() / np.median(
        img[ok])


def test_correccion_interactiva_sin_ventana_devuelve_el_centro():
    import matplotlib

    matplotlib.use("Agg")
    centro = (270.0, 245.0)
    assert dif.correct_center_interactive(_anillos(), centro) == centro


def test_selector_de_centro_click_manual():
    import matplotlib
    from types import SimpleNamespace

    matplotlib.use("Agg")
    img = _halo(lado=1200)  # se muestra reducida x2
    picker = dif._CenterPicker(img, (300.0, 300.0), guide_radii=[100])
    # Antes de "Corregir manualmente", los clicks no hacen nada
    picker.on_click(SimpleNamespace(inaxes=picker.ax, xdata=50.0,
                                    ydata=60.0))
    assert picker.center == (300.0, 300.0)
    picker.start_manual()
    assert picker.ax_confirm.get_visible()
    picker.on_click(SimpleNamespace(inaxes=picker.ax, xdata=50.0,
                                    ydata=60.0))
    # Coordenadas de la copia reducida -> imagen completa
    assert picker.center == pytest.approx(dif._to_full((60.0, 50.0), 2))
    assert picker.circles[0].center == (50.0, 60.0)
    # Un click fuera de la imagen (por ejemplo, en un botón) no mueve
    picker.on_click(SimpleNamespace(inaxes=None, xdata=1.0, ydata=1.0))
    assert picker.center == pytest.approx(dif._to_full((60.0, 50.0), 2))
    picker.accept()


# --------------------------------------------------------------------
# Bloque 3: perfil radial, anillos y Scherrer
# --------------------------------------------------------------------

# Calibración de los tests: 0.002 1/Å por píxel, 200 kV
META = {"calibracion_inv_angs_px": 0.002,
        "lambda_angs": dif.electron_wavelength(200),
        "pixel_detector_um": 14.0}


def _anillos_gauss(radios=(80, 130, 190), sigma=3.0, lado=LADO,
                   centro=CENTRO, seed=0) -> np.ndarray:
    """Halo más anillos gaussianos de ancho conocido (imagen lineal)."""
    img = _halo(lado=lado, centro=centro, seed=seed)
    y, x = np.indices(img.shape)
    r = np.hypot(y - centro[0], x - centro[1])
    for radio in radios:
        img += 0.03 * np.exp(-0.5 * ((r - radio) / sigma) ** 2)
    return img


def test_perfil_radial_ejes():
    perfil = dif.compute_radial_profile(_anillos_gauss(), CENTRO, META)
    r_max = int(min(CENTRO[0], CENTRO[1], LADO - 1 - CENTRO[0],
                    LADO - 1 - CENTRO[1]))
    assert perfil["r_px"][0] == 0 and perfil["r_px"][-1] == r_max
    d = perfil["d_angs"]
    assert np.isinf(d[0])
    assert np.all(np.diff(d[1:]) < 0)  # monótonamente decreciente
    assert np.allclose(perfil["q_inv_angs"][1:], 1 / d[1:])
    # d = 1/(r·cal) = λL/R con R = r · píxel del detector
    assert d[100] == pytest.approx(1 / (100 * 0.002))
    assert perfil["r_mm"][100] == pytest.approx(100 * 14e-3)


def test_perfil_radial_promedia_cada_radio():
    y, x = np.indices((LADO, LADO))
    r = np.hypot(y - CENTRO[0], x - CENTRO[1])
    img = 1.0 / (1 + r)  # depende solo del radio
    perfil = dif.compute_radial_profile(img, CENTRO, META)
    esperado = 1.0 / (1 + perfil["r_px"])
    assert np.allclose(perfil["intensity"][5:], esperado[5:], rtol=0.03)


def test_perfil_radial_ignora_nan():
    img = _anillos_gauss()
    con_nan = img.copy()
    y, x = np.indices(img.shape)
    con_nan[(np.abs(y - CENTRO[0]) < 12) & (x > CENTRO[1])] = np.nan
    a = dif.compute_radial_profile(img, CENTRO, META)
    b = dif.compute_radial_profile(con_nan, CENTRO, META)
    assert np.all(b["n_pixeles"][20:] < a["n_pixeles"][20:])
    # El perfil (simétrico) no cambia al sacar un sector
    assert np.allclose(a["intensity"][20:], b["intensity"][20:], rtol=0.02)


def test_anillos_tres_picos_conocidos():
    perfil = dif.compute_radial_profile(_anillos_gauss(), CENTRO, META)
    anillos = _silencio(dif.detect_diffraction_rings, perfil)
    assert anillos["n_anillos"] == 3
    radios = [a["r_px"] for a in anillos["anillos"]]
    assert np.allclose(radios, [80, 130, 190], atol=1)
    for a in anillos["anillos"]:
        assert a["d_angs"] == pytest.approx(1 / (a["r_px"] * 0.002))


def test_anillos_fwhm():
    sigma = 3.0
    perfil = dif.compute_radial_profile(_anillos_gauss(sigma=sigma),
                                        CENTRO, META)
    anillos = _silencio(dif.detect_diffraction_rings, perfil)
    # FWHM de la gaussiana (2.355 sigma) ensanchado por el suavizado del
    # perfil (sigma 1.5 px): sqrt(3² + 1.5²) · 2.355 = 7.9 px
    esperado = 2.355 * np.hypot(sigma, dif.PROFILE_SMOOTH)
    for a in anillos["anillos"]:
        assert a["fwhm_px"] == pytest.approx(esperado, rel=0.15)
        assert a["fwhm_d_angs"] == pytest.approx(
            a["d_angs"] * a["fwhm_px"] / a["r_px"])


def test_anillos_ratios_y_serie_cubica():
    # Anillos FCC: d ∝ 1/sqrt(h²+k²+l²) con 3, 4, 8 -> r ∝ sqrt(3, 4, 8)
    # (111 y 200 quedan a 15 px, unos dos FWHM)
    radios = 100 * np.sqrt(np.array([3, 4, 8]) / 3)
    perfil = dif.compute_radial_profile(_anillos_gauss(radios=radios),
                                        CENTRO, META)
    anillos = _silencio(dif.detect_diffraction_rings, perfil)
    q2 = [a["q2_relativo"] for a in anillos["anillos"]]
    assert np.allclose(q2, [1, 4 / 3, 8 / 3], atol=0.03)
    ratios = anillos["ratios_d"]
    assert ratios.shape == (3, 3)
    assert np.allclose(np.diag(ratios), 1)
    assert np.allclose(ratios * ratios.T, 1)


def test_anillos_sin_anillos():
    perfil = dif.compute_radial_profile(_halo(), CENTRO, META)
    anillos = _silencio(dif.detect_diffraction_rings, perfil)
    assert anillos["n_anillos"] == 0
    assert anillos["ratios_d"].shape == (0, 0)


def _anillo_manual(fwhm_px: float, d: float = 2.0,
                   cal: float = 0.002) -> dict:
    """Diccionario mínimo de detect_diffraction_rings con un anillo."""
    return {
        "anillos": [{"id": 1, "d_angs": d, "q_inv_angs": 1 / d,
                     "fwhm_px": fwhm_px, "intensidad": 1.0}],
        "calibracion_inv_angs_px": cal,
        "lambda_angs": dif.electron_wavelength(200),
    }


def test_scherrer_valor_conocido():
    # Δq = 10 px · 0.002 = 0.02 1/Å -> D ≈ K/Δq = 0.94/0.02 Å = 4.7 nm
    res = _silencio(dif.estimate_crystallite_size, _anillo_manual(10.0))
    assert res["por_anillo"][0]["D_nm"] == pytest.approx(4.7, rel=0.01)
    assert res["D_medio_nm"] == pytest.approx(4.7, rel=0.01)
    assert res["es_minimo"] is True


@pytest.mark.parametrize("fwhm", [1.0, 3.0, 10.0, 30.0])
def test_scherrer_rango_razonable(fwhm):
    res = _silencio(dif.estimate_crystallite_size, _anillo_manual(fwhm))
    assert 1 <= res["por_anillo"][0]["D_nm"] <= 200


def test_scherrer_resta_el_ancho_instrumental():
    sin = _silencio(dif.estimate_crystallite_size, _anillo_manual(10.0))
    con = _silencio(dif.estimate_crystallite_size, _anillo_manual(10.0),
                    instrumental_fwhm_px=6.0)
    # sqrt(10² - 6²) = 8 px -> D más grande
    assert con["por_anillo"][0]["fwhm_corregido_px"] == pytest.approx(8.0)
    assert con["por_anillo"][0]["D_nm"] > sin["por_anillo"][0]["D_nm"]
    assert con["es_minimo"] is False


def test_scherrer_anillo_tan_angosto_como_el_instrumento():
    res = _silencio(dif.estimate_crystallite_size, _anillo_manual(5.0),
                    instrumental_fwhm_px=5.0)
    assert res["por_anillo"][0]["D_nm"] is None
    assert res["D_medio_nm"] is None
    assert res["advertencias"]


def test_scherrer_avisa_fuera_de_rango():
    # Anillo muy ancho -> D < 2 nm
    res = _silencio(dif.estimate_crystallite_size, _anillo_manual(40.0))
    assert res["por_anillo"][0]["fuera_de_rango"] is True
    assert any("fuera del rango" in a for a in res["advertencias"])


# --------------------------------------------------------------------
# Bloque 4: spots y geometría de la red
# --------------------------------------------------------------------


def _con_spots(puntos, sigma: float = 1.5, alto: float = 0.05,
               seed: int = 0) -> np.ndarray:
    """Halo con spots gaussianos en las posiciones (y, x) dadas."""
    img = _halo(seed=seed)
    y, x = np.indices(img.shape)
    for py, px in puntos:
        img += alto * np.exp(-0.5 * ((y - py) ** 2 + (x - px) ** 2)
                             / sigma**2)
    return img


def _seis_simetricos(radio: float = 100.0) -> list[tuple[float, float]]:
    """6 spots a 60° (hexágono) alrededor de CENTRO."""
    return [(CENTRO[0] + radio * np.sin(np.radians(a)),
             CENTRO[1] + radio * np.cos(np.radians(a)))
            for a in range(0, 360, 60)]


def _puntos_red(a_vec, b_vec, n: int = 4) -> list[tuple[float, float]]:
    """Puntos h·a + k·b (|h|, |k| <= n, sin el origen) dentro de la
    imagen."""
    pts = []
    for h in range(-n, n + 1):
        for k in range(-n, n + 1):
            if h == k == 0:
                continue
            dy, dx = h * np.asarray(a_vec) + k * np.asarray(b_vec)
            py, px = CENTRO[0] + dy, CENTRO[1] + dx
            if 15 < py < LADO - 15 and 15 < px < LADO - 15:
                pts.append((py, px))
    return pts


def test_spots_seis_simetricos():
    puntos = _seis_simetricos()
    res = _silencio(dif.detect_diffraction_spots, _con_spots(puntos),
                    CENTRO, META)
    assert res["n_spots"] == 6
    assert res["indice_simetria"] > 0.9
    for py, px in puntos:
        d = min(np.hypot(s["y_px"] - py, s["x_px"] - px)
                for s in res["spots"])
        assert d < 1


def test_spots_datos_de_cada_spot():
    res = _silencio(dif.detect_diffraction_spots,
                    _con_spots(_seis_simetricos(120)), CENTRO, META)
    for s in res["spots"]:
        assert s["r_px"] == pytest.approx(120, abs=1)
        assert 0 <= s["phi_deg"] < 360
        assert s["d_angs"] == pytest.approx(1 / (s["r_px"] * 0.002))
        assert s["tiene_par_simetrico"]
    angulos = sorted(s["phi_deg"] for s in res["spots"])
    assert np.allclose(np.diff(angulos), 60, atol=1)


def test_spots_excluye_el_haz_directo():
    puntos = _seis_simetricos() + [CENTRO]
    res = _silencio(dif.detect_diffraction_spots, _con_spots(puntos),
                    CENTRO, META)
    assert res["n_spots"] == 6
    assert min(s["r_px"] for s in res["spots"]) > 50


def test_spots_sin_metadatos_d_es_none():
    res = _silencio(dif.detect_diffraction_spots,
                    _con_spots(_seis_simetricos()), CENTRO)
    assert all(s["d_angs"] is None for s in res["spots"])


def test_spots_asimetricos_avisa(capsys):
    # 6 spots sin su opuesto: un solo lado del patrón
    puntos = [(CENTRO[0] + 100 * np.sin(np.radians(a)),
               CENTRO[1] + 100 * np.cos(np.radians(a)))
              for a in (10, 35, 70, 100, 130, 160)]
    res = dif.detect_diffraction_spots(_con_spots(puntos), CENTRO, META)
    assert res["indice_simetria"] < 0.7
    assert "Aviso" in capsys.readouterr().out


def test_spots_no_cuenta_ruido_como_spots():
    res = _silencio(dif.detect_diffraction_spots, _halo(), CENTRO, META)
    assert res["n_spots"] <= 2


def test_spots_par_fuera_de_la_imagen_no_cuenta():
    # El opuesto de este spot (y = 270 + 250 = 520) cae fuera de la
    # imagen: no es verificable
    puntos = _seis_simetricos() + [(CENTRO[0] - 250, CENTRO[1])]
    res = _silencio(dif.detect_diffraction_spots, _con_spots(puntos),
                    CENTRO, META)
    assert res["n_verificables"] == 6
    assert res["indice_simetria"] == 1.0


def test_red_hexagonal():
    a = (0.0, 45.0)
    b = (45 * np.sin(np.radians(60)), 45 * np.cos(np.radians(60)))
    spots = _silencio(dif.detect_diffraction_spots,
                      _con_spots(_puntos_red(a, b)), CENTRO, META)
    geo = _silencio(dif.analyze_spot_geometry, spots, CENTRO, META)
    red = geo["red"]
    assert red["a_estrella_px"] == pytest.approx(45, abs=0.5)
    assert red["b_estrella_px"] == pytest.approx(45, abs=0.5)
    assert red["gamma_estrella_deg"] == pytest.approx(60, abs=1)
    assert red["gamma_deg"] == pytest.approx(120, abs=1)
    # a = 1/(a*·sen γ*) con a* = 45 px · 0.002 1/Å
    assert red["a_angs"] == pytest.approx(
        1 / (45 * 0.002 * np.sin(np.radians(60))), rel=0.02)
    assert geo["fraccion_indexada"] > 0.95
    angulos = [ang for ang, _ in geo["angulos_caracteristicos"][:2]]
    assert any(abs(ang - 60) < 3 for ang in angulos)
    assert any(abs(ang - 120) < 3 for ang in angulos)
    primera = geo["familias"][0]
    assert primera["separacion_angular_deg"] == pytest.approx(60, abs=1)


def test_red_cuadrada():
    spots = _silencio(dif.detect_diffraction_spots,
                      _con_spots(_puntos_red((0, 50), (50, 0))),
                      CENTRO, META)
    geo = _silencio(dif.analyze_spot_geometry, spots, CENTRO, META)
    red = geo["red"]
    assert red["gamma_estrella_deg"] == pytest.approx(90, abs=1)
    # Con γ* = 90° la fórmula general coincide con a = 1/a*
    assert red["a_angs"] == pytest.approx(1 / (50 * 0.002), rel=0.02)
    assert geo["fraccion_indexada"] > 0.95


def test_red_rectangular():
    spots = _silencio(dif.detect_diffraction_spots,
                      _con_spots(_puntos_red((0, 40), (65, 0))),
                      CENTRO, META)
    red = _silencio(dif.analyze_spot_geometry, spots, CENTRO, META)["red"]
    assert red["a_estrella_px"] == pytest.approx(40, abs=0.5)
    assert red["b_estrella_px"] == pytest.approx(65, abs=0.5)
    assert red["gamma_estrella_deg"] == pytest.approx(90, abs=1)


def test_spots_de_varios_cristales_indexan_poco():
    rng = np.random.default_rng(3)
    radios = rng.uniform(60, 200, 40)
    angulos = rng.uniform(0, 2 * np.pi, 40)
    puntos = [(CENTRO[0] + r * np.sin(t), CENTRO[1] + r * np.cos(t))
              for r, t in zip(radios, angulos)]
    spots = _silencio(dif.detect_diffraction_spots, _con_spots(puntos),
                      CENTRO, META)
    geo = _silencio(dif.analyze_spot_geometry, spots, CENTRO, META)
    assert geo["fraccion_indexada"] < 0.5
    assert spots["indice_simetria"] < 0.7


def test_reduccion_de_base():
    a = np.array([0.0, 45.0])
    b = np.array([39.0, 22.5])
    # Base no reducida de la misma red: (a, b + 3a)
    ra, rb = dif._reduce_basis(a, b + 3 * a)
    assert np.linalg.norm(ra) == pytest.approx(45, abs=0.1)
    assert np.linalg.norm(rb) == pytest.approx(np.linalg.norm(b), abs=0.1)
    cos = ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))
    assert 0 <= cos <= 0.5 + 1e-9  # ángulo entre 60° y 90°


# --------------------------------------------------------------------
# Bloque 5: perfil 1D
# --------------------------------------------------------------------


def _perfil_1d(posiciones=(80, 130, 190), largo: int = 400,
               sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """Perfil que empieza en el haz directo: halo + picos gaussianos."""
    rng = np.random.default_rng(seed)
    r = np.arange(largo, dtype=float)
    perfil = 1.0 / (1 + r / 30.0)
    for p in posiciones:
        perfil += 0.03 * np.exp(-0.5 * ((r - p) / sigma) ** 2)
    return perfil + rng.normal(0, 0.0005, largo)


def _franja(perfil: np.ndarray, filas: int = 15, seed: int = 1):
    """Imagen angosta: el perfil repetido en filas, con ruido."""
    rng = np.random.default_rng(seed)
    return np.tile(perfil, (filas, 1)) + rng.normal(0, 0.002,
                                                    (filas, len(perfil)))


def _radios(res: dict) -> list[float]:
    return [a["r_px"] for a in res["picos"]["anillos"]]


def test_perfil_1d_vector():
    res = _silencio(dif.analyze_1d_profile, _perfil_1d(), META)
    assert res["eje"] == "vector"
    assert res["origen_px"] == 0
    assert np.allclose(_radios(res), [80, 130, 190], atol=1)
    for a in res["picos"]["anillos"]:
        assert a["d_angs"] == pytest.approx(1 / (a["r_px"] * 0.002))


def test_perfil_1d_franja_horizontal_y_vertical():
    franja = _franja(_perfil_1d())
    h = _silencio(dif.analyze_1d_profile, franja, META)
    v = _silencio(dif.analyze_1d_profile, franja.T, META)
    assert h["eje"] == "horizontal" and v["eje"] == "vertical"
    assert np.allclose(_radios(h), [80, 130, 190], atol=1)
    assert np.allclose(_radios(v), _radios(h), atol=0.1)


def test_perfil_1d_invertido():
    # El haz directo al final: el origen es el último punto
    perfil = _perfil_1d()[::-1]
    res = _silencio(dif.analyze_1d_profile, perfil, META)
    assert res["origen_px"] == len(perfil) - 1
    assert np.allclose(_radios(res), [80, 130, 190], atol=1)


def test_perfil_1d_dos_lados():
    # Corte que atraviesa el centro: el haz queda en el medio (px 399)
    medio = _perfil_1d()
    perfil = np.concatenate([medio[::-1], medio[1:]])
    res = _silencio(dif.analyze_1d_profile, perfil, META)
    assert res["origen_px"] == pytest.approx(399, abs=1)
    assert "dos lados" in res["criterio_origen"]
    assert np.allclose(_radios(res), [80, 130, 190], atol=1)


def test_perfil_1d_origen_indicado():
    medio = _perfil_1d()
    perfil = np.concatenate([np.full(50, medio[0]), medio])
    res = _silencio(dif.analyze_1d_profile, perfil, META, origin=50)
    assert res["criterio_origen"] == "indicado por el usuario"
    assert np.allclose(_radios(res), [80, 130, 190], atol=1)


def test_perfil_1d_fwhm_y_scherrer():
    res = _silencio(dif.analyze_1d_profile, _perfil_1d(sigma=3.0), META)
    esperado = 2.355 * np.hypot(3.0, dif.PROFILE_SMOOTH)
    for a in res["picos"]["anillos"]:
        assert a["fwhm_px"] == pytest.approx(esperado, rel=0.15)
    for fila in res["cristalito"]["por_anillo"]:
        assert 1 <= fila["D_nm"] <= 200


def test_perfil_1d_savgol():
    perfil = _perfil_1d()
    res = _silencio(dif.analyze_1d_profile, perfil, META)
    suave = res["intensity_savgol"]
    assert suave.shape == res["perfil"]["intensity"].shape
    # Suaviza: menos ruido punto a punto que el perfil original
    assert np.std(np.diff(suave, 2)) < np.std(np.diff(perfil, 2))


def test_perfil_1d_muy_corto_da_error():
    with pytest.raises(ValueError):
        dif.analyze_1d_profile(np.ones(10), META)


# --------------------------------------------------------------------
# Bloque 6: figura y pipeline
# --------------------------------------------------------------------


@pytest.fixture
def sin_ventanas():
    """Backend sin pantalla para las figuras."""
    import matplotlib

    matplotlib.use("Agg")


def _pipeline(tmp_path, imagen, **kwargs) -> dict:
    """Corre pipeline_diffraction sin ventanas ni preguntas.

    Usa un archivo .png vacío como ruta (sin metadatos: se usan los
    valores por defecto).
    """
    archivo = tmp_path / "patron.png"
    archivo.write_bytes(b"")
    return _silencio(dif.pipeline_diffraction, archivo, image=imagen,
                     interactive=False, show=False, output_dir=tmp_path,
                     **kwargs)


def test_pipeline_anillos(tmp_path, sin_ventanas):
    res = _pipeline(tmp_path, _normalizar(_anillos_gauss()))
    assert res["es_difraccion"] is True
    assert res["pattern_type"] == "anillos"
    assert _distancia(res["center"], CENTRO) < 2
    assert res["results"]["anillos"]["n_anillos"] == 3
    assert res["results"]["cristalito"]["D_medio_nm"] is not None
    assert res["radial_profile"] is not None
    assert res["image_preprocessed"].dtype == np.float32
    assert (tmp_path / "patron_difraccion.png").is_file()


def test_pipeline_puntos(tmp_path, sin_ventanas):
    res = _pipeline(tmp_path, _red())
    assert res["pattern_type"] == "puntos"
    assert res["results"]["spots"]["indice_simetria"] > 0.9
    red = res["results"]["geometria"]["red"]
    assert red["gamma_estrella_deg"] == pytest.approx(60, abs=2)


def test_pipeline_no_es_difraccion(tmp_path, sin_ventanas):
    res = _pipeline(tmp_path, _campo_claro())
    assert res == {"es_difraccion": False,
                   "filepath": str(tmp_path / "patron.png")}
    assert not (tmp_path / "patron_difraccion.png").exists()


def test_pipeline_forzado(tmp_path, sin_ventanas):
    # force=True saltea la detección: un patrón se analiza igual
    res = _pipeline(tmp_path, _red(), force=True)
    assert res["es_difraccion"] is True
    # Forzar una imagen sin patrón: el "centro" cae contra el borde y el
    # error lo explica (en lugar de un resultado sin sentido)
    with pytest.raises(ValueError, match="borde"):
        _pipeline(tmp_path, _campo_claro(), force=True)


def test_pipeline_perfil_1d(tmp_path, sin_ventanas):
    res = _pipeline(tmp_path, _franja(_perfil_1d()))
    assert res["pattern_type"] == "perfil_1d"
    assert res["center"] is None and res["radial_profile"] is None
    picos = res["results"]["perfil_1d"]["picos"]["anillos"]
    assert np.allclose([p["r_px"] for p in picos], [80, 130, 190], atol=1)
    assert (tmp_path / "patron_difraccion.png").is_file()


def test_pipeline_amorfo(tmp_path, sin_ventanas):
    img = _halo()
    y, x = np.indices(img.shape)
    r = np.hypot(y - CENTRO[0], x - CENTRO[1])
    img = img + 0.003 * np.exp(-0.5 * ((r - 90) / 25) ** 2)
    res = _pipeline(tmp_path, _normalizar(img))
    assert res["pattern_type"] == "amorfo"
    assert res["results"]["cristalito"] is None
    assert "anillos" in res["results"]


@pytest.mark.parametrize("tipo", ["anillos", "puntos", "perfil_1d"])
def test_figura_se_guarda(tmp_path, sin_ventanas, tipo):
    imagen = {"anillos": _normalizar(_anillos_gauss()), "puntos": _red(),
              "perfil_1d": _franja(_perfil_1d())}[tipo]
    res = _pipeline(tmp_path, imagen)
    salida = _silencio(dif.show_diffraction_results, imagen, res["center"],
                       res["pattern_type"], res["results"],
                       res["radial_profile"], res["metadata"],
                       tmp_path / f"{tipo}.png", output_dir=tmp_path,
                       show=False)
    assert salida == tmp_path / f"{tipo}_difraccion.png"
    assert salida.stat().st_size > 10_000


def test_figura_tipo_desconocido(tmp_path, sin_ventanas):
    with pytest.raises(ValueError):
        dif.show_diffraction_results(_red(), CENTRO, "otro", {}, None,
                                     META | {"fuente": "x"},
                                     tmp_path / "a.png",
                                     output_dir=tmp_path, show=False)


def test_exportado_desde_el_paquete():
    import tem

    for nombre in ("is_diffraction_pattern", "detect_pattern_type",
                   "read_camera_metadata", "find_beam_center",
                   "correct_center_interactive", "preprocess_diffraction",
                   "compute_radial_profile", "detect_diffraction_rings",
                   "estimate_crystallite_size", "detect_diffraction_spots",
                   "analyze_spot_geometry", "analyze_1d_profile",
                   "show_diffraction_results", "pipeline_diffraction"):
        assert nombre in tem.__all__ and hasattr(tem, nombre)
