"""Análisis morfológico de las regiones segmentadas en imágenes TEM.

Etapa 3 del módulo TEM del proyecto "Analizador de Imágenes de
Microscopía". Recibe la máscara de la Etapa 2 y mide tamaño, forma y
orientación de cada región. Todas las medidas están en píxeles. No
asume nada sobre qué material es cada fase.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from skimage import measure

from .loader import DEFAULT_OUTPUT_DIR
from .segmentation import CLEAN_MIN_FRACTION, PHASE_COLORS, _phase_labels

# --------------------------------------------------------------------
# Parámetros del Bloque 2 (tamaño y forma)
# --------------------------------------------------------------------

# Mínimo de regiones para ajustar la log-normal
LOGNORM_MIN_REGIONS: int = 5
# p-valor del test KS a partir del cual el ajuste se considera bueno
KS_PVALUE_OK: float = 0.05
# Reglas de clasificación de forma (se aplican en este orden)
CIRCULAR_MIN_CIRCULARITY: float = 0.85
ELONGATED_MIN_ASPECT: float = 2.0
IRREGULAR_MAX_SOLIDITY: float = 0.80
# Rangos para la distribución de la excentricidad
ECCENTRICITY_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# --------------------------------------------------------------------
# Parámetros del Bloque 3 (orientación)
# --------------------------------------------------------------------

# Las regiones casi circulares no tienen eje mayor definido: con
# excentricidad menor que esta no se usan para la orientación
ORIENTATION_MIN_ECCENTRICITY: float = 0.5
# Con S mayor que este valor la fase se considera orientada
ORIENTED_MIN_S: float = 0.5
# Con menos regiones no se clasifica: al azar, S vale ~0.89/sqrt(n)
# (con 3 regiones ya da ~0.5)
ORIENTATION_MIN_REGIONS: int = 10
# Sectores de la rosa de orientaciones (de 10° en 0°-180°)
ROSE_BINS: int = 18

# ====================================================================
# BLOQUE 1 — Extracción de propiedades por región
# ====================================================================


def _majority_phase(mask: np.ndarray) -> int | None:
    """Fase con más píxeles (fondo o matriz).

    Args:
        mask: Máscara de etiquetas (-1 = interfaz, 0, 1, ... = fases).

    Returns:
        Etiqueta de la fase mayoritaria o None si no hay fases.
    """
    phases = _phase_labels(mask)
    if not phases:
        return None
    return max(phases, key=lambda k: int(np.sum(mask == k)))


def _orientation_deg(orientation_rad: float) -> float:
    """Convierte la orientación de scikit-image a grados desde la
    horizontal, entre 0° y 180°.

    scikit-image mide el ángulo entre el eje mayor y el eje de las
    filas (vertical), entre -90° y 90°. Acá: 0° = horizontal, 90° =
    vertical, medido contra reloj tal como se ve la imagen.

    Args:
        orientation_rad: Valor de RegionProperties.orientation.

    Returns:
        Ángulo en grados, en [0, 180).
    """
    return float((np.degrees(orientation_rad) + 90.0) % 180.0)


def _phase_regions(
    mask: np.ndarray, phase: int, image: np.ndarray | None = None
) -> tuple[list, int]:
    """Regiones válidas de una fase (sin fragmentos).

    Es el criterio único de la Etapa 3 para separar regiones: lo usan
    extract_region_properties y show_morphology_results, así las
    medidas y los mapas salen siempre de las mismas regiones.

    Args:
        mask: Máscara de etiquetas.
        phase: Fase a separar.
        image: Imagen de intensidad (opcional) para regionprops.

    Returns:
        Tupla (lista de RegionProperties con área >= tamaño mínimo,
        cantidad de fragmentos ignorados).
    """
    min_region = max(1, int(round(CLEAN_MIN_FRACTION * mask.size)))
    labels = measure.label(mask == phase, connectivity=1)
    valid, n_fragments = [], 0
    for region in measure.regionprops(labels, intensity_image=image):
        if region.area < min_region:
            n_fragments += 1
        else:
            valid.append(region)
    return valid, n_fragments


def extract_region_properties(
    mask: np.ndarray,
    image: np.ndarray,
    phases: list[int] | None = None,
) -> dict[str, list[dict]]:
    """Mide cada región conectada de cada fase.

    Por defecto analiza todas las fases menos la mayoritaria (fondo o
    matriz), así funciona igual en campo claro y en campo oscuro. Las
    regiones se etiquetan con vecindad de 4 píxeles y se ignoran los
    fragmentos de menos de CLEAN_MIN_FRACTION de la imagen, con el
    mismo criterio que compute_phase_stats de la Etapa 2.

    Args:
        mask: Máscara de la Etapa 2 (-1 = interfaz, 0, 1, ... = fases).
        image: Imagen 2D con el mismo shape que mask.
        phases: Fases a analizar. None para todas menos la mayoritaria.

    Returns:
        Diccionario {'fase_N': [región, ...]}. Cada región es un dict
        con id, area_px, perimetro_px, eje_mayor_px, eje_menor_px,
        orientacion_deg (0° = horizontal, 0°-180°), centroide (y, x),
        bbox (fila_min, col_min, fila_max, col_max), solidez,
        excentricidad y toca_borde.

    Raises:
        ValueError: Si mask e image no tienen el mismo shape.
    """
    if mask.shape != image.shape:
        raise ValueError(
            f"mask {mask.shape} e image {image.shape} deben tener el "
            "mismo shape."
        )

    present = _phase_labels(mask)
    majority = _majority_phase(mask)
    if phases is None:
        selected = [k for k in present if k != majority]
        print(
            f"Fases analizadas: {selected} (fase {majority} excluida "
            "por ser la mayoritaria: fondo o matriz)"
        )
    else:
        selected = [k for k in phases if k in present]
        print(f"Fases analizadas: {selected}")

    min_region = max(1, int(round(CLEAN_MIN_FRACTION * mask.size)))
    height, width = mask.shape
    props: dict[str, list[dict]] = {}

    for k in selected:
        valid, n_fragments = _phase_regions(mask, k, image)
        regions: list[dict] = []
        for region in valid:
            min_row, min_col, max_row, max_col = region.bbox
            regions.append({
                "id": len(regions) + 1,
                "area_px": float(region.area),
                "perimetro_px": float(region.perimeter),
                "eje_mayor_px": float(region.axis_major_length),
                "eje_menor_px": float(region.axis_minor_length),
                "orientacion_deg": _orientation_deg(region.orientation),
                "centroide": (float(region.centroid[0]),
                              float(region.centroid[1])),
                "bbox": (int(min_row), int(min_col), int(max_row),
                         int(max_col)),
                "solidez": float(region.solidity),
                "excentricidad": float(region.eccentricity),
                # Cortada por el borde: área y forma truncadas
                "toca_borde": bool(
                    min_row == 0 or min_col == 0
                    or max_row == height or max_col == width
                ),
            })
        props[f"fase_{k}"] = regions
        n_border = sum(r["toca_borde"] for r in regions)
        print(
            f"  fase_{k}: {len(regions)} región(es) ({n_border} tocan el "
            f"borde; {n_fragments} fragmento(s) < {min_region} px "
            "ignorados)"
        )
    return props


# ====================================================================
# BLOQUE 2 — Análisis de tamaño y forma
# ====================================================================


def _interior(regions: list[dict]) -> list[dict]:
    """Regiones que no tocan el borde (sus medidas no están truncadas).

    Args:
        regions: Regiones de una fase (de extract_region_properties).

    Returns:
        Las regiones con toca_borde = False.
    """
    return [r for r in regions if not r["toca_borde"]]


def _circularity(region: dict) -> float:
    """Circularidad 4π·área / perímetro², recortada a 1.

    El perímetro digital subestima el real en regiones chicas y la
    fórmula puede dar más de 1 (un círculo de radio 5 px da 1.16).

    Args:
        region: Región de extract_region_properties.

    Returns:
        Circularidad en [0, 1] (0 si el perímetro es 0).
    """
    if region["perimetro_px"] <= 0:
        return 0.0
    value = 4 * np.pi * region["area_px"] / region["perimetro_px"] ** 2
    return float(min(1.0, value))


def _aspect_ratio(region: dict) -> float | None:
    """Relación de aspecto eje mayor / eje menor.

    Args:
        region: Región de extract_region_properties.

    Returns:
        La relación, o None si el eje menor es 0 (línea de 1 píxel).
    """
    if region["eje_menor_px"] <= 0:
        return None
    return region["eje_mayor_px"] / region["eje_menor_px"]


def _classify(circularity: float, aspect: float | None,
              solidity: float) -> str:
    """Clasifica una forma con las reglas en orden.

    Args:
        circularity: Circularidad en [0, 1].
        aspect: Relación de aspecto (None si no se pudo calcular).
        solidity: Solidez en [0, 1].

    Returns:
        'circular', 'elongada', 'irregular' o 'compacta'.
    """
    if circularity > CIRCULAR_MIN_CIRCULARITY:
        return "circular"
    if aspect is not None and aspect > ELONGATED_MIN_ASPECT:
        return "elongada"
    if solidity < IRREGULAR_MAX_SOLIDITY:
        return "irregular"
    return "compacta"


def analyze_size_distribution(props: dict[str, list[dict]]) -> dict:
    """Estadísticas de área por fase y ajuste log-normal.

    Usa solo las regiones que no tocan el borde. El ajuste log-normal
    (scipy.stats.lognorm con loc fijo en 0) se evalúa con el test de
    Kolmogorov-Smirnov. Como los parámetros se ajustan con los mismos
    datos, el p-valor es optimista; y con cientos de regiones el test
    detecta diferencias chicas: buen_ajuste es orientativo.

    Args:
        props: Resultado de extract_region_properties.

    Returns:
        Diccionario {'fase_N': {...}} con n_regiones,
        n_excluidas_borde, area_media_px, area_mediana_px, area_std_px,
        area_min_px, area_max_px, iqr_px, percentiles (p10, p25, p75,
        p90) y ajuste_lognormal (shape, loc, scale, ks_statistic,
        ks_pvalue, buen_ajuste; None con menos de LOGNORM_MIN_REGIONS
        regiones). Con 0 regiones, los valores son None.
    """
    result: dict[str, dict] = {}
    print("Distribución de tamaños (regiones que no tocan el borde):")
    for phase, regions in props.items():
        inner = _interior(regions)
        areas = np.array([r["area_px"] for r in inner], dtype=float)
        entry: dict = {
            "n_regiones": int(areas.size),
            "n_excluidas_borde": len(regions) - len(inner),
        }
        if areas.size == 0:
            entry.update({
                k: None for k in (
                    "area_media_px", "area_mediana_px", "area_std_px",
                    "area_min_px", "area_max_px", "iqr_px", "percentiles",
                    "ajuste_lognormal",
                )
            })
            result[phase] = entry
            print(f"  {phase}: sin regiones para analizar")
            continue

        p10, p25, p75, p90 = np.percentile(areas, [10, 25, 75, 90])
        entry.update({
            "area_media_px": float(areas.mean()),
            "area_mediana_px": float(np.median(areas)),
            "area_std_px": float(areas.std()),
            "area_min_px": float(areas.min()),
            "area_max_px": float(areas.max()),
            "iqr_px": float(p75 - p25),
            "percentiles": {"p10": float(p10), "p25": float(p25),
                            "p75": float(p75), "p90": float(p90)},
            "ajuste_lognormal": None,
        })

        ajuste = "sin ajuste (menos de " \
            f"{LOGNORM_MIN_REGIONS} regiones)"
        if areas.size >= LOGNORM_MIN_REGIONS and np.ptp(areas) > 0:
            shape, loc, scale = stats.lognorm.fit(areas, floc=0)
            ks = stats.kstest(areas, "lognorm", args=(shape, loc, scale))
            entry["ajuste_lognormal"] = {
                "shape": float(shape),
                "loc": float(loc),
                "scale": float(scale),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "buen_ajuste": bool(ks.pvalue > KS_PVALUE_OK),
            }
            ajuste = (
                f"log-normal s={shape:.2f}, escala={scale:.0f} px, "
                f"KS p={ks.pvalue:.3f} "
                f"({'buen' if ks.pvalue > KS_PVALUE_OK else 'mal'} "
                "ajuste)"
            )
        result[phase] = entry
        print(
            f"  {phase}: {areas.size} regiones "
            f"({entry['n_excluidas_borde']} excluidas por borde) | "
            f"media {entry['area_media_px']:.0f} px, mediana "
            f"{entry['area_mediana_px']:.0f} px, IQR "
            f"{entry['iqr_px']:.0f} px | {ajuste}"
        )
    return result


def analyze_shape(props: dict[str, list[dict]]) -> dict:
    """Estadísticas de forma por fase y forma predominante.

    Usa solo las regiones que no tocan el borde. La forma predominante
    se decide con los valores medios y estas reglas, en orden:
    circularidad > CIRCULAR_MIN_CIRCULARITY → 'circular'; relación de
    aspecto > ELONGATED_MIN_ASPECT → 'elongada'; solidez <
    IRREGULAR_MAX_SOLIDITY → 'irregular'; si no, 'compacta'. Las
    mismas reglas aplicadas a cada región dan conteo_formas.

    Args:
        props: Resultado de extract_region_properties.

    Returns:
        Diccionario {'fase_N': {...}} con n_regiones,
        circularidad_media, excentricidad_media,
        excentricidad_percentiles (p10, p50, p90),
        excentricidad_histograma (conteo por rango de
        ECCENTRICITY_BINS), solidez_media, relacion_aspecto_media,
        forma_predominante y conteo_formas. Con 0 regiones, los valores
        son None.
    """
    result: dict[str, dict] = {}
    print("Forma (regiones que no tocan el borde):")
    for phase, regions in props.items():
        inner = _interior(regions)
        if not inner:
            result[phase] = {
                "n_regiones": 0, "circularidad_media": None,
                "excentricidad_media": None,
                "excentricidad_percentiles": None,
                "excentricidad_histograma": None, "solidez_media": None,
                "relacion_aspecto_media": None,
                "forma_predominante": None, "conteo_formas": None,
            }
            print(f"  {phase}: sin regiones para analizar")
            continue

        circ = np.array([_circularity(r) for r in inner])
        ecc = np.array([r["excentricidad"] for r in inner])
        sol = np.array([r["solidez"] for r in inner])
        aspects = [_aspect_ratio(r) for r in inner]
        valid_aspects = np.array([a for a in aspects if a is not None])
        aspect_mean = (float(valid_aspects.mean())
                       if valid_aspects.size else None)

        counts = {"circular": 0, "elongada": 0, "irregular": 0,
                  "compacta": 0}
        for c, a, s in zip(circ, aspects, sol):
            counts[_classify(c, a, s)] += 1
        predominant = _classify(float(circ.mean()), aspect_mean,
                                float(sol.mean()))
        hist, _ = np.histogram(ecc, bins=ECCENTRICITY_BINS)
        p10, p50, p90 = np.percentile(ecc, [10, 50, 90])

        result[phase] = {
            "n_regiones": len(inner),
            "circularidad_media": float(circ.mean()),
            "excentricidad_media": float(ecc.mean()),
            "excentricidad_percentiles": {
                "p10": float(p10), "p50": float(p50), "p90": float(p90),
            },
            "excentricidad_histograma": [int(n) for n in hist],
            "solidez_media": float(sol.mean()),
            "relacion_aspecto_media": aspect_mean,
            "forma_predominante": predominant,
            "conteo_formas": counts,
        }
        aspect_txt = f"{aspect_mean:.2f}" if aspect_mean else "-"
        conteo = ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        print(
            f"  {phase}: {len(inner)} regiones | circularidad "
            f"{circ.mean():.2f}, excentricidad {ecc.mean():.2f}, solidez "
            f"{sol.mean():.2f}, aspecto {aspect_txt} → "
            f"'{predominant}' (por región: {conteo})"
        )
    return result


# ====================================================================
# BLOQUE 3 — Análisis de orientación
# ====================================================================


def _random_s(n: int) -> float:
    """Índice S esperado si las orientaciones fueran al azar.

    Para n ángulos uniformes, S sigue una distribución de Rayleigh
    con media sqrt(pi) / (2 sqrt(n)) ≈ 0.89 / sqrt(n).

    Args:
        n: Cantidad de regiones.

    Returns:
        Valor esperado de S (0 si n es 0).
    """
    return float(math.sqrt(math.pi) / (2 * math.sqrt(n))) if n else 0.0


def analyze_orientation(props: dict[str, list[dict]]) -> dict:
    """Estadística de orientación por fase.

    Usa las regiones que no tocan el borde y tienen excentricidad de al
    menos ORIENTATION_MIN_ECCENTRICITY (las casi circulares no tienen
    eje mayor definido). Los ángulos son axiales (0° y 180° son la
    misma dirección): el ángulo medio y el desvío se calculan con
    estadística circular de período π. El índice de orden es
    S = sqrt(<cos 2θ>² + <sin 2θ>²), entre 0 (al azar) y 1 (todas
    alineadas). Con menos de ORIENTATION_MIN_REGIONS regiones no se
    clasifica, porque S sale alto por casualidad.

    Args:
        props: Resultado de extract_region_properties.

    Returns:
        Diccionario {'fase_N': {...}} con n_regiones,
        n_excluidas_isotropas, angulo_medio_deg, std_circular_deg,
        indice_orden_S, S_esperado_aleatorio, clasificacion
        ('orientado', 'aleatorio' o 'insuficiente') y angulos_deg.
        Sin regiones, los valores son None.
    """
    result: dict[str, dict] = {}
    print("Orientación (regiones que no tocan el borde, excentricidad "
          f">= {ORIENTATION_MIN_ECCENTRICITY}):")
    for phase, regions in props.items():
        inner = _interior(regions)
        used = [r for r in inner
                if r["excentricidad"] >= ORIENTATION_MIN_ECCENTRICITY]
        angles_deg = [r["orientacion_deg"] for r in used]
        entry: dict = {
            "n_regiones": len(used),
            "n_excluidas_isotropas": len(inner) - len(used),
            "angulos_deg": angles_deg,
        }
        if not used:
            entry.update({
                "angulo_medio_deg": None, "std_circular_deg": None,
                "indice_orden_S": None, "S_esperado_aleatorio": None,
                "clasificacion": None,
            })
            result[phase] = entry
            print(f"  {phase}: sin regiones con orientación definida")
            continue

        theta = np.radians(angles_deg)
        mean = stats.circmean(theta, high=np.pi, low=0)
        std = stats.circstd(theta, high=np.pi, low=0)
        s_index = float(np.hypot(np.cos(2 * theta).mean(),
                                 np.sin(2 * theta).mean()))
        if len(used) < ORIENTATION_MIN_REGIONS:
            label = "insuficiente"
        elif s_index > ORIENTED_MIN_S:
            label = "orientado"
        else:
            label = "aleatorio"
        entry.update({
            "angulo_medio_deg": float(np.degrees(mean) % 180.0),
            "std_circular_deg": float(np.degrees(std)),
            "indice_orden_S": s_index,
            "S_esperado_aleatorio": _random_s(len(used)),
            "clasificacion": label,
        })
        result[phase] = entry
        print(
            f"  {phase}: {len(used)} regiones "
            f"({entry['n_excluidas_isotropas']} casi circulares "
            f"excluidas) | ángulo medio "
            f"{entry['angulo_medio_deg']:.0f}°, desvío "
            f"{entry['std_circular_deg']:.0f}°, S = {s_index:.2f} (al "
            f"azar ~{entry['S_esperado_aleatorio']:.2f}) → '{label}'"
        )
    return result


def plot_orientation_rose(
    props: dict[str, list[dict]],
    filepath: str | Path,
    orientation_stats: dict | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Rosa de orientaciones (gráfico polar) por fase.

    Cada rosa divide 0°-180° en ROSE_BINS sectores y repite los datos
    en 180°-360° para que quede simétrica. El radio es la cantidad de
    regiones; la línea punteada marca el ángulo medio. Se guarda como
    '<nombre>_orientacion.png'.

    Args:
        props: Resultado de extract_region_properties.
        filepath: Ruta del archivo de origen (para título y nombre).
        orientation_stats: Resultado de analyze_orientation. Si es
            None, se calcula.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    if orientation_stats is None:
        orientation_stats = analyze_orientation(props)
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_orientacion.png"

    phases = list(props)
    n = max(1, len(phases))
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows),
                             subplot_kw={"projection": "polar"},
                             squeeze=False)
    axes = axes.ravel()
    width = np.pi / ROSE_BINS

    for ax, phase in zip(axes, phases):
        color = PHASE_COLORS[int(phase.split("_")[1]) % len(PHASE_COLORS)]
        st = orientation_stats[phase]
        angles = np.radians(st["angulos_deg"])
        counts, edges = np.histogram(angles, bins=ROSE_BINS,
                                     range=(0, np.pi))
        # Sectores en 0°-180° y su copia en 180°-360°
        centers = edges[:-1] + width / 2
        ax.bar(np.concatenate([centers, centers + np.pi]),
               np.concatenate([counts, counts]), width=width,
               color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        if st["angulo_medio_deg"] is not None:
            mean = np.radians(st["angulo_medio_deg"])
            top = max(1, counts.max()) * 1.05
            for a in (mean, mean + np.pi):
                ax.plot([a, a], [0, top], "k--", linewidth=1.5)
            ax.set_title(
                f"Fase {phase.split('_')[1]} — S={st['indice_orden_S']:.2f} "
                f"({st['clasificacion']})\nn = {st['n_regiones']}, "
                f"ángulo medio {st['angulo_medio_deg']:.0f}°",
                pad=18,
            )
        else:
            ax.set_title(f"Fase {phase.split('_')[1]} — sin regiones",
                         pad=18)
    for ax in axes[len(phases):]:
        ax.axis("off")

    fig.suptitle(f"Orientación de regiones: {path.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ====================================================================
# BLOQUE 4 — Visualización morfológica
# ====================================================================


def _property_maps(
    mask: np.ndarray, props: dict[str, list[dict]]
) -> tuple[np.ndarray, np.ndarray]:
    """Mapas con el área y la excentricidad de cada región.

    Args:
        mask: Máscara de etiquetas.
        props: Resultado de extract_region_properties (define qué
            fases se dibujan).

    Returns:
        Tupla (mapa de área, mapa de excentricidad). Fuera de las
        regiones analizadas los valores son NaN.
    """
    area_map = np.full(mask.shape, np.nan)
    ecc_map = np.full(mask.shape, np.nan)
    for phase in props:
        for region in _phase_regions(mask, int(phase.split("_")[1]))[0]:
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            area_map[rows, cols] = region.area
            ecc_map[rows, cols] = region.eccentricity
    return area_map, ecc_map


def show_morphology_results(
    image: np.ndarray,
    mask: np.ndarray,
    props: dict[str, list[dict]],
    size_stats: dict,
    shape_stats: dict,
    filepath: str | Path,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la figura completa del análisis morfológico.

    Paneles: (1) imagen con una elipse ajustada por región, del color
    de su fase (punteada si toca el borde); (2) mapa de calor del área
    de cada región (escala logarítmica); (3) mapa de calor de la
    excentricidad; (4 en adelante) por fase, histograma de áreas con la
    log-normal (roja si el ajuste es bueno, gris punteada si no) y
    líneas de media (rojo) y mediana (verde). Se guarda como
    '<nombre>_morfologia.png'.

    Args:
        image: Imagen original 2D.
        mask: Máscara de la Etapa 2.
        props: Resultado de extract_region_properties.
        size_stats: Resultado de analyze_size_distribution.
        shape_stats: Resultado de analyze_shape.
        filepath: Ruta del archivo de origen (para título y nombre).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Ellipse

    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_morfologia.png"

    phases = list(props)
    total = 3 + len(phases)
    cols = 3 if total <= 6 else 4
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows),
                             squeeze=False)
    axes = axes.ravel()

    # Panel 1: elipses ajustadas sobre la imagen
    axes[0].imshow(image, cmap="gray")
    for phase in phases:
        color = PHASE_COLORS[int(phase.split("_")[1]) % len(PHASE_COLORS)]
        for r in props[phase]:
            axes[0].add_patch(Ellipse(
                (r["centroide"][1], r["centroide"][0]),
                width=r["eje_mayor_px"], height=r["eje_menor_px"],
                # En pantalla el eje y apunta hacia abajo: ángulo negativo
                angle=-r["orientacion_deg"], fill=False, edgecolor=color,
                linewidth=1, linestyle=":" if r["toca_borde"] else "-",
            ))
        axes[0].plot([], [], color=color,
                     label=f"{phase} ({len(props[phase])})")
    # Las elipses de zonas grandes pueden salir de la imagen: el panel
    # se limita a la imagen para que no se achique
    axes[0].set_xlim(-0.5, image.shape[1] - 0.5)
    axes[0].set_ylim(image.shape[0] - 0.5, -0.5)
    axes[0].set_title("Elipses ajustadas (punteadas: tocan el borde)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axis("off")

    # Paneles 2 y 3: mapas de calor de área y excentricidad
    area_map, ecc_map = _property_maps(mask, props)
    # Fuera de las regiones (NaN) se pinta gris oscuro
    cmap_area = plt.get_cmap("plasma").with_extremes(bad="#202020")
    cmap_ecc = plt.get_cmap("viridis").with_extremes(bad="#202020")
    if np.isfinite(area_map).any():
        shown = axes[1].imshow(
            np.ma.masked_invalid(area_map), cmap=cmap_area,
            norm=LogNorm(vmin=np.nanmin(area_map),
                         vmax=max(np.nanmax(area_map),
                                  np.nanmin(area_map) + 1)),
            interpolation="nearest",
        )
        fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.04,
                     label="Área (px, escala log)")
    axes[1].set_title("Área de cada región")
    axes[1].axis("off")
    shown = axes[2].imshow(np.ma.masked_invalid(ecc_map), cmap=cmap_ecc,
                           vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.04,
                 label="Excentricidad (0 = círculo, 1 = línea)")
    axes[2].set_title("Excentricidad de cada región")
    axes[2].axis("off")

    # Paneles por fase: distribución de tamaños
    for ax, phase in zip(axes[3:], phases):
        sizes = size_stats[phase]
        areas = np.array([r["area_px"] for r in _interior(props[phase])])
        form = shape_stats[phase]["forma_predominante"]
        if areas.size == 0:
            ax.text(0.5, 0.5, "Sin regiones que no toquen el borde",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{phase}")
            ax.axis("off")
            continue
        low = np.log10(areas.min()) - 0.05
        high = np.log10(areas.max()) + 0.05
        bins = np.logspace(low, high, 25)
        ax.hist(areas, bins=bins, density=True, color="tab:blue",
                alpha=0.6, label="áreas")
        fit = sizes["ajuste_lognormal"]
        if fit:
            x = np.logspace(low, high, 200)
            pdf = stats.lognorm.pdf(x, fit["shape"], fit["loc"],
                                    fit["scale"])
            if fit["buen_ajuste"]:
                ax.plot(x, pdf, color="red", linewidth=2,
                        label=f"log-normal (p={fit['ks_pvalue']:.3f})")
            else:
                ax.plot(x, pdf, color="gray", linestyle=":", linewidth=2,
                        label=f"log-normal, mal ajuste "
                              f"(p={fit['ks_pvalue']:.3f})")
        ax.axvline(sizes["area_media_px"], color="red", linestyle="--",
                   label=f"media {sizes['area_media_px']:.0f} px")
        ax.axvline(sizes["area_mediana_px"], color="green",
                   linestyle="--",
                   label=f"mediana {sizes['area_mediana_px']:.0f} px")
        ax.set_xscale("log")
        ax.set_xlabel("Área (px, escala log)")
        ax.set_ylabel("Densidad")
        ax.set_title(f"{phase}: '{form}', n = {sizes['n_regiones']}")
        ax.legend(fontsize=8)
    for ax in axes[3 + len(phases):]:
        ax.axis("off")

    fig.suptitle(f"Análisis morfológico TEM: {path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ====================================================================
# BLOQUE 5 — Pipeline principal
# ====================================================================

# Claves que debe traer el diccionario de pipeline_segmentation
_SEGMENTATION_KEYS: tuple[str, ...] = (
    "original", "processed", "mask", "n_phases", "filepath",
)


def pipeline_morphology(
    segmentation_dict: dict,
    phases: list[int] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Análisis morfológico completo de una imagen ya segmentada.

    Secuencia: extract_region_properties → analyze_size_distribution
    → analyze_shape → analyze_orientation → plot_orientation_rose →
    show_morphology_results. Todas las medidas en píxeles.

    Args:
        segmentation_dict: Diccionario de pipeline_segmentation, con
            'original', 'processed', 'mask', 'n_phases' y 'filepath'.
        phases: Fases a analizar. None para todas menos la mayoritaria
            (fondo o matriz).
        output_dir: Carpeta donde guardar las figuras. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra las figuras con plt.show().

    Returns:
        Diccionario con 'original', 'processed', 'mask', 'props',
        'size_stats', 'shape_stats', 'orientation_stats', 'filepath' y
        'n_phases'. Si la entrada trae 'lineas_watershed' (Etapa 2), se
        pasa tal cual: la usa la Etapa 4.

    Raises:
        ValueError: Si falta alguna clave en segmentation_dict.
    """
    missing = [k for k in _SEGMENTATION_KEYS
               if k not in segmentation_dict]
    if missing:
        raise ValueError(
            f"Al diccionario de entrada le falta(n): {', '.join(missing)}."
            " Debe ser el resultado de pipeline_segmentation()."
        )

    original = segmentation_dict["original"]
    processed = segmentation_dict["processed"]
    mask = segmentation_dict["mask"]
    filepath = segmentation_dict["filepath"]

    print("--- 1. Propiedades por región ---")
    props = extract_region_properties(mask, processed, phases)
    print("--- 2. Tamaño y forma ---")
    size_stats = analyze_size_distribution(props)
    shape_stats = analyze_shape(props)
    print("--- 3. Orientación ---")
    orientation_stats = analyze_orientation(props)
    plot_orientation_rose(props, filepath, orientation_stats,
                          output_dir=output_dir, show=show)
    print("--- 4. Figura morfológica ---")
    show_morphology_results(original, mask, props, size_stats,
                            shape_stats, filepath, output_dir=output_dir,
                            show=show)

    print(f"\nResumen morfológico: {Path(filepath).name}")
    for phase in props:
        size, form = size_stats[phase], shape_stats[phase]
        orient = orientation_stats[phase]
        if not size["n_regiones"]:
            print(f"  {phase}: sin regiones que no toquen el borde")
            continue
        s_txt = (f"S = {orient['indice_orden_S']:.2f} "
                 f"({orient['clasificacion']})"
                 if orient["indice_orden_S"] is not None else "S = -")
        print(
            f"  {phase}: {size['n_regiones']} regiones | área media "
            f"{size['area_media_px']:.0f} px (mediana "
            f"{size['area_mediana_px']:.0f}) | "
            f"'{form['forma_predominante']}' | {s_txt}"
        )

    result = {
        "original": original,
        "processed": processed,
        "mask": mask,
        "props": props,
        "size_stats": size_stats,
        "shape_stats": shape_stats,
        "orientation_stats": orientation_stats,
        "filepath": filepath,
        "n_phases": segmentation_dict["n_phases"],
    }
    if "lineas_watershed" in segmentation_dict:
        result["lineas_watershed"] = segmentation_dict["lineas_watershed"]
    return result
