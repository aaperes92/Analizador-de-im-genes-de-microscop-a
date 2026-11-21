"""Análisis espacial y de distribución de las fases en imágenes TEM.

Etapa 4, Módulo C, del módulo TEM del proyecto "Analizador de Imágenes
de Microscopía". Las fases se llaman siempre "Fase 1", "Fase 2", ...
y nunca se asignan elementos químicos. Todas las distancias están en
píxeles.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from .fraction import phase_names, phase_title
from .loader import DEFAULT_OUTPUT_DIR
from .segmentation import PHASE_COLORS

# --------------------------------------------------------------------
# Parámetros comunes
# --------------------------------------------------------------------

# Con menos partículas no se clasifica la distribución
MIN_PARTICLES: int = 10
# Semilla fija: la misma imagen da siempre el mismo resultado
RANDOM_SEED: int = 0

# --------------------------------------------------------------------
# Parámetros del Bloque C1 (RDF)
# --------------------------------------------------------------------

# La RDF se calcula hasta esta fracción del lado menor de la imagen
RDF_MAX_FRACTION: float = 0.25
# Repeticiones con puntos al azar (referencia y franja del azar)
RDF_RANDOM_REPEATS: int = 40
# Rango corto para clasificar: de 1 a 3 diámetros típicos
RDF_SHORT_RANGE: tuple[float, float] = (1.0, 3.0)
# Percentiles de la franja de lo que da el azar
RDF_ENVELOPE: tuple[float, float] = (2.5, 97.5)


# ====================================================================
# Datos de entrada compartidos por los bloques del Módulo C
# ====================================================================


def _phase_points(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    background: int | None = None,
) -> dict[str, dict]:
    """Centroides y áreas de cada fase, con los nombres de la Etapa 4.

    props viene de la Etapa 3, con las fases nombradas por su etiqueta
    de la máscara. Se usan todas las regiones, también las que tocan el
    borde (acá importa la posición de cada partícula).

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        {'fase_N': {'etiqueta_mascara', 'centroides' (array (n, 2) de
        (y, x)), 'areas' (array (n,))}}. Si props no tiene la fase,
        centroides y areas son None.
    """
    _, names = phase_names(mask, background)
    result: dict[str, dict] = {}
    for name, label in names.items():
        regions = props.get(f"fase_{label}")
        if regions is None:
            result[name] = {"etiqueta_mascara": label, "centroides": None,
                            "areas": None}
            continue
        result[name] = {
            "etiqueta_mascara": label,
            "centroides": np.array([r["centroide"] for r in regions],
                                   dtype=float).reshape(-1, 2),
            "areas": np.array([r["area_px"] for r in regions], dtype=float),
        }
    return result


# ====================================================================
# BLOQUE C1 — Función de distribución radial (RDF)
# ====================================================================


def _pair_histogram(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Cantidad de pares de puntos en cada intervalo de distancia.

    Args:
        points: Array (n, 2) de coordenadas.
        edges: Bordes de los intervalos de distancia.

    Returns:
        Conteo de pares (cada par una sola vez) por intervalo.
    """
    distances = cdist(points, points)
    upper = distances[np.triu_indices(len(points), k=1)]
    counts, _ = np.histogram(upper, bins=edges)
    return counts.astype(np.float64)


def compute_rdf(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    n_bins: int = 50,
    background: int | None = None,
) -> dict:
    """Función de distribución radial g(r) de las partículas de cada fase.

    g(r) compara la cantidad de pares de partículas a distancia r con
    la que habría si estuvieran al azar: > 1 más pares que al azar
    (agrupadas), < 1 menos (se excluyen), ≈ 1 al azar. La fórmula
    g(r) = dn(r) / (ρ · 2πr · dr) supone un plano infinito; en una
    imagen finita las partículas cerca del borde tienen menos vecinas.
    Por eso la referencia se calcula con la misma cantidad de puntos al
    azar en una ventana del mismo tamaño (RDF_RANDOM_REPEATS veces),
    lo que corrige el efecto del borde.

    Como las partículas tienen tamaño, g(r) ≈ 0 a distancias menores
    que su diámetro aunque estén agrupadas. La clasificación usa el
    promedio de g entre 1 y 3 diámetros típicos (RDF_SHORT_RANGE) y lo
    compara con la franja que dan las repeticiones al azar
    (RDF_ENVELOPE): por encima 'clustering', por debajo 'exclusion',
    dentro 'aleatorio'. Con menos de MIN_PARTICLES, 'insuficiente'.

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        n_bins: Cantidad de intervalos de distancia.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fase_N': {...}} con n, r (centros, px), g,
        g_azar_inf, g_azar_sup, g_corto, g_corto_azar (inferior,
        superior), rango_corto_px, diametro_tipico_px y clasificacion.
        Con menos de MIN_PARTICLES partículas, los arrays son None.
    """
    height, width = mask.shape
    r_max = RDF_MAX_FRACTION * min(height, width)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(RANDOM_SEED)
    result: dict[str, dict] = {}

    print(f"RDF g(r) hasta {r_max:.0f} px, referencia con "
          f"{RDF_RANDOM_REPEATS} repeticiones al azar:")
    for name, data in _phase_points(props, mask, background).items():
        points = data["centroides"]
        n = 0 if points is None else len(points)
        entry: dict = {"n": n, "r": None, "g": None, "g_azar_inf": None,
                       "g_azar_sup": None, "g_corto": None,
                       "g_corto_azar": None, "rango_corto_px": None,
                       "diametro_tipico_px": None,
                       "clasificacion": "insuficiente"}
        if n < MIN_PARTICLES:
            result[name] = entry
            print(f"  {phase_title(name)}: {n} partícula(s) (< "
                  f"{MIN_PARTICLES}) → insuficiente")
            continue

        observed = _pair_histogram(points, edges)
        scale = np.array([height, width], dtype=float)
        random_counts = np.array([
            _pair_histogram(rng.random((n, 2)) * scale, edges)
            for _ in range(RDF_RANDOM_REPEATS)
        ])
        expected = random_counts.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(expected > 0, observed / expected, np.nan)
            g_random = np.where(expected > 0, random_counts / expected,
                                np.nan)

        # Rango corto: de 1 a 3 diámetros típicos de partícula
        diameter = 2.0 * np.sqrt(np.median(data["areas"]) / np.pi)
        low, high = (f * diameter for f in RDF_SHORT_RANGE)
        short = (centers >= low) & (centers <= high)
        if not short.any():
            # Partículas muy grandes para la ventana: primer tercio
            short = centers <= r_max / 3
            low, high = 0.0, r_max / 3
        # Solo intervalos donde la referencia al azar tiene pares
        short &= expected > 0
        g_short = float(np.mean(g[short]))
        random_short = np.mean(g_random[:, short], axis=1)
        env_low, env_high = np.percentile(random_short, RDF_ENVELOPE)
        if g_short > env_high:
            label = "clustering"
        elif g_short < env_low:
            label = "exclusion"
        else:
            label = "aleatorio"

        # Franja por distancia (solo donde la referencia tiene pares)
        lo_curve = np.full(centers.size, np.nan)
        hi_curve = np.full(centers.size, np.nan)
        valid = expected > 0
        lo_curve[valid], hi_curve[valid] = np.percentile(
            g_random[:, valid], RDF_ENVELOPE, axis=0
        )
        entry.update({
            "r": centers, "g": g, "g_azar_inf": lo_curve,
            "g_azar_sup": hi_curve, "g_corto": g_short,
            "g_corto_azar": (float(env_low), float(env_high)),
            "rango_corto_px": (float(low), float(high)),
            "diametro_tipico_px": float(diameter),
            "clasificacion": label,
        })
        result[name] = entry
        print(f"  {phase_title(name)}: {n} partículas, diámetro típico "
              f"{diameter:.0f} px | g entre {low:.0f} y {high:.0f} px = "
              f"{g_short:.2f} (azar: {env_low:.2f}-{env_high:.2f}) → "
              f"'{label}'")
    return result


# ====================================================================
# BLOQUE C2 — Mapa de densidad 2D
# ====================================================================

# Partículas esperadas por celda con grid_size='auto'
DENSITY_PER_CELL: float = 5.0
# Límites de la grilla automática (celdas por lado)
DENSITY_GRID_RANGE: tuple[int, int] = (2, 32)
# Suavizado del mapa (en celdas)
DENSITY_SIGMA: float = 1.0
# Límites del coeficiente de variación (especificación)
CV_UNIFORM: float = 0.3
CV_HETEROGENEOUS: float = 0.7
# Repeticiones al azar para el CV de referencia
DENSITY_RANDOM_REPEATS: int = 200


def _density_grid(points: np.ndarray, shape: tuple[int, int],
                  grid: int) -> np.ndarray:
    """Cantidad de puntos en cada celda de una grilla grid × grid.

    Args:
        points: Array (n, 2) de coordenadas (y, x).
        shape: (alto, ancho) de la imagen.
        grid: Celdas por lado.

    Returns:
        Array (grid, grid) de conteos (fila = y, columna = x).
    """
    counts, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=grid,
        range=[[0, shape[0]], [0, shape[1]]],
    )
    return counts


def _smooth_cv(counts: np.ndarray) -> tuple[np.ndarray, float]:
    """Mapa suavizado y su coeficiente de variación.

    Args:
        counts: Conteos por celda.

    Returns:
        Tupla (mapa suavizado, CV = desvío / media; 0 si la media es 0).
    """
    smooth = ndimage.gaussian_filter(counts, DENSITY_SIGMA, mode="nearest")
    mean = smooth.mean()
    return smooth, float(smooth.std() / mean) if mean > 0 else 0.0


def _classify_cv(cv: float) -> str:
    """Clasificación del CV con los límites de la especificación."""
    if cv < CV_UNIFORM:
        return "uniforme"
    if cv <= CV_HETEROGENEOUS:
        return "moderadamente heterogénea"
    return "muy heterogénea"


def compute_density_map(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    grid_size: int | str = "auto",
    background: int | None = None,
) -> dict:
    """Mapa de densidad de partículas en una grilla y su uniformidad.

    Cuenta los centroides de cada celda de una grilla grid_size ×
    grid_size y suaviza el mapa con una gaussiana de DENSITY_SIGMA
    celdas. El coeficiente de variación (CV) del mapa suavizado se
    clasifica con los límites CV_UNIFORM y CV_HETEROGENEOUS.

    Con pocas partículas por celda, el CV es alto aunque las partículas
    estén al azar. Por eso se compara con el CV de
    DENSITY_RANDOM_REPEATS distribuciones al azar con la misma cantidad
    de puntos y la misma grilla: si no supera su percentil 97.5, la
    clasificación final es 'uniforme (compatible con azar)'.

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        grid_size: Celdas por lado, o 'auto' para unas
            DENSITY_PER_CELL partículas por celda (entre los límites de
            DENSITY_GRID_RANGE).
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fase_N': {...}} con n, grid_size, celda_px,
        mapa_conteos, mapa_suavizado, cv, cv_azar_medio, cv_azar_p975,
        heterogeneidad_significativa, clasificacion_cv, clasificacion,
        posicion_max_px y posicion_min_px (centro de la celda, (y, x)).
        Con menos de MIN_PARTICLES partículas, la clasificación es
        'insuficiente'; sin partículas, los mapas son None.

    Raises:
        ValueError: Si grid_size no es 'auto' ni un entero >= 1.
    """
    if not (grid_size == "auto"
            or (isinstance(grid_size, int) and grid_size >= 1)):
        raise ValueError(
            f"grid_size debe ser 'auto' o un entero >= 1; llegó "
            f"{grid_size!r}."
        )
    shape = mask.shape
    rng = np.random.default_rng(RANDOM_SEED)
    result: dict[str, dict] = {}
    print("Mapa de densidad (CV comparado con distribuciones al azar):")
    for name, data in _phase_points(props, mask, background).items():
        points = data["centroides"]
        n = 0 if points is None else len(points)
        if grid_size == "auto":
            low, high = DENSITY_GRID_RANGE
            grid = int(np.clip(np.floor(np.sqrt(n / DENSITY_PER_CELL)),
                               low, high))
        else:
            grid = grid_size
        cell = (shape[0] / grid, shape[1] / grid)
        entry: dict = {
            "n": n, "grid_size": grid, "celda_px": cell,
            "mapa_conteos": None, "mapa_suavizado": None, "cv": None,
            "cv_azar_medio": None, "cv_azar_p975": None,
            "heterogeneidad_significativa": None,
            "clasificacion_cv": None, "clasificacion": "insuficiente",
            "posicion_max_px": None, "posicion_min_px": None,
        }
        if n == 0:
            result[name] = entry
            print(f"  {phase_title(name)}: sin partículas")
            continue

        counts = _density_grid(points, shape, grid)
        smooth, cv = _smooth_cv(counts)
        scale = np.array(shape, dtype=float)
        random_cv = np.array([
            _smooth_cv(_density_grid(rng.random((n, 2)) * scale, shape,
                                     grid))[1]
            for _ in range(DENSITY_RANDOM_REPEATS)
        ])
        p975 = float(np.percentile(random_cv, 97.5))
        significant = cv > p975

        def cell_center(index: int) -> tuple[float, float]:
            row, col = np.unravel_index(index, smooth.shape)
            return ((row + 0.5) * cell[0], (col + 0.5) * cell[1])

        if n < MIN_PARTICLES:
            label = "insuficiente"
        elif significant:
            label = _classify_cv(cv)
        else:
            label = "uniforme (compatible con azar)"
        entry.update({
            "mapa_conteos": counts, "mapa_suavizado": smooth, "cv": cv,
            "cv_azar_medio": float(random_cv.mean()),
            "cv_azar_p975": p975,
            "heterogeneidad_significativa": bool(significant),
            "clasificacion_cv": _classify_cv(cv),
            "clasificacion": label,
            "posicion_max_px": cell_center(int(np.argmax(smooth))),
            "posicion_min_px": cell_center(int(np.argmin(smooth))),
        })
        result[name] = entry
        print(f"  {phase_title(name)}: {n} partículas, grilla {grid}×{grid}"
              f" | CV = {cv:.2f} (azar: {random_cv.mean():.2f}, p97.5 "
              f"{p975:.2f}) → '{label}'")
    return result


# ====================================================================
# BLOQUE C3 — Detección de clustering
# ====================================================================

# Fracción de partículas que se muestrea en el test de Hopkins
HOPKINS_SAMPLE_FRACTION: float = 0.1
HOPKINS_MIN_SAMPLE: int = 5
# Repeticiones del test (se promedia H)
HOPKINS_REPEATS: int = 20
# Límites de clasificación de H
HOPKINS_CLUSTERED: float = 0.75
HOPKINS_REGULAR: float = 0.25
# Potencia de las distancias en H (2 = dimensión de la imagen, variante
# de Lawson y Jurs): con 1, una grilla regular daba H = 0.28
HOPKINS_POWER: int = 2
# eps de DBSCAN = este factor × distancia media al vecino más cercano
DBSCAN_EPS_FACTOR: float = 2.0
# min_samples de DBSCAN: percentil de las vecinas esperadas al azar
# (Poisson) dentro de eps, más 1. Con n // 20 (especificación) no se
# encontraban grupos en prueba1 y sí en puntos al azar
DBSCAN_RANDOM_PERCENTILE: float = 0.99


def _hopkins(points: np.ndarray, shape: tuple[int, int],
             rng: np.random.Generator) -> float:
    """Estadístico de Hopkins promediado en HOPKINS_REPEATS muestras.

    H = Σu^d / (Σu^d + Σw^d), con d = HOPKINS_POWER, donde u son las
    distancias de m puntos al azar a la partícula más cercana y w las
    de m partículas reales a su vecina más cercana. Al azar H ≈ 0.5.

    Args:
        points: Array (n, 2) de coordenadas (y, x), n >= 2.
        shape: (alto, ancho) de la imagen.
        rng: Generador aleatorio.

    Returns:
        H promedio, entre 0 y 1.
    """
    tree = cKDTree(points)
    n = len(points)
    m = min(n, max(HOPKINS_MIN_SAMPLE,
                   int(round(HOPKINS_SAMPLE_FRACTION * n))))
    scale = np.array(shape, dtype=float)
    values = []
    for _ in range(HOPKINS_REPEATS):
        random_pts = rng.random((m, 2)) * scale
        u, _ = tree.query(random_pts, k=1)
        chosen = points[rng.choice(n, m, replace=False)]
        # k=2: el primer vecino de una partícula es ella misma
        w, _ = tree.query(chosen, k=2)
        u_d = (u ** HOPKINS_POWER).sum()
        w_d = (w[:, 1] ** HOPKINS_POWER).sum()
        values.append(u_d / (u_d + w_d))
    return float(np.mean(values))


def detect_clustering(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    background: int | None = None,
) -> dict:
    """Agrupamiento de partículas con el test de Hopkins y DBSCAN.

    Hopkins decide si hay agrupamiento: H > HOPKINS_CLUSTERED
    'clustering', H < HOPKINS_REGULAR 'regular', si no 'aleatorio'
    (promedio de HOPKINS_REPEATS muestras). Los puntos al azar cerca
    del borde empujan H un poco hacia 'clustering'.

    DBSCAN localiza y mide los grupos. eps = DBSCAN_EPS_FACTOR × la
    distancia media al vecino más cercano; min_samples es el
    percentil DBSCAN_RANDOM_PERCENTILE de la cantidad de vecinas que
    tendría una partícula al azar dentro de eps (Poisson con media
    densidad · π · eps²), más 1. Así una partícula es núcleo de un
    grupo solo si tiene más vecinas de lo que daría el azar.

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fase_N': {...}} con n, hopkins_statistic,
        hopkins_clasificacion, n_clusters_dbscan,
        n_particulas_en_clusters, n_particulas_aisladas, clusters
        (lista de {id, size, centroide (y, x), radio_px}), eps_px,
        min_samples y etiquetas_dbscan (una por partícula, -1 =
        aislada). Con menos de MIN_PARTICLES, la clasificación es
        'insuficiente' y los valores None.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    result: dict[str, dict] = {}
    print("Agrupamiento (Hopkins decide; DBSCAN localiza los grupos):")
    for name, data in _phase_points(props, mask, background).items():
        points = data["centroides"]
        n = 0 if points is None else len(points)
        entry: dict = {
            "n": n, "hopkins_statistic": None,
            "hopkins_clasificacion": "insuficiente",
            "n_clusters_dbscan": None, "n_particulas_en_clusters": None,
            "n_particulas_aisladas": None, "clusters": [],
            "eps_px": None, "min_samples": None, "etiquetas_dbscan": None,
        }
        if n < MIN_PARTICLES:
            result[name] = entry
            print(f"  {phase_title(name)}: {n} partícula(s) (< "
                  f"{MIN_PARTICLES}) → insuficiente")
            continue

        # Método 1: Hopkins
        h = _hopkins(points, mask.shape, rng)
        if h > HOPKINS_CLUSTERED:
            h_label = "clustering"
        elif h < HOPKINS_REGULAR:
            h_label = "regular"
        else:
            h_label = "aleatorio"

        # Método 2: DBSCAN
        nearest, _ = cKDTree(points).query(points, k=2)
        eps = DBSCAN_EPS_FACTOR * float(nearest[:, 1].mean())
        # Vecinas esperadas al azar dentro de eps (Poisson)
        expected = n / float(mask.size) * np.pi * eps ** 2
        min_samples = max(2, int(stats.poisson.ppf(
            DBSCAN_RANDOM_PERCENTILE, expected)) + 1)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            points)
        clusters = []
        for k in sorted(set(labels) - {-1}):
            members = points[labels == k]
            center = members.mean(axis=0)
            radius = float(np.max(np.hypot(*(members - center).T)))
            clusters.append({
                "id": int(k) + 1, "size": int(len(members)),
                "centroide": (float(center[0]), float(center[1])),
                "radio_px": radius,
            })
        isolated = int(np.sum(labels == -1))
        entry.update({
            "hopkins_statistic": h, "hopkins_clasificacion": h_label,
            "n_clusters_dbscan": len(clusters),
            "n_particulas_en_clusters": n - isolated,
            "n_particulas_aisladas": isolated, "clusters": clusters,
            "eps_px": eps, "min_samples": min_samples,
            "etiquetas_dbscan": labels,
        })
        result[name] = entry
        print(f"  {phase_title(name)}: {n} partículas | Hopkins H = "
              f"{h:.2f} → '{h_label}' | DBSCAN (eps {eps:.0f} px, "
              f"min {min_samples}): {len(clusters)} grupo(s), "
              f"{isolated} aislada(s)")
    return result


# ====================================================================
# BLOQUE C4 — Detección de gradientes espaciales
# ====================================================================

# Cantidad de franjas en cada dirección
GRADIENT_STRIPS: int = 10
# Nivel de significación antes de la corrección de Bonferroni
GRADIENT_ALPHA: float = 0.05
# Tests por fase: 3 magnitudes × 2 direcciones
GRADIENT_N_TESTS: int = 6
# Mínimo de franjas con datos para ajustar la recta
GRADIENT_MIN_STRIPS: int = 3

# Cómo se describe una pendiente en cada dirección
_DIRECTION_WORDS: dict[str, tuple[str, str]] = {
    "vertical": ("aumenta hacia abajo", "disminuye hacia abajo"),
    "horizontal": ("aumenta hacia la derecha",
                   "disminuye hacia la derecha"),
}


def _fit_strips(positions: np.ndarray, values: np.ndarray) -> dict:
    """Recta por los valores de las franjas (linregress).

    Args:
        positions: Centro de cada franja (px).
        values: Valor medido en cada franja (NaN = sin datos).

    Returns:
        Diccionario con posiciones_px, valores, pendiente_por_100px,
        r2, p_value y significativo (False si hay menos de
        GRADIENT_MIN_STRIPS franjas con datos).
    """
    ok = np.isfinite(values)
    entry = {"posiciones_px": positions, "valores": values,
             "pendiente_por_100px": None, "r2": None, "p_value": None,
             "significativo": False}
    # Sin variación real (iguales salvo redondeo) no hay recta que ajustar
    if (ok.sum() < GRADIENT_MIN_STRIPS
            or np.allclose(values[ok], values[ok][0], rtol=1e-9,
                           atol=1e-12)):
        return entry
    fit = stats.linregress(positions[ok], values[ok])
    threshold = GRADIENT_ALPHA / GRADIENT_N_TESTS
    entry.update({
        "pendiente_por_100px": float(100.0 * fit.slope),
        "intercepto": float(fit.intercept),
        "r2": float(fit.rvalue ** 2),
        "p_value": float(fit.pvalue),
        "significativo": bool(fit.pvalue < threshold),
    })
    return entry


def detect_spatial_gradients(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    image: np.ndarray,
    background: int | None = None,
) -> dict:
    """Gradientes espaciales de densidad, tamaño e intensidad por fase.

    La imagen se divide en GRADIENT_STRIPS franjas horizontales
    (dirección 'vertical': de arriba hacia abajo) y verticales
    (dirección 'horizontal': de izquierda a derecha). En cada franja se
    mide la cantidad de partículas (densidad), su área media (tamaño,
    solo regiones que no tocan el borde) y la intensidad media de los
    píxeles de la fase. Una recta (scipy.stats.linregress) indica si
    hay gradiente; es significativo si su p-valor pasa la corrección
    de Bonferroni: p < GRADIENT_ALPHA / GRADIENT_N_TESTS (6 tests por
    fase). Un gradiente de intensidad puede deberse a iluminación
    despareja o a cambios de espesor, no solo a la fase.

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        image: Imagen preprocesada en [0, 1].
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fase_N': {densidad, tamanio, intensidad,
        gradientes_significativos, umbral_p, n}}; densidad, tamanio e
        intensidad tienen 'vertical' y 'horizontal' (ver _fit_strips).
        Con menos de MIN_PARTICLES partículas, densidad y tamanio son
        None.

    Raises:
        ValueError: Si image y mask no tienen el mismo shape.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image {image.shape} y mask {mask.shape} deben tener el "
            "mismo shape."
        )
    height, width = mask.shape
    threshold = GRADIENT_ALPHA / GRADIENT_N_TESTS
    rows, cols = np.mgrid[:height, :width]
    result: dict[str, dict] = {}
    print(f"Gradientes espaciales ({GRADIENT_STRIPS} franjas, "
          f"significativo si p < {threshold:.4f} con Bonferroni):")
    for name, data in _phase_points(props, mask, background).items():
        label = data["etiqueta_mascara"]
        points = data["centroides"]
        n = 0 if points is None else len(points)
        region = mask == label
        regions = props.get(f"fase_{label}") or []
        inner = np.array([not r["toca_borde"] for r in regions],
                         dtype=bool)
        entry: dict = {"n": n, "umbral_p": threshold,
                       "densidad": None, "tamanio": None,
                       "intensidad": {}, "gradientes_significativos": []}
        if n >= MIN_PARTICLES:
            entry["densidad"], entry["tamanio"] = {}, {}

        for direction, axis, size, coords in (
            ("vertical", 0, height, rows),
            ("horizontal", 1, width, cols),
        ):
            edges = np.linspace(0, size, GRADIENT_STRIPS + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            # Intensidad media de la fase en cada franja
            strip_of_px = np.clip(np.digitize(coords[region], edges) - 1,
                                  0, GRADIENT_STRIPS - 1)
            sums = np.bincount(strip_of_px, weights=image[region],
                               minlength=GRADIENT_STRIPS)
            counts = np.bincount(strip_of_px, minlength=GRADIENT_STRIPS)
            with np.errstate(invalid="ignore", divide="ignore"):
                intensity = np.where(counts > 0, sums / counts, np.nan)
            entry["intensidad"][direction] = _fit_strips(centers,
                                                         intensity)
            if n < MIN_PARTICLES:
                continue
            strip = np.clip(np.digitize(points[:, axis], edges) - 1, 0,
                            GRADIENT_STRIPS - 1)
            n_per_strip = np.bincount(strip, minlength=GRADIENT_STRIPS)
            entry["densidad"][direction] = _fit_strips(
                centers, n_per_strip.astype(float))
            area_sums = np.bincount(strip[inner],
                                    weights=data["areas"][inner],
                                    minlength=GRADIENT_STRIPS)
            area_counts = np.bincount(strip[inner],
                                      minlength=GRADIENT_STRIPS)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_area = np.where(area_counts > 0,
                                     area_sums / area_counts, np.nan)
            entry["tamanio"][direction] = _fit_strips(centers, mean_area)

        for quantity in ("densidad", "tamanio", "intensidad"):
            for direction, fit in (entry[quantity] or {}).items():
                if fit["significativo"]:
                    up, down = _DIRECTION_WORDS[direction]
                    word = up if fit["pendiente_por_100px"] > 0 else down
                    entry["gradientes_significativos"].append(
                        f"{quantity} {direction}: {word} (p = "
                        f"{fit['p_value']:.1e}, R² = {fit['r2']:.2f})"
                    )
        result[name] = entry
        found = entry["gradientes_significativos"]
        extra = "" if n >= MIN_PARTICLES else (
            f" (densidad y tamaño: {n} partícula(s), insuficiente)")
        print(f"  {phase_title(name)}: "
              + ("; ".join(found) if found else "sin gradientes "
                 "significativos") + extra)
    return result


# ====================================================================
# BLOQUE C5 — Visualización y pipeline espacial
# ====================================================================

# Las elipses de los grupos abarcan esta cantidad de desvíos por eje
CLUSTER_ELLIPSE_SIGMAS: float = 2.0


def _cluster_ellipse(members: np.ndarray, color: str) -> Ellipse:
    """Elipse que resume la forma de un grupo de partículas.

    Args:
        members: Array (k, 2) de coordenadas (y, x) del grupo.
        color: Color del borde.

    Returns:
        Elipse centrada en el grupo, de CLUSTER_ELLIPSE_SIGMAS desvíos
        por eje (un círculo mínimo si el grupo tiene menos de 3
        partículas).
    """
    xy = members[:, ::-1]  # (x, y) para matplotlib
    center = xy.mean(axis=0)
    if len(xy) < 3:
        radius = max(10.0, float(np.max(np.hypot(*(xy - center).T))))
        return Ellipse(center, 2 * radius, 2 * radius, fill=False,
                       edgecolor=color, linewidth=1.5)
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    values = np.clip(values, 1.0, None)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    k = 2 * CLUSTER_ELLIPSE_SIGMAS
    return Ellipse(center, k * np.sqrt(values[1]), k * np.sqrt(values[0]),
                   angle=angle, fill=False, edgecolor=color, linewidth=1.5)


def _phase_summary(name: str, rdf: dict, density: dict, clustering: dict,
                   gradients: dict) -> str:
    """Texto con las clasificaciones de una fase."""
    r, d, c = rdf[name], density[name], clustering[name]
    g = gradients[name]["gradientes_significativos"]
    h = ("-" if c["hopkins_statistic"] is None
         else f"{c['hopkins_statistic']:.2f}")
    return (f"{phase_title(name)} (n={r['n']})\n"
            f"  RDF: {r['clasificacion']}\n"
            f"  Densidad: {d['clasificacion']}\n"
            f"  Hopkins H={h}: {c['hopkins_clasificacion']}\n"
            f"  DBSCAN: {c['n_clusters_dbscan'] or 0} grupo(s)\n"
            f"  Gradientes: {len(g) if g else 'ninguno'}")


def show_spatial_results(
    image: np.ndarray,
    mask: np.ndarray,
    rdf: dict,
    density_map: dict,
    clustering: dict,
    gradients: dict,
    filepath: str | Path,
    props: dict[str, list[dict]] | None = None,
    background: int | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la figura del análisis espacial.

    Primera fila: g(r) de todas las fases (línea punteada en g = 1,
    franja del azar y zonas con g > 1 en verde claro) y un resumen de
    las clasificaciones. Luego, una fila por fase: (1) mapa de
    densidad sobre la imagen con los grupos de DBSCAN como elipses;
    (3) partículas coloreadas por grupo (aisladas en gris); (4)
    partículas por franja en las dos direcciones, con la recta de
    regresión (continua si el gradiente es significativo). Se guarda
    como '<nombre>_espacial.png'.

    Args:
        image: Imagen que se muestra de fondo (en el pipeline, la
            original).
        mask: Máscara de la Etapa 2.
        rdf: Resultado de compute_rdf.
        density_map: Resultado de compute_density_map.
        clustering: Resultado de detect_clustering.
        gradients: Resultado de detect_spatial_gradients.
        filepath: Ruta del archivo de origen (para título y nombre).
        props: Resultado de extract_region_properties (para dibujar
            las partículas y las elipses; sin props no se dibujan).
        background: Etiqueta del fondo usada en el análisis.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_espacial.png"
    points = (_phase_points(props, mask, background) if props is not None
              else {})
    phases = list(rdf)
    height, width = mask.shape

    fig = plt.figure(figsize=(19, 6 + 6 * len(phases)))
    grid = fig.add_gridspec(1 + len(phases), 3)

    # Primera fila: RDF de todas las fases y resumen
    ax_rdf = fig.add_subplot(grid[0, :2])
    for name, r in rdf.items():
        if r["g"] is None:
            continue
        label = _phase_points({}, mask, background)[name][
            "etiqueta_mascara"]
        color = PHASE_COLORS[label % len(PHASE_COLORS)]
        ax_rdf.fill_between(r["r"], r["g_azar_inf"], r["g_azar_sup"],
                            color=color, alpha=0.12)
        ax_rdf.fill_between(r["r"], 1, r["g"], where=r["g"] > 1,
                            color="lightgreen", alpha=0.5)
        ax_rdf.plot(r["r"], r["g"], "o-", color=color, markersize=3,
                    label=f"{phase_title(name)}: {r['clasificacion']}")
    ax_rdf.axhline(1, color="black", linestyle="--", linewidth=1,
                   label="g = 1 (al azar)")
    ax_rdf.set_xlabel("r: distancia entre centroides (px)")
    ax_rdf.set_ylabel("g(r)")
    ax_rdf.set_title("Función de distribución radial (verde: g > 1; "
                     "franja: lo que da el azar)")
    ax_rdf.legend(fontsize=9)
    ax_sum = fig.add_subplot(grid[0, 2])
    ax_sum.axis("off")
    ax_sum.text(0.02, 0.98, "\n\n".join(
        _phase_summary(n, rdf, density_map, clustering, gradients)
        for n in phases), va="top", family="monospace", fontsize=10,
        transform=ax_sum.transAxes)
    ax_sum.set_title("Resumen por fase")

    # Una fila por fase
    for row, name in enumerate(phases, start=1):
        d, c, g = density_map[name], clustering[name], gradients[name]
        pts = points.get(name, {}).get("centroides")
        title = phase_title(name)

        # Panel 1: densidad + elipses de los grupos
        ax1 = fig.add_subplot(grid[row, 0])
        ax1.imshow(image, cmap="gray")
        if d["mapa_suavizado"] is not None:
            ax1.imshow(d["mapa_suavizado"], cmap="hot", alpha=0.5,
                       extent=(0, width, height, 0),
                       interpolation="bilinear")
        cmap = plt.get_cmap("tab20")
        labels = c["etiquetas_dbscan"]
        if pts is not None and labels is not None:
            for cl in c["clusters"]:
                members = pts[labels == cl["id"] - 1]
                ax1.add_patch(_cluster_ellipse(
                    members, cmap((cl["id"] - 1) % 20)))
        ax1.set_xlim(0, width)
        ax1.set_ylim(height, 0)
        ax1.set_title(f"{title}: densidad ({d['clasificacion']}) y "
                      "grupos", fontsize=10)
        ax1.axis("off")

        # Panel 3: partículas por grupo de DBSCAN
        ax3 = fig.add_subplot(grid[row, 1])
        ax3.imshow(image, cmap="gray")
        if pts is not None and labels is not None:
            alone = labels == -1
            ax3.plot(pts[alone, 1], pts[alone, 0], "o", color="lightgray",
                     markersize=4, mec="black", mew=0.4)
            for cl in c["clusters"]:
                sel = labels == cl["id"] - 1
                ax3.plot(pts[sel, 1], pts[sel, 0], "o",
                         color=cmap((cl["id"] - 1) % 20), markersize=4,
                         mec="black", mew=0.4)
            sub = (f"{c['n_clusters_dbscan']} grupo(s), "
                   f"{c['n_particulas_aisladas']} aislada(s)")
        else:
            sub = c["hopkins_clasificacion"]
        ax3.set_title(f"{title}: DBSCAN — {sub}", fontsize=10)
        ax3.axis("off")

        # Panel 4: partículas por franja y regresión
        ax4 = fig.add_subplot(grid[row, 2])
        if g["densidad"] is None:
            ax4.text(0.5, 0.5, "insuficiente\n(menos de 10 partículas)",
                     ha="center", va="center", transform=ax4.transAxes)
            ax4.axis("off")
        else:
            for direction, color in (("vertical", "tab:blue"),
                                     ("horizontal", "tab:orange")):
                fit = g["densidad"][direction]
                x = fit["posiciones_px"]
                ax4.plot(x, fit["valores"], "o", color=color,
                         label=f"{direction} (p="
                               + (f"{fit['p_value']:.1e}"
                                  if fit["p_value"] is not None else "-")
                               + ")")
                if fit["p_value"] is not None:
                    ax4.plot(x, fit["intercepto"]
                             + fit["pendiente_por_100px"] / 100 * x,
                             color=color,
                             linestyle="-" if fit["significativo"]
                             else ":")
            ax4.set_xlabel("Posición de la franja (px): fila ↓ / "
                           "columna →")
            ax4.set_ylabel("Partículas por franja")
            ax4.legend(fontsize=8)
        ax4.set_title(f"{title}: gradiente de densidad (continua = "
                      "significativo)", fontsize=10)

    fig.suptitle(f"Análisis espacial: {path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# Claves que debe traer el diccionario de pipeline_compositional
_COMPOSITIONAL_KEYS: tuple[str, ...] = (
    "original", "processed", "mask", "props", "filepath", "background",
)


def pipeline_spatial(
    compositional_dict: dict,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Análisis espacial y de distribución completo.

    Secuencia: compute_rdf → compute_density_map → detect_clustering
    → detect_spatial_gradients → show_spatial_results. Se usa el mismo
    fondo que eligió el Módulo A.

    Args:
        compositional_dict: Diccionario de pipeline_compositional
            (Módulo B).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        El diccionario de entrada más 'rdf', 'density_map',
        'clustering' y 'spatial_gradients'.

    Raises:
        ValueError: Si falta alguna clave en compositional_dict.
    """
    missing = [k for k in _COMPOSITIONAL_KEYS
               if k not in compositional_dict]
    if missing:
        raise ValueError(
            f"Al diccionario de entrada le falta(n): {', '.join(missing)}."
            " Debe ser el resultado de pipeline_compositional()."
        )
    props = compositional_dict["props"]
    mask = compositional_dict["mask"]
    bg = compositional_dict["background"]

    print("--- C1. RDF ---")
    rdf = compute_rdf(props, mask, background=bg)
    print("--- C2. Mapa de densidad ---")
    density = compute_density_map(props, mask, background=bg)
    print("--- C3. Agrupamiento ---")
    clustering = detect_clustering(props, mask, bg)
    print("--- C4. Gradientes ---")
    gradients = detect_spatial_gradients(
        props, mask, compositional_dict["processed"], bg)
    print("--- C5. Figura ---")
    show_spatial_results(
        compositional_dict["original"], mask, rdf, density, clustering,
        gradients, compositional_dict["filepath"], props=props,
        background=bg, output_dir=output_dir, show=show,
    )

    print(f"\nResumen espacial: "
          f"{Path(compositional_dict['filepath']).name}")
    for name in rdf:
        print("  " + _phase_summary(name, rdf, density, clustering,
                                    gradients).replace("\n", " |"))

    return {
        **compositional_dict,
        "rdf": rdf,
        "density_map": density,
        "clustering": clustering,
        "spatial_gradients": gradients,
    }
