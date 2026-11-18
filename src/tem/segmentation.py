"""Segmentación automática de fases en imágenes TEM.

Etapa 2 del módulo TEM del proyecto "Analizador de Imágenes de
Microscopía". El módulo detecta y separa fases por intensidad; no
asigna elementos químicos.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage, signal, stats
from skimage import feature, filters, measure, morphology, segmentation
from sklearn.mixture import GaussianMixture

from .loader import DEFAULT_OUTPUT_DIR

# --------------------------------------------------------------------
# Parámetros del Bloque 1 (detección de fases)
# --------------------------------------------------------------------

# Bins del histograma en el rango [0, 1]
HIST_BINS: int = 256
# Suavizado del histograma antes de buscar picos (en bins)
HIST_SMOOTH_SIGMA: float = 2.0
# Prominencia mínima de un pico, como fracción del máximo del
# histograma suavizado (se adapta a cada imagen)
PEAK_MIN_PROMINENCE: float = 0.05
# Distancia mínima entre picos (en bins)
PEAK_MIN_DISTANCE: int = 8
# Un pico es fase solo si su tramo tiene al menos esta fracción de
# píxeles
PEAK_MIN_MASS: float = 0.05
# Dos picos se fusionan si el valle entre ellos no baja de esta
# fracción del pico más bajo. Con 0.7 se fusionaba la partícula con el
# fondo en prueba13.dm3 (valle al 71 %); 0.75 la separa.
VALLEY_MAX_RATIO: float = 0.75
# Número mínimo de fases que se devuelve
MIN_PHASES: int = 2

# --------------------------------------------------------------------
# Parámetros del Bloque 2 (umbralización)
# --------------------------------------------------------------------

# Tamaño de bloque automático del umbral local: fracción del lado menor
ADAPTIVE_BLOCK_FRACTION: float = 0.10
# Píxeles al azar con los que se entrena el GMM (más es más lento)
GMM_MAX_SAMPLES: int = 200_000
# Semilla fija: la misma imagen da siempre el mismo resultado
RANDOM_SEED: int = 0
# Lado de los bloques para la medida de iluminación
ILLUMINATION_BLOCK: int = 64
# Si la medida de iluminación supera este valor se imprime una
# advertencia (no cambia el método: ver select_threshold_method)
ILLUMINATION_WARNING: float = 0.15
# Hasta este número de fases se usa multi-Otsu; por encima, GMM
OTSU_MAX_PHASES: int = 3
# Con 2 fases y una asimetría del histograma mayor que esta se usa el
# método del triángulo. En 22 imágenes de prueba, campo oscuro dio
# entre +1.75 y +13 y campo claro entre -1.81 y +0.83.
DARKFIELD_SKEW: float = 1.5

# --------------------------------------------------------------------
# Parámetros del Bloque 3 (separación y limpieza)
# --------------------------------------------------------------------

# Etiqueta de interfaces y líneas de corte del watershed
INTERFACE_LABEL: int = -1
# Tamaño mínimo automático de una región: fracción del total de
# píxeles (0.01 %: 419 px en una imagen de 2048 x 2048)
CLEAN_MIN_FRACTION: float = 0.0001
# Radio del disco de la apertura que suaviza bordes
CLEAN_OPENING_RADIUS: int = 1
# Detección de regiones que se tocan: cada región se erosiona hasta
# quedarse con la parte a más de esta fracción de su distancia máxima
# al borde. Un cuello más angosto que eso se corta. (Un disco fijo de
# radio 2 solo detectaba cuellos de ~5 px o menos.)
TOUCH_CORE_FRACTION: float = 0.5
# Distancia mínima entre marcadores del watershed (en píxeles)
WATERSHED_MIN_DISTANCE: int = 10
# Regiones de más de esta fracción de la imagen no se cortan con
# watershed: son zonas (p. ej. espesores en prueba7.dm3), no partículas
WATERSHED_MAX_REGION: float = 0.05

# --------------------------------------------------------------------
# Parámetros del Bloque 5 (visualización)
# --------------------------------------------------------------------

# Color de cada fase, de la más oscura a la más clara (las interfaces
# se dibujan en blanco)
PHASE_COLORS: tuple[str, ...] = (
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
    "tab:cyan", "tab:gray",
)
INTERFACE_COLOR: str = "white"


# ====================================================================
# BLOQUE 1 — Detección automática de fases
# ====================================================================


def _valleys_between(hist: np.ndarray, peaks: list[int]) -> list[int]:
    """Busca el mínimo del histograma entre cada par de picos.

    Args:
        hist: Histograma (suavizado).
        peaks: Índices de los picos, en orden ascendente.

    Returns:
        Índices de los valles (uno menos que la cantidad de picos).
    """
    return [
        left + int(np.argmin(hist[left:right + 1]))
        for left, right in zip(peaks, peaks[1:])
    ]


def _peaks_mass(hist: np.ndarray, peaks: list[int]) -> list[float]:
    """Fracción de píxeles en el tramo de cada pico.

    El tramo de un pico va del valle anterior al valle siguiente (o al
    borde del histograma).

    Args:
        hist: Histograma (suavizado).
        peaks: Índices de los picos, en orden ascendente.

    Returns:
        Lista con la fracción de píxeles de cada pico (suma 1).
    """
    total = float(hist.sum())
    if not peaks or total == 0.0:
        return []
    edges = [0, *_valleys_between(hist, peaks), len(hist)]
    return [
        float(hist[edges[i]:edges[i + 1]].sum()) / total
        for i in range(len(peaks))
    ]


def analyze_histogram(image: np.ndarray) -> dict:
    """Calcula el histograma y detecta sus picos y valles.

    Los picos se buscan sobre una copia suavizada del histograma, con
    una prominencia mínima proporcional a su altura máxima. Los bins
    extremos (0 y 255) se reemplazan por sus vecinos, porque la
    normalización por percentiles acumula ahí los píxeles recortados.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Diccionario con:
        - 'n_peaks': cantidad de picos.
        - 'peaks_positions': posición de cada pico en [0, 1].
        - 'valleys_positions': posición de cada valle en [0, 1].
        - 'histogram': histograma original (256 bins).
        - 'bin_edges': bordes de los bins.
        - 'histogram_smooth': histograma suavizado usado para buscar.
        - 'peaks_mass': fracción de píxeles de cada pico.
    """
    hist, bin_edges = np.histogram(image, bins=HIST_BINS, range=(0, 1))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bins extremos: se reemplazan para que no generen picos falsos
    work = hist.astype(np.float64)
    work[0], work[-1] = work[1], work[-2]
    smooth = ndimage.gaussian_filter1d(work, sigma=HIST_SMOOTH_SIGMA)

    # Prominencia adaptativa: fracción del máximo de esta imagen
    peaks, _ = signal.find_peaks(
        smooth,
        prominence=PEAK_MIN_PROMINENCE * smooth.max(),
        distance=PEAK_MIN_DISTANCE,
    )
    peaks_idx = [int(p) for p in peaks]
    valleys_idx = _valleys_between(smooth, peaks_idx)

    result = {
        "n_peaks": len(peaks_idx),
        "peaks_positions": [float(centers[p]) for p in peaks_idx],
        "valleys_positions": [float(centers[v]) for v in valleys_idx],
        "histogram": hist,
        "bin_edges": bin_edges,
        "histogram_smooth": smooth,
        "peaks_mass": _peaks_mass(smooth, peaks_idx),
    }

    posiciones = ", ".join(f"{p:.3f}" for p in result["peaks_positions"])
    print(f"Histograma: {result['n_peaks']} pico(s) detectado(s)")
    print(f"  Posiciones: [{posiciones}]")
    return result


def _shannon_entropy(hist: np.ndarray) -> float:
    """Entropía de Shannon de un histograma, en bits.

    Args:
        hist: Conteos por bin.

    Returns:
        Entropía en bits (0 si el histograma está vacío).
    """
    total = float(hist.sum())
    if total == 0.0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _merge_peaks(
    smooth: np.ndarray, peaks: list[int], centers: np.ndarray
) -> tuple[list[int], list[str]]:
    """Fusiona los picos que no representan una fase real.

    Repite hasta que todos los picos cumplan dos condiciones:
    1. Su tramo tiene al menos PEAK_MIN_MASS de los píxeles.
    2. El valle con cada vecino baja de VALLEY_MAX_RATIO veces el pico
       más bajo del par.
    En cada vuelta elimina el peor pico: primero el de menor masa y,
    si todos tienen masa suficiente, el más bajo del valle menos
    profundo.

    Args:
        smooth: Histograma suavizado.
        peaks: Índices de los picos, en orden ascendente.
        centers: Centro de cada bin, para informar posiciones.

    Returns:
        Tupla (picos que quedan, mensajes con cada fusión).
    """
    kept = list(peaks)
    messages: list[str] = []
    while len(kept) > 1:
        masses = _peaks_mass(smooth, kept)
        worst = int(np.argmin(masses))
        if masses[worst] < PEAK_MIN_MASS:
            messages.append(
                f"pico {centers[kept[worst]]:.3f} fusionado: solo "
                f"{masses[worst]:.1%} de los píxeles (< "
                f"{PEAK_MIN_MASS:.0%})"
            )
            kept.pop(worst)
            continue

        valleys = _valleys_between(smooth, kept)
        ratios = [
            smooth[v] / min(smooth[a], smooth[b])
            for v, a, b in zip(valleys, kept, kept[1:])
        ]
        shallow = int(np.argmax(ratios))
        if ratios[shallow] <= VALLEY_MAX_RATIO:
            break
        # Se elimina el más bajo de los dos picos del valle
        left, right = kept[shallow], kept[shallow + 1]
        drop = shallow if smooth[left] < smooth[right] else shallow + 1
        messages.append(
            f"pico {centers[kept[drop]]:.3f} fusionado: el valle entre "
            f"{centers[left]:.3f} y {centers[right]:.3f} queda al "
            f"{ratios[shallow]:.0%} del pico más bajo (> "
            f"{VALLEY_MAX_RATIO:.0%})"
        )
        kept.pop(drop)
    return kept, messages


def detect_n_phases(image: np.ndarray) -> int:
    """Estima cuántas fases de intensidad tiene la imagen.

    Detecta los picos del histograma y fusiona los que no son fases
    reales (poca masa o valle poco profundo). La entropía de Shannon
    del histograma se informa como dato, pero no decide.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Número de fases detectado (mínimo 2).
    """
    info = analyze_histogram(image)
    bin_edges = info["bin_edges"]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smooth = info["histogram_smooth"]
    peaks = [int(np.argmin(np.abs(centers - p)))
             for p in info["peaks_positions"]]

    entropy = _shannon_entropy(info["histogram"])
    max_entropy = np.log2(HIST_BINS)
    print(
        f"  Entropía del histograma: {entropy:.2f} bits "
        f"({entropy / max_entropy:.0%} del máximo, dato informativo)"
    )

    if info["n_peaks"] <= 1:
        print(
            f"  Decisión: {info['n_peaks']} pico(s) → se asume el caso "
            f"mínimo de {MIN_PHASES} fases."
        )
        return MIN_PHASES

    kept, messages = _merge_peaks(smooth, peaks, centers)
    for message in messages:
        print(f"  {message}")

    n_phases = max(MIN_PHASES, len(kept))
    posiciones = ", ".join(f"{centers[p]:.3f}" for p in kept)
    print(
        f"  Decisión: {info['n_peaks']} pico(s) detectado(s), "
        f"{len(kept)} tras fusionar [{posiciones}] → {n_phases} fases."
    )
    return n_phases


# ====================================================================
# BLOQUE 2 — Métodos de umbralización
# ====================================================================


def threshold_otsu_multi(image: np.ndarray, n_phases: int) -> np.ndarray:
    """Umbrales de Otsu multinivel para separar n_phases fases.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        n_phases: Número de fases (2 o más).

    Returns:
        Array con n_phases - 1 umbrales en orden ascendente.
    """
    thresholds = filters.threshold_multiotsu(image, classes=n_phases)
    valores = ", ".join(f"{t:.3f}" for t in thresholds)
    print(f"Otsu multinivel ({n_phases} fases): umbrales [{valores}]")
    return thresholds


def threshold_adaptive_local(
    image: np.ndarray, block_size: int | str = "auto"
) -> np.ndarray:
    """Umbral local: cada píxel se compara con su entorno.

    Sirve cuando la iluminación no es uniforme. Separa siempre en 2
    fases (oscura y clara).

    Args:
        image: Imagen 2D normalizada a [0, 1].
        block_size: Lado del entorno en píxeles (impar) o 'auto' para
            usar el 10 % del lado menor, redondeado al impar más
            cercano.

    Returns:
        Máscara booleana: True donde el píxel es más claro que su
        entorno.

    Raises:
        ValueError: Si block_size no es 'auto' ni un entero impar >= 3.
    """
    if block_size == "auto":
        size = ADAPTIVE_BLOCK_FRACTION * min(image.shape)
        # Impar más cercano (como mínimo 3)
        block = max(3, 2 * int(round((size - 1) / 2)) + 1)
    elif isinstance(block_size, int) and block_size >= 3 and block_size % 2:
        block = block_size
    else:
        raise ValueError(
            f"block_size debe ser 'auto' o un entero impar >= 3; "
            f"llegó {block_size!r}."
        )

    local = filters.threshold_local(image, block_size=block)
    print(f"Umbral local adaptativo: block_size = {block} px")
    return image > local


def threshold_gmm(image: np.ndarray, n_phases: int) -> np.ndarray:
    """Umbrales a partir de un modelo de mezcla de gaussianas (GMM).

    Ajusta n_phases gaussianas a las intensidades, las ordena por media
    y pone cada umbral en el punto medio entre medias vecinas. Se
    entrena con una muestra al azar de GMM_MAX_SAMPLES píxeles.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        n_phases: Número de fases (componentes gaussianas).

    Returns:
        Array con n_phases - 1 umbrales en orden ascendente.
    """
    pixels = image.ravel()
    rng = np.random.default_rng(RANDOM_SEED)
    if pixels.size > GMM_MAX_SAMPLES:
        pixels = rng.choice(pixels, GMM_MAX_SAMPLES, replace=False)

    gmm = GaussianMixture(n_components=n_phases, random_state=RANDOM_SEED)
    gmm.fit(pixels.reshape(-1, 1))

    # Componentes ordenadas por media ascendente
    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])
    thresholds = (means[:-1] + means[1:]) / 2

    print(f"GMM ({n_phases} componentes, {pixels.size} píxeles):")
    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f"  componente {i}: media {mean:.3f}, desvío {std:.3f}")
    valores = ", ".join(f"{t:.3f}" for t in thresholds)
    print(f"  umbrales (puntos medios): [{valores}]")
    return thresholds


def illumination_uniformity(image: np.ndarray) -> float:
    """Desvío estándar de las medias locales en bloques.

    Valores altos pueden indicar iluminación no uniforme, pero también
    objetos grandes: no distingue una cosa de la otra.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Desvío estándar de las medias de bloques de
        ILLUMINATION_BLOCK x ILLUMINATION_BLOCK píxeles.
    """
    b = ILLUMINATION_BLOCK
    h, w = (image.shape[0] // b) * b, (image.shape[1] // b) * b
    # Imagen más chica que un bloque: se usa la imagen entera
    if h == 0 or w == 0:
        return 0.0
    blocks = image[:h, :w].reshape(h // b, b, w // b, b)
    return float(blocks.mean(axis=(1, 3)).std())


def threshold_triangle_method(image: np.ndarray) -> np.ndarray:
    """Umbral por el método del triángulo (2 fases).

    Pensado para histogramas con un pico dominante y una cola larga,
    como en campo oscuro: pone el corte donde termina el pico del
    fondo, así no lo arrastran unas pocas partículas muy brillantes.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Array con un umbral.
    """
    threshold = np.array([filters.threshold_triangle(image)])
    print(f"Método del triángulo: umbral [{threshold[0]:.3f}]")
    return threshold


def histogram_skewness(image: np.ndarray) -> float:
    """Asimetría (skewness) de las intensidades de la imagen.

    Positiva y grande: fondo oscuro dominante con una cola de píxeles
    brillantes (campo oscuro). Cerca de 0 o negativa: campo claro. Se
    calcula sobre una muestra de GMM_MAX_SAMPLES píxeles.

    Args:
        image: Imagen 2D normalizada a [0, 1].

    Returns:
        Asimetría de la distribución (0 si la imagen es constante).
    """
    pixels = image.ravel()
    if pixels.size > GMM_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED)
        pixels = rng.choice(pixels, GMM_MAX_SAMPLES, replace=False)
    if np.ptp(pixels) == 0:
        return 0.0
    return float(stats.skew(pixels))


def select_threshold_method(
    image: np.ndarray, n_phases: int
) -> tuple[str, np.ndarray]:
    """Elige el método de umbralización y calcula los umbrales.

    Con 2 fases y un histograma muy asimétrico (asimetría mayor que
    DARKFIELD_SKEW, típico de campo oscuro) usa el método del
    triángulo. Si no, hasta OTSU_MAX_PHASES fases usa Otsu multinivel
    y con más fases usa GMM. El umbral adaptativo no se elige
    automáticamente, porque la medida de iluminación no distingue una
    iluminación despareja de objetos grandes; si la medida es alta se
    imprime una advertencia para revisarlo y, si corresponde, pedirlo
    con method='adaptive' en pipeline_segmentation.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        n_phases: Número de fases (de detect_n_phases).

    Returns:
        Tupla (nombre del método: 'triangle', 'otsu' o 'gmm',
        umbrales).
    """
    skew = histogram_skewness(image)
    print(f"Asimetría del histograma: {skew:.2f}")
    if skew > DARKFIELD_SKEW:
        if n_phases == 2:
            print(
                f"Método elegido: 'triangle' porque la asimetría supera "
                f"{DARKFIELD_SKEW} (fondo oscuro dominante, típico de "
                "campo oscuro) y hay 2 fases."
            )
            return "triangle", threshold_triangle_method(image)
        print(
            f"  La asimetría supera {DARKFIELD_SKEW} pero hay "
            f"{n_phases} fases: el triángulo solo separa 2."
        )

    uniformity = illumination_uniformity(image)
    print(
        f"Iluminación: desvío de medias en bloques de "
        f"{ILLUMINATION_BLOCK} px = {uniformity:.3f}"
    )
    if uniformity > ILLUMINATION_WARNING:
        print(
            f"  Advertencia: supera {ILLUMINATION_WARNING}. Puede ser "
            "iluminación no uniforme u objetos grandes; si es "
            "iluminación, probá method='adaptive'."
        )

    if n_phases <= OTSU_MAX_PHASES:
        print(
            f"Método elegido: 'otsu' porque hay {n_phases} fases "
            f"(<= {OTSU_MAX_PHASES}): Otsu multinivel es exacto y rápido."
        )
        return "otsu", threshold_otsu_multi(image, n_phases)

    print(
        f"Método elegido: 'gmm' porque hay {n_phases} fases "
        f"(> {OTSU_MAX_PHASES}): el costo de Otsu multinivel crece muy "
        "rápido con las fases y el GMM escala mejor."
    )
    return "gmm", threshold_gmm(image, n_phases)


# ====================================================================
# BLOQUE 3 — Separación y limpieza de regiones
# ====================================================================


def _phase_labels(mask: np.ndarray) -> list[int]:
    """Etiquetas de fase presentes en la máscara (ignora las -1).

    Args:
        mask: Máscara de etiquetas.

    Returns:
        Lista ordenada de etiquetas >= 0.
    """
    return [int(v) for v in np.unique(mask) if v >= 0]


def apply_segmentation(
    image: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Asigna a cada píxel la etiqueta de su fase.

    Con umbrales, la etiqueta es el tramo en que cae la intensidad:
    0 por debajo del primer umbral, 1 entre el primero y el segundo,
    etc. Si thresholds es una máscara del mismo shape que la imagen
    (caso del umbral adaptativo), se usa directamente: booleana pasa
    a 0 y 1, y con enteros se conservan las etiquetas.

    Args:
        image: Imagen 2D normalizada a [0, 1].
        thresholds: Umbrales en [0, 1] o máscara del shape de image.

    Returns:
        Máscara de enteros (int32) con el shape de image.
    """
    values = np.asarray(thresholds)
    if values.shape == image.shape:
        mask = values.astype(np.int32)
    else:
        cuts = np.sort(values.ravel())
        mask = np.digitize(image, cuts).astype(np.int32)

    print("Segmentación aplicada:")
    for k in _phase_labels(mask):
        print(f"  fase {k}: {np.mean(mask == k):.1%} de la imagen")
    return mask


def clean_mask(
    mask: np.ndarray, min_size: int | str = "auto"
) -> np.ndarray:
    """Elimina regiones chicas, rellena huecos y suaviza bordes.

    Para cada fase: elimina las regiones de menos de min_size píxeles,
    rellena los huecos de menos de min_size píxeles y aplica una
    apertura con un disco de radio CLEAN_OPENING_RADIUS. Los píxeles
    que quedan sin fase pasan a la fase que los reclamó al rellenar
    huecos o, si ninguna lo hizo, a la fase más cercana. Si una fase
    se borraría entera, se deja sin limpiar (el número de fases no
    cambia). Las etiquetas -1 de la entrada se conservan.

    Args:
        mask: Máscara de etiquetas (de apply_segmentation).
        min_size: Tamaño mínimo en píxeles o 'auto' para usar
            CLEAN_MIN_FRACTION del total de píxeles.

    Returns:
        Máscara limpia (int32) con las mismas etiquetas.

    Raises:
        ValueError: Si min_size no es 'auto' ni un entero >= 1.
    """
    if min_size == "auto":
        size = max(1, int(round(CLEAN_MIN_FRACTION * mask.size)))
    elif isinstance(min_size, int) and min_size >= 1:
        size = min_size
    else:
        raise ValueError(
            f"min_size debe ser 'auto' o un entero >= 1; llegó "
            f"{min_size!r}."
        )

    footprint = morphology.disk(CLEAN_OPENING_RADIUS)
    phases = _phase_labels(mask)
    cleaned: dict[int, np.ndarray] = {}
    print(f"Limpieza (tamaño mínimo {size} px):")
    for k in phases:
        region = mask == k
        labels = measure.label(region, connectivity=1)
        sizes = np.bincount(labels.ravel())[1:]  # el 0 no es la fase
        n_small = int(np.sum(sizes < size))
        if n_small == sizes.size:
            # Todas sus regiones son chicas: se conserva la fase
            print(
                f"  fase {k}: sus {sizes.size} región(es) son menores "
                f"que {size} px; se deja sin limpiar para no perderla"
            )
            cleaned[k] = region
            continue
        # max_size es inclusivo: se eliminan las de tamaño < size
        region = morphology.remove_small_objects(
            region, max_size=size - 1
        )
        region = morphology.remove_small_holes(region, max_size=size - 1)
        region = morphology.opening(region, footprint)
        cleaned[k] = region
        print(
            f"  fase {k}: {n_small} región(es) chica(s) eliminada(s) "
            f"de {sizes.size}"
        )

    # 1) Cada píxel conserva su fase si la limpieza no se la quitó
    result = np.full(mask.shape, INTERFACE_LABEL, dtype=np.int32)
    for k in phases:
        result[(mask == k) & cleaned[k]] = k
    # 2) Píxeles sin fase que otra fase reclamó al rellenar huecos
    for k in phases:
        free = (result == INTERFACE_LABEL) & (mask >= 0)
        result[free & cleaned[k]] = k
    # 3) El resto toma la fase del píxel asignado más cercano
    unassigned = (result == INTERFACE_LABEL) & (mask >= 0)
    if unassigned.any():
        nearest = ndimage.distance_transform_edt(
            result < 0, return_distances=False, return_indices=True
        )
        result[unassigned] = result[tuple(nearest)][unassigned]

    # Seguridad: ninguna fase debe desaparecer
    for k in phases:
        if not np.any(result == k):
            result[mask == k] = k
            print(f"  fase {k}: desaparecía al limpiar; se restauró")
    return result


def separate_touching_regions(mask: np.ndarray) -> np.ndarray:
    """Separa con watershed las regiones de una fase que se tocan.

    No procesa la fase mayoritaria (fondo o matriz). En las demás,
    erosiona cada región en proporción a su grosor: se queda con los
    píxeles a más de TOUCH_CORE_FRACTION de la distancia máxima al
    borde de esa región. Si quedan dos o más núcleos, había regiones
    unidas por un cuello. A esas regiones se les aplica watershed sobre
    el mapa de
    distancias, con los máximos locales como marcadores. Las regiones
    de más de WATERSHED_MAX_REGION del área no se cortan (son zonas,
    no partículas). Las líneas de corte se marcan con INTERFACE_LABEL
    (-1).

    Args:
        mask: Máscara de etiquetas (de clean_mask).

    Returns:
        Máscara (int32) con las líneas de corte en -1, o una copia sin
        cambios si no hay regiones que se toquen.
    """
    out = mask.astype(np.int32, copy=True)
    phases = _phase_labels(mask)
    if len(phases) < 2:
        print("Watershed: no se aplicó (hay una sola fase)")
        return out
    areas = {k: int(np.sum(mask == k)) for k in phases}
    majority = max(areas, key=areas.get)

    applied: list[int] = []
    for k in phases:
        if k == majority:
            continue
        region = mask == k
        labels = measure.label(region, connectivity=2)
        if labels.max() == 0:
            continue

        # Erosión adaptativa: núcleo de cada región según su grosor
        depth = ndimage.distance_transform_edt(region)
        max_depth = np.zeros(labels.max() + 1)
        max_depth[1:] = ndimage.maximum(
            depth, labels, np.arange(1, labels.max() + 1)
        )
        eroded = depth > TOUCH_CORE_FRACTION * max_depth[labels]
        eroded_labels = measure.label(eroded, connectivity=2)

        # Núcleos que caen dentro de cada región original
        pairs = np.unique(
            np.stack([labels[eroded], eroded_labels[eroded]]), axis=1
        )
        pieces = np.bincount(pairs[0], minlength=labels.max() + 1)
        touching_ids = np.flatnonzero(pieces >= 2)
        # Las regiones muy grandes son zonas, no partículas: no se cortan
        sizes = np.bincount(labels.ravel(), minlength=labels.max() + 1)
        large = sizes[touching_ids] > WATERSHED_MAX_REGION * mask.size
        if large.any():
            print(
                f"Fase {k}: {int(large.sum())} región(es) de más del "
                f"{WATERSHED_MAX_REGION:.0%} del área se dejan sin cortar"
            )
        touching_ids = touching_ids[~large]
        if touching_ids.size == 0:
            continue

        target = np.isin(labels, touching_ids)
        distance = ndimage.distance_transform_edt(target)
        peaks = feature.peak_local_max(
            distance,
            min_distance=WATERSHED_MIN_DISTANCE,
            labels=labels * target,
            exclude_border=False,
        )
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        basins = segmentation.watershed(
            -distance, markers, mask=target, watershed_line=True
        )
        out[target & (basins == 0)] = INTERFACE_LABEL

        n_before = int(labels.max())
        n_after = int(measure.label(out == k, connectivity=1).max())
        applied.append(k)
        print(
            f"Watershed en fase {k}: {touching_ids.size} región(es) "
            f"tocándose; regiones de la fase {n_before} → {n_after}"
        )

    if not applied:
        print(
            f"Watershed: no se aplicó (no hay regiones tocándose; la "
            f"fase mayoritaria {majority} no se procesa)"
        )
    else:
        print(
            f"Watershed aplicado en {len(applied)} fase(s): {applied} "
            f"(fase mayoritaria {majority} excluida)"
        )
    return out


def detect_interfaces(mask: np.ndarray) -> np.ndarray:
    """Marca con -1 los píxeles de borde entre fases distintas.

    Un píxel es interfaz si alguno de sus 8 vecinos es de otra fase
    (las etiquetas -1 no cuentan como fase). Así se marca un píxel a
    cada lado del borde: la interfaz tiene 2 píxeles de ancho.

    Args:
        mask: Máscara de etiquetas (puede traer -1 del watershed).

    Returns:
        Máscara (int32): -1 = interfaz o línea de corte, 0, 1, ... =
        fases.
    """
    valid = mask >= 0
    # Las -1 no deben ganar ni en el mínimo ni en el máximo vecino
    para_min = np.where(valid, mask, mask.max() + 1)
    para_max = np.where(valid, mask, -1)
    vecino_min = ndimage.minimum_filter(para_min, size=3)
    vecino_max = ndimage.maximum_filter(para_max, size=3)
    interface = valid & ((vecino_min < mask) | (vecino_max > mask))

    lines_pct = float(np.mean(mask == INTERFACE_LABEL))
    out = mask.astype(np.int32, copy=True)
    out[interface] = INTERFACE_LABEL
    print(
        f"Interfaces: {np.mean(interface):.2%} del área "
        f"(+ {lines_pct:.2%} de líneas de watershed)"
    )
    return out


# ====================================================================
# BLOQUE 4 — Estadísticas por fase
# ====================================================================


def compute_phase_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    """Calcula área, intensidad y regiones de cada fase.

    Los porcentajes de las fases se calculan sobre el área sin
    interfaces, así que suman 100 %. El porcentaje de interfaces
    (interfaces entre fases y líneas de corte del watershed, ambas con
    etiqueta -1) es sobre el área total. Las regiones se cuentan con
    vecindad de 4 píxeles, para que las líneas de corte de 1 píxel
    separen siempre las regiones.

    Las regiones de menos de CLEAN_MIN_FRACTION del total de píxeles
    (el mismo tamaño mínimo de clean_mask) son fragmentos que deja el
    marcado de interfaces: no cuentan en n_regiones ni en los tamaños,
    se informan en n_fragmentos y su área sí se suma a la de la fase.

    Args:
        image: Imagen 2D (en el pipeline, la preprocesada en [0, 1]).
        mask: Máscara final: -1 = interfaz, 0, 1, ... = fases.

    Returns:
        Diccionario con una clave 'fase_N' por fase (area_px,
        area_pct, intensidad_media, intensidad_std, n_regiones,
        tamanio_medio_px, tamanio_min_px, tamanio_max_px,
        n_fragmentos) y la clave
        'interfaces' (area_px, area_pct). Todos los valores son int o
        float de Python.

    Raises:
        ValueError: Si image y mask no tienen el mismo shape.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image {image.shape} y mask {mask.shape} deben tener el "
            "mismo shape."
        )

    interface = mask == INTERFACE_LABEL
    total = int(mask.size)
    phase_total = total - int(interface.sum())
    min_region = max(1, int(round(CLEAN_MIN_FRACTION * total)))

    stats: dict[str, dict[str, int | float]] = {}
    for k in _phase_labels(mask):
        region = mask == k
        area = int(region.sum())
        values = image[region]
        labels = measure.label(region, connectivity=1)
        all_sizes = np.bincount(labels.ravel())[1:]
        # Fragmentos: se informan aparte y no cuentan como regiones
        is_fragment = all_sizes < min_region
        sizes = all_sizes[~is_fragment]
        stats[f"fase_{k}"] = {
            "area_px": area,
            "area_pct": 100.0 * area / phase_total if phase_total else 0.0,
            "intensidad_media": float(values.mean()),
            "intensidad_std": float(values.std()),
            "n_regiones": int(sizes.size),
            "tamanio_medio_px": float(sizes.mean()) if sizes.size else 0.0,
            "tamanio_min_px": int(sizes.min()) if sizes.size else 0,
            "tamanio_max_px": int(sizes.max()) if sizes.size else 0,
            "n_fragmentos": int(is_fragment.sum()),
        }
    stats["interfaces"] = {
        "area_px": int(interface.sum()),
        "area_pct": 100.0 * float(interface.sum()) / total,
    }

    print(f"Estadísticas por fase (regiones >= {min_region} px):")
    print(
        f"  {'fase':8} {'área %':>7} {'int. media':>10} {'int. std':>8} "
        f"{'regiones':>8} {'tam. medio':>10} {'tam. mín':>8} "
        f"{'tam. máx':>9} {'fragm.':>6}"
    )
    for name, s in stats.items():
        if name == "interfaces":
            continue
        print(
            f"  {name:8} {s['area_pct']:>7.2f} "
            f"{s['intensidad_media']:>10.3f} {s['intensidad_std']:>8.3f} "
            f"{s['n_regiones']:>8} {s['tamanio_medio_px']:>10.1f} "
            f"{s['tamanio_min_px']:>8} {s['tamanio_max_px']:>9} "
            f"{s['n_fragmentos']:>6}"
        )
    print(
        f"  interfaces: {stats['interfaces']['area_px']} px "
        f"({stats['interfaces']['area_pct']:.2f} % del área total)"
    )
    return stats


# ====================================================================
# BLOQUE 5 — Visualización
# ====================================================================


def show_segmentation_results(
    original: np.ndarray,
    mask: np.ndarray,
    stats: dict,
    filepath: str | Path,
    thresholds: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la figura completa de la segmentación.

    Paneles: (1) imagen original; (2) máscara coloreada, con las
    interfaces en blanco, barra de colores y leyenda; (3) histograma de
    la original con el histograma de cada fase en su color y, si se
    pasan, los umbrales; (4 en adelante) una imagen por fase, con la
    original solo donde está esa fase, sobre fondo negro. La grilla se
    adapta al número de fases. Se guarda como
    '<nombre>_segmentation.png'.

    Args:
        original: Imagen original 2D normalizada a [0, 1].
        mask: Máscara final: -1 = interfaz, 0, 1, ... = fases.
        stats: Resultado de compute_phase_stats.
        filepath: Ruta del archivo de origen (para título y nombre).
        thresholds: Umbrales usados, para marcarlos en el histograma.
            None si no hay (por ejemplo, con el umbral adaptativo).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_segmentation.png"

    phases = _phase_labels(mask)
    n = len(phases)
    colors = [PHASE_COLORS[k % len(PHASE_COLORS)] for k in phases]

    # Grilla dinámica: 3 paneles fijos + 1 por fase
    total = 3 + n
    cols = 3 if total <= 6 else 4
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows))
    axes = axes.ravel()

    # Panel 1: original
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    # Panel 2: máscara coloreada (índice 0 = interfaz, 1.. = fases)
    index = np.full(mask.shape, 0, dtype=np.int32)
    for i, k in enumerate(phases, start=1):
        index[mask == k] = i
    cmap = ListedColormap([INTERFACE_COLOR, *colors])
    shown = axes[1].imshow(index, cmap=cmap, vmin=-0.5, vmax=n + 0.5,
                           interpolation="nearest")
    colorbar = fig.colorbar(shown, ax=axes[1], ticks=range(n + 1),
                            fraction=0.046, pad=0.04)
    colorbar.ax.set_yticklabels(
        ["interfaz", *(f"fase {k}" for k in phases)]
    )
    axes[1].legend(
        handles=[
            Patch(facecolor=c, edgecolor="black",
                  label=f"fase {k}: {stats[f'fase_{k}']['area_pct']:.1f} %")
            for k, c in zip(phases, colors)
        ] + [Patch(facecolor=INTERFACE_COLOR, edgecolor="black",
                   label=f"interfaz: "
                         f"{stats['interfaces']['area_pct']:.1f} %")],
        loc="lower right", fontsize=8,
    )
    axes[1].set_title("Máscara de fases")
    axes[1].axis("off")

    # Panel 3: histograma total, por fase y umbrales
    bins = np.linspace(0, 1, 257)
    axes[2].hist(original.ravel(), bins=bins, color="lightgray",
                 label="toda la imagen")
    for k, color in zip(phases, colors):
        axes[2].hist(original[mask == k], bins=bins, color=color,
                     alpha=0.5, label=f"fase {k}")
    if thresholds is not None:
        cuts = np.sort(np.ravel(thresholds))
        for i, cut in enumerate(cuts):
            # El umbral i separa la fase i de la i+1: color de la i+1
            color = PHASE_COLORS[(i + 1) % len(PHASE_COLORS)]
            axes[2].axvline(cut, color=color, linestyle="--",
                            linewidth=2,
                            label=f"umbral {cut:.3f}")
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Intensidad normalizada")
    axes[2].set_ylabel("Cantidad de píxeles")
    axes[2].set_title("Histograma de la original por fase")
    axes[2].legend(fontsize=8)

    # Paneles 4..: cada fase sobre fondo negro
    for ax, k in zip(axes[3:], phases):
        ax.imshow(np.where(mask == k, original, 0.0), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_title(
            f"Fase {k} — {stats[f'fase_{k}']['area_pct']:.2f}% del área"
        )
        ax.axis("off")
    for ax in axes[3 + n:]:
        ax.axis("off")

    fig.suptitle(f"Segmentación TEM: {path.name} — {n} fases detectadas",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    # Se cierra para no acumular figuras abiertas en memoria
    plt.close(fig)
    return out_path


# ====================================================================
# BLOQUE 6 — Pipeline principal
# ====================================================================

# Métodos de umbralización que acepta pipeline_segmentation
SEGMENTATION_METHODS: tuple[str, ...] = (
    "auto", "otsu", "gmm", "triangle", "adaptive",
)
# Claves que debe traer el diccionario de pipeline_tem
_PREPROCESSED_KEYS: tuple[str, ...] = (
    "original", "processed", "filepath", "method_used",
)


def pipeline_segmentation(
    preprocessed_dict: dict,
    method: str = "auto",
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Segmenta en fases una imagen ya preprocesada por pipeline_tem.

    Secuencia: detect_n_phases → umbrales (automático o el método
    pedido) → apply_segmentation → clean_mask →
    separate_touching_regions → detect_interfaces →
    compute_phase_stats → show_segmentation_results.

    Args:
        preprocessed_dict: Diccionario de pipeline_tem, con las claves
            'original', 'processed', 'filepath' y 'method_used'.
        method: 'auto', 'otsu', 'gmm', 'triangle' (campo oscuro) o
            'adaptive' (umbral local). 'triangle' y 'adaptive' siempre
            dan 2 fases.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Diccionario con 'original', 'processed', 'mask', 'n_phases'
        (fases en la máscara final), 'threshold_method' (el método
        aplicado, nunca 'auto'), 'stats', 'filepath', 'thresholds'
        (umbrales usados; None con 'adaptive') y 'lineas_watershed'
        (máscara booleana de las líneas de corte del watershed: en
        'mask' tienen -1 igual que las interfaces y no se distinguen).

    Raises:
        ValueError: Si falta alguna clave en preprocessed_dict o el
            método no es válido.
    """
    missing = [k for k in _PREPROCESSED_KEYS if k not in preprocessed_dict]
    if missing:
        raise ValueError(
            f"Al diccionario de entrada le falta(n): {', '.join(missing)}. "
            "Debe ser el resultado de pipeline_tem()."
        )
    if method not in SEGMENTATION_METHODS:
        raise ValueError(
            f"Método '{method}' no válido. Opciones: "
            f"{', '.join(SEGMENTATION_METHODS)}."
        )

    original = preprocessed_dict["original"]
    processed = preprocessed_dict["processed"]
    filepath = preprocessed_dict["filepath"]

    print("--- 1. Detección de fases ---")
    n_detected = detect_n_phases(processed)

    print("--- 2. Umbralización ---")
    thresholds: np.ndarray | None
    if method == "auto":
        method_used, thresholds = select_threshold_method(
            processed, n_detected
        )
        segmentation_input = thresholds
    elif method == "triangle":
        method_used = "triangle"
        if n_detected > 2:
            print(
                f"  Aviso: se detectaron {n_detected} fases, pero el "
                "método del triángulo solo separa 2."
            )
        thresholds = threshold_triangle_method(processed)
        segmentation_input = thresholds
    elif method == "otsu":
        method_used = "otsu"
        thresholds = threshold_otsu_multi(processed, n_detected)
        segmentation_input = thresholds
    elif method == "gmm":
        method_used = "gmm"
        thresholds = threshold_gmm(processed, n_detected)
        segmentation_input = thresholds
    else:
        method_used = "adaptive"
        if n_detected > 2:
            print(
                f"  Aviso: se detectaron {n_detected} fases, pero el "
                "umbral local solo separa 2."
            )
        thresholds = None
        segmentation_input = threshold_adaptive_local(processed)

    print("--- 3. Separación y limpieza ---")
    mask = apply_segmentation(processed, segmentation_input)
    mask = clean_mask(mask)
    mask = separate_touching_regions(mask)
    # Las líneas de corte se guardan aparte: después de marcar las
    # interfaces, ambas son -1 y ya no se pueden separar
    watershed_lines = mask == INTERFACE_LABEL
    mask = detect_interfaces(mask)

    print("--- 4. Estadísticas ---")
    stats = compute_phase_stats(processed, mask)

    print("--- 5. Figura ---")
    show_segmentation_results(
        original, mask, stats, filepath, thresholds=thresholds,
        output_dir=output_dir, show=show,
    )

    n_phases = len(_phase_labels(mask))
    print(
        f"\nResumen: {Path(filepath).name} — {n_phases} fases, método "
        f"'{method_used}'"
    )
    for name, s in stats.items():
        if name == "interfaces":
            continue
        print(
            f"  {name}: {s['area_pct']:.2f} % del área, "
            f"{s['n_regiones']} región(es), intensidad media "
            f"{s['intensidad_media']:.3f}"
        )
    print(f"  interfaces: {stats['interfaces']['area_pct']:.2f} %")

    return {
        "original": original,
        "processed": processed,
        "mask": mask,
        "n_phases": n_phases,
        "threshold_method": method_used,
        "stats": stats,
        "filepath": filepath,
        "thresholds": thresholds,
        "lineas_watershed": watershed_lines,
    }
