"""Análisis composicional relativo de las fases en imágenes TEM.

Etapa 4, Módulo B, del módulo TEM del proyecto "Analizador de Imágenes
de Microscopía". Las fases se llaman siempre "Fase 1", "Fase 2", ...
y nunca se asignan elementos químicos.

En TEM la intensidad depende del espesor, de la difracción y del
enfoque, no solo de la composición: este módulo compara intensidades
relativas entre fases dentro de una misma imagen, no mide composición
química.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from skimage import measure

from .fraction import _interface_neighbors, _phase_label_map, phase_title
from .loader import DEFAULT_OUTPUT_DIR
from .segmentation import CLEAN_MIN_FRACTION, INTERFACE_LABEL, PHASE_COLORS

# --------------------------------------------------------------------
# Parámetros del Bloque B1
# --------------------------------------------------------------------

# Bins del histograma de intensidad por fase (rango 0-1)
PROFILE_BINS: int = 100
# Límites de la clasificación del contraste entre fases
CONTRAST_HIGH: float = 0.3
CONTRAST_LOW: float = 0.1
# Distancia a cada lado de la interfaz para el perfil (px)
INTERFACE_HALF_WIDTH: int = 20
# Píxeles de cada extremo del perfil que definen las mesetas
PLATEAU_PX: int = 5
# Mínimo de píxeles de interfaz para analizar un par
MIN_INTERFACE_PX: int = 100
# Cambio mínimo de intensidad entre mesetas para medir el ancho
MIN_STEP: float = 0.02
# Se considera que el perfil llegó a una meseta en un extremo si ahí
# su pendiente es menor que esta fracción de la pendiente máxima
PLATEAU_MAX_SLOPE_RATIO: float = 0.1


# ====================================================================
# BLOQUE B1 — Perfil de intensidad por fase
# ====================================================================


def _classify_contrast(index: float) -> str:
    """'alto', 'medio' o 'bajo' según CONTRAST_HIGH y CONTRAST_LOW."""
    if index > CONTRAST_HIGH:
        return "alto"
    if index >= CONTRAST_LOW:
        return "medio"
    return "bajo"


def compute_intensity_profiles(
    image: np.ndarray, mask: np.ndarray, background: int | None = None
) -> dict:
    """Distribución de intensidades del fondo y de cada fase.

    El contraste entre dos fases es |media_A - media_B| / (media_A +
    media_B): alto (> CONTRAST_HIGH), medio o bajo (< CONTRAST_LOW). Se
    incluye el fondo, así hay al menos un par aunque haya una sola
    fase. Las intensidades son relativas a la imagen (normalizada en
    la Etapa 1): no sirven para comparar imágenes distintas.

    Args:
        image: Imagen preprocesada en [0, 1].
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario con 'perfiles' ({'fondo'/'fase_N': {histograma,
        bin_edges, media, mediana, moda, std, asimetria, curtosis
        (en exceso: 0 para una normal), iqr, n_pixeles,
        etiqueta_mascara}}) y 'contraste' ({'A-B': {indice,
        clasificacion}}).

    Raises:
        ValueError: Si image y mask no tienen el mismo shape.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image {image.shape} y mask {mask.shape} deben tener el "
            "mismo shape."
        )
    labels = _phase_label_map(mask, background)
    edges = np.linspace(0.0, 1.0, PROFILE_BINS + 1)
    profiles: dict[str, dict] = {}
    for name, label in labels.items():
        values = image[mask == label].astype(np.float64)
        hist, _ = np.histogram(values, bins=edges)
        q25, q75 = np.percentile(values, [25, 75])
        mode_bin = int(np.argmax(hist))
        constant = values.size < 2 or np.ptp(values) == 0
        profiles[name] = {
            "etiqueta_mascara": label,
            "n_pixeles": int(values.size),
            "histograma": hist,
            "bin_edges": edges,
            "media": float(values.mean()),
            "mediana": float(np.median(values)),
            "moda": float((edges[mode_bin] + edges[mode_bin + 1]) / 2),
            "std": float(values.std()),
            # Sin variación, asimetría y curtosis no están definidas
            "asimetria": 0.0 if constant else float(stats.skew(values)),
            "curtosis": 0.0 if constant else float(stats.kurtosis(values)),
            "iqr": float(q75 - q25),
        }

    names = list(profiles)
    contrast: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = profiles[a]["media"], profiles[b]["media"]
            index = abs(ma - mb) / (ma + mb) if ma + mb > 0 else 0.0
            contrast[f"{a}-{b}"] = {
                "indice": float(index),
                "clasificacion": _classify_contrast(index),
            }

    print("Perfiles de intensidad (imagen preprocesada, 0-1):")
    print(f"  {'':8} {'media':>6} {'mediana':>7} {'moda':>5} {'std':>6} "
          f"{'asim.':>6} {'curt.':>6} {'IQR':>6}")
    for name, p in profiles.items():
        print(f"  {phase_title(name).replace('fondo', 'Fondo'):8} "
              f"{p['media']:>6.3f} {p['mediana']:>7.3f} {p['moda']:>5.2f} "
              f"{p['std']:>6.3f} {p['asimetria']:>6.2f} "
              f"{p['curtosis']:>6.2f} {p['iqr']:>6.3f}")
    print("Contraste entre fases |mA - mB| / (mA + mB):")
    for key, c in contrast.items():
        a, b = (phase_title(n).replace("fondo", "Fondo")
                for n in key.split("-"))
        print(f"  {a} / {b}: {c['indice']:.3f} ({c['clasificacion']})")
    return {"perfiles": profiles, "contraste": contrast}


def _crossing(distances: np.ndarray, fraction: np.ndarray,
              level: float) -> float | None:
    """Primera distancia donde la curva normalizada cruza un nivel.

    Args:
        distances: Distancias al borde (px), en orden creciente.
        fraction: Perfil normalizado (0 en la meseta inicial, 1 en la
            final).
        level: Nivel a cruzar (por ejemplo 0.1 o 0.9).

    Returns:
        Distancia interpolada, o None si no lo cruza.
    """
    above = np.flatnonzero(fraction >= level)
    if above.size == 0:
        return None
    j = int(above[0])
    if j == 0:
        return float(distances[0])
    x0, x1 = distances[j - 1], distances[j]
    y0, y1 = fraction[j - 1], fraction[j]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def compute_intensity_gradients(
    image: np.ndarray,
    mask: np.ndarray,
    background: int | None = None,
    watershed_lines: np.ndarray | None = None,
) -> dict:
    """Perfil de intensidad a través de la interfaz de cada par de fases.

    Para cada par A/B (los mismos que compute_interface_fraction) se
    calcula la distancia con signo al borde: negativa del lado de A y
    positiva del lado de B. La intensidad se promedia a cada distancia
    entera entre -INTERFACE_HALF_WIDTH y +INTERFACE_HALF_WIDTH, con los
    píxeles que están a esa distancia de las dos fases y no son de una
    tercera fase. Es el perfil perpendicular promediado a lo largo de
    toda la interfaz.

    Sobre el perfil se mide el ancho de transición (entre el 10 % y el
    90 % del cambio entre las mesetas de los extremos) y la pendiente
    máxima. Si el perfil sigue cambiando en algún extremo
    (meseta_alcanzada = False), el ancho está limitado por la ventana
    y es solo un límite inferior: pasa con los bordes graduales de las
    partículas reales. Además, el percentil 95 del gradiente de Sobel
    (scipy.ndimage.sobel) sobre los píxeles de interfaz del par: el
    máximo píxel a píxel lo decide el ruido.

    Args:
        image: Imagen preprocesada en [0, 1].
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.
        watershed_lines: Líneas de corte del watershed (se excluyen).

    Returns:
        Diccionario {'A-B': {...}} con distancias_px, perfil,
        ancho_transicion_px (None si el cambio de intensidad es menor
        que MIN_STEP), meseta_alcanzada, pendiente_max
        (intensidad/px), sobel_p95 y n_pixeles (píxeles de interfaz
        del par). Los pares con menos de MIN_INTERFACE_PX píxeles de
        interfaz se omiten.
    """
    labels = _phase_label_map(mask, background)
    _, reach = _interface_neighbors(mask, background, watershed_lines)
    excluded = (watershed_lines if watershed_lines is not None
                else np.zeros(mask.shape, dtype=bool))
    sobel = np.hypot(ndimage.sobel(image.astype(np.float64), axis=0),
                     ndimage.sobel(image.astype(np.float64), axis=1))
    half = INTERFACE_HALF_WIDTH
    distances = np.arange(-half, half + 1)
    # Distancia de cada píxel a cada fase (se calcula una sola vez)
    dist_to: dict[str, np.ndarray] = {}

    result: dict[str, dict] = {}
    names = list(labels)
    print("Perfiles de interfaz (distancia con signo al borde):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair_px = reach[a] & reach[b]
            n_px = int(pair_px.sum())
            if n_px == 0:
                continue
            title = (f"{phase_title(a)} / {phase_title(b)}"
                     .replace("fondo", "Fondo"))
            if n_px < MIN_INTERFACE_PX:
                print(f"  {title}: {n_px} px de interfaz (< "
                      f"{MIN_INTERFACE_PX}), se omite")
                continue
            for name in (a, b):
                if name not in dist_to:
                    dist_to[name] = ndimage.distance_transform_edt(
                        mask != labels[name]
                    )
            d_a, d_b = dist_to[a], dist_to[b]
            zone = ((d_a <= half) & (d_b <= half) & ~excluded
                    & np.isin(mask, [labels[a], labels[b],
                                     INTERFACE_LABEL]))
            # Negativa del lado de A, positiva del lado de B. La banda
            # de interfaz (2 px de -1) aleja 1.5 px a los píxeles de cada
            # fase y 0.5 px a los de la banda: se corrige para que 0 sea
            # el borde real
            raw = d_a[zone] - d_b[zone]
            in_band = mask[zone] == INTERFACE_LABEL
            raw = raw - np.sign(raw) * np.where(in_band, 0.5, 1.5)
            signed = np.rint(raw).astype(int)
            keep = np.abs(signed) <= half
            idx = signed[keep] + half
            sums = np.bincount(idx, weights=image[zone][keep],
                               minlength=distances.size)
            counts = np.bincount(idx, minlength=distances.size)
            profile = np.full(distances.size, np.nan)
            ok = counts > 0
            profile[ok] = sums[ok] / counts[ok]
            # Las distancias sin píxeles se completan interpolando
            profile = np.interp(distances, distances[ok], profile[ok])

            start = float(profile[:PLATEAU_PX].mean())
            end = float(profile[-PLATEAU_PX:].mean())
            step = end - start
            width = None
            if abs(step) >= MIN_STEP:
                norm = (profile - start) / step
                s10 = _crossing(distances, norm, 0.1)
                s90 = _crossing(distances, norm, 0.9)
                if s10 is not None and s90 is not None:
                    width = abs(s90 - s10)
            slopes = np.abs(np.diff(profile))
            slope = float(slopes.max())
            # ¿El perfil ya no cambia en los dos extremos?
            limit = PLATEAU_MAX_SLOPE_RATIO * slope
            plateau = bool(slopes[:PLATEAU_PX].mean() < limit
                           and slopes[-PLATEAU_PX:].mean() < limit)
            result[f"{a}-{b}"] = {
                "distancias_px": distances,
                "perfil": profile,
                "ancho_transicion_px": width,
                "meseta_alcanzada": plateau,
                "pendiente_max": slope,
                "sobel_p95": float(np.percentile(sobel[pair_px], 95)),
                "n_pixeles": n_px,
            }
            if width is None:
                width_txt = "-"
            elif plateau:
                width_txt = f"{width:.1f} px"
            else:
                width_txt = f">= {width:.1f} px (sin meseta)"
            print(f"  {title}: ancho 10-90 % = {width_txt}, pendiente "
                  f"máx. = {slope:.4f}/px, Sobel p95 = "
                  f"{result[f'{a}-{b}']['sobel_p95']:.3f} ({n_px} px)")
    if not result:
        print("  No hay pares de fases con interfaz suficiente.")
    return result


# ====================================================================
# BLOQUE B2 — Mapas composicionales
# ====================================================================

# Suavizado (gaussiano, en px) antes de buscar inhomogeneidades
INHOMOGENEITY_SIGMA: float = 2.0
# Umbral de z-score para una zona anómala
INHOMOGENEITY_Z: float = 2.5


def _phase_stats(image: np.ndarray, region: np.ndarray) -> tuple[float,
                                                                   float]:
    """Media y desvío de la imagen dentro de una región.

    Args:
        image: Imagen 2D.
        region: Máscara booleana de la región.

    Returns:
        Tupla (media, desvío). (0, 0) si la región está vacía.
    """
    values = image[region].astype(np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def build_phase_intensity_map(
    image: np.ndarray, mask: np.ndarray, background: int | None = None
) -> dict:
    """Mapas de intensidad y de z-score del fondo y de cada fase.

    El z-score de un píxel es (intensidad - media de su fase) / desvío
    de su fase. Fuera de la fase, los mapas valen NaN. El mapa
    zscore_global junta todos: cada píxel con el z-score respecto de
    su propia fase (NaN en interfaces). Los mapas son float32.

    Args:
        image: Imagen preprocesada en [0, 1].
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fondo'/'fase_N': {intensidad, zscore, media,
        std}} más 'zscore_global'. Si una fase es uniforme (std 0), su
        z-score vale 0.

    Raises:
        ValueError: Si image y mask no tienen el mismo shape.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image {image.shape} y mask {mask.shape} deben tener el "
            "mismo shape."
        )
    result: dict = {}
    global_z = np.full(mask.shape, np.nan, dtype=np.float32)
    for name, label in _phase_label_map(mask, background).items():
        region = mask == label
        mean, std = _phase_stats(image, region)
        intensity = np.full(mask.shape, np.nan, dtype=np.float32)
        intensity[region] = image[region]
        z = np.full(mask.shape, np.nan, dtype=np.float32)
        z[region] = (image[region] - mean) / std if std > 0 else 0.0
        global_z[region] = z[region]
        result[name] = {"intensidad": intensity, "zscore": z,
                        "media": mean, "std": std}
    result["zscore_global"] = global_z
    return result


def detect_inhomogeneities(
    image: np.ndarray, mask: np.ndarray, background: int | None = None
) -> dict:
    """Zonas de cada fase con intensidad anómala respecto de su fase.

    Para no marcar ruido: (1) la imagen se suaviza dentro de cada fase
    con una gaussiana de INHOMOGENEITY_SIGMA px (solo promedia píxeles
    de esa fase); (2) se calcula el z-score del valor suavizado con la
    media y el desvío de la fase sin suavizar; (3) los píxeles con
    z > INHOMOGENEITY_Z son de alta intensidad y con z <
    -INHOMOGENEITY_Z de baja; (4) solo cuentan las zonas conectadas de
    al menos CLEAN_MIN_FRACTION de la imagen (419 px en 2048 x 2048).

    El desvío de la fase incluye su estructura real (por ejemplo, dos
    niveles de gris): una inhomogeneidad es una zona que se aparta del
    comportamiento típico de su fase, no necesariamente un defecto.

    Args:
        image: Imagen preprocesada en [0, 1].
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fondo'/'fase_N': {alta, baja, n_alta, n_baja,
        area_anomala_pct}}, donde alta y baja son listas de dicts con
        area_px, centroide (y, x), z_medio y bbox.
    """
    min_size = max(1, int(round(CLEAN_MIN_FRACTION * mask.size)))
    result: dict[str, dict] = {}
    print(f"Inhomogeneidades (|z| > {INHOMOGENEITY_Z} tras suavizar "
          f"σ={INHOMOGENEITY_SIGMA} px, zonas >= {min_size} px):")
    for name, label in _phase_label_map(mask, background).items():
        region = mask == label
        mean, std = _phase_stats(image, region)
        entry: dict = {"alta": [], "baja": [], "n_alta": 0, "n_baja": 0,
                       "area_anomala_pct": 0.0}
        if std > 0:
            # Suavizado normalizado: solo promedia píxeles de la fase
            weight = ndimage.gaussian_filter(region.astype(np.float64),
                                             INHOMOGENEITY_SIGMA)
            smooth = ndimage.gaussian_filter(
                np.where(region, image, 0.0).astype(np.float64),
                INHOMOGENEITY_SIGMA,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                smooth = smooth / weight
            z = np.where(region, (smooth - mean) / std, 0.0)
            anomalous_px = 0
            for kind, zone in (("alta", z > INHOMOGENEITY_Z),
                               ("baja", z < -INHOMOGENEITY_Z)):
                labels = measure.label(zone, connectivity=2)
                for r in measure.regionprops(labels):
                    if r.area < min_size:
                        continue
                    rows, cols = r.coords[:, 0], r.coords[:, 1]
                    entry[kind].append({
                        "area_px": int(r.area),
                        "centroide": (float(r.centroid[0]),
                                      float(r.centroid[1])),
                        "z_medio": float(z[rows, cols].mean()),
                        "bbox": tuple(int(v) for v in r.bbox),
                    })
                    anomalous_px += int(r.area)
            entry["n_alta"] = len(entry["alta"])
            entry["n_baja"] = len(entry["baja"])
            entry["area_anomala_pct"] = (100.0 * anomalous_px
                                         / int(region.sum()))
        result[name] = entry
        title = phase_title(name).replace("fondo", "Fondo")
        print(f"  {title}: {entry['n_alta']} zona(s) alta(s), "
              f"{entry['n_baja']} baja(s) — "
              f"{entry['area_anomala_pct']:.2f} % del área de la fase")
    return result


# ====================================================================
# BLOQUE B3 — Visualización y pipeline composicional
# ====================================================================


def _display_name(key: str) -> str:
    """'fondo' → 'Fondo', 'fase_1' → 'Fase 1'."""
    return phase_title(key).replace("fondo", "Fondo")


def _width_text(gradient: dict) -> str:
    """Ancho de transición para mostrar ('≥' si no hay meseta)."""
    width = gradient["ancho_transicion_px"]
    if width is None:
        return "-"
    if gradient["meseta_alcanzada"]:
        return f"{width:.1f} px"
    return f"≥ {width:.1f} px (sin meseta)"


def show_compositional_results(
    image: np.ndarray,
    mask: np.ndarray,
    profiles: dict,
    gradients: dict,
    inhomogeneities: dict,
    filepath: str | Path,
    maps: dict | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la figura del análisis composicional.

    Paneles: (1) imagen con las inhomogeneidades (rojo: alta
    intensidad, azul: baja); (2) mapa de z-score respecto de cada fase
    (RdBu, blanco = media de la fase); (3) histogramas de intensidad
    del fondo y de cada fase con sus medias; (4 en adelante) perfil de
    cada interfaz con el ancho de transición. Se guarda como
    '<nombre>_composicional.png'.

    Args:
        image: Imagen que se muestra en el panel 1 (en el pipeline, la
            original).
        mask: Máscara de la Etapa 2.
        profiles: Resultado de compute_intensity_profiles.
        gradients: Resultado de compute_intensity_gradients.
        inhomogeneities: Resultado de detect_inhomogeneities.
        filepath: Ruta del archivo de origen (para título y nombre).
        maps: Resultado de build_phase_intensity_map. Si es None, el
            mapa de z-score se calcula con image.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_composicional.png"
    if maps is None:
        maps = build_phase_intensity_map(image, mask)

    pairs = list(gradients)
    total = 3 + len(pairs)
    cols = 3 if total <= 6 else 4
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows),
                             squeeze=False)
    axes = axes.ravel()

    # Panel 1: inhomogeneidades sobre la imagen
    axes[0].imshow(image, cmap="gray")
    n_total = 0
    for data in inhomogeneities.values():
        for kind, color in (("alta", "red"), ("baja", "blue")):
            for zone in data[kind]:
                y0, x0, y1, x1 = zone["bbox"]
                axes[0].add_patch(plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False,
                    edgecolor=color, linewidth=1.5))
                n_total += 1
    axes[0].plot([], [], "s", color="red",
                 label="más clara que su fase")
    axes[0].plot([], [], "s", color="blue",
                 label="más oscura que su fase")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(f"Inhomogeneidades: {n_total}")
    axes[0].axis("off")

    # Panel 2: z-score respecto de cada fase
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#808080")
    shown = axes[1].imshow(np.ma.masked_invalid(maps["zscore_global"]),
                           cmap=cmap, vmin=-3, vmax=3,
                           interpolation="nearest")
    fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.04,
                 label="z-score (0 = media de su fase)")
    axes[1].set_title("z-score respecto de cada fase")
    axes[1].axis("off")

    # Panel 3: histogramas por fase
    for key, p in profiles["perfiles"].items():
        color = PHASE_COLORS[p["etiqueta_mascara"] % len(PHASE_COLORS)]
        centers = (p["bin_edges"][:-1] + p["bin_edges"][1:]) / 2
        axes[2].fill_between(centers, p["histograma"], step="mid",
                             color=color, alpha=0.5,
                             label=f"{_display_name(key)} "
                                   f"(media {p['media']:.2f})")
        axes[2].axvline(p["media"], color=color, linestyle="--")
    # Contraste de cada par, uno por línea, en un recuadro
    contrast = "\n".join(
        f"{' / '.join(_display_name(n) for n in k.split('-'))}: "
        f"{c['indice']:.2f} ({c['clasificacion']})"
        for k, c in profiles["contraste"].items()
    )
    axes[2].text(0.02, 0.98, f"Contraste:\n{contrast}", fontsize=7,
                 va="top", ha="left", transform=axes[2].transAxes,
                 bbox={"facecolor": "white", "alpha": 0.8,
                       "edgecolor": "gray"})
    axes[2].set_title("Intensidad por fase (línea: media)")
    axes[2].set_xlabel("Intensidad (imagen preprocesada)")
    axes[2].set_ylabel("Píxeles")
    axes[2].legend(fontsize=8, loc="upper right")

    # Paneles por par: perfil de interfaz
    for ax, key in zip(axes[3:], pairs):
        g = gradients[key]
        a, b = (_display_name(n) for n in key.split("-"))
        d, profile = g["distancias_px"], g["perfil"]
        ax.plot(d, profile, "o-", color="black", markersize=3)
        ax.axvline(0, color="gray", linestyle=":")
        start = profile[:PLATEAU_PX].mean()
        end = profile[-PLATEAU_PX:].mean()
        ax.axhline(start, color="tab:blue", linestyle="--", linewidth=1)
        ax.axhline(end, color="tab:red", linestyle="--", linewidth=1)
        if g["ancho_transicion_px"] is not None:
            norm = (profile - start) / (end - start)
            order = np.argsort(norm)
            x10 = float(np.interp(0.1, norm[order], d[order]))
            x90 = float(np.interp(0.9, norm[order], d[order]))
            ax.axvspan(min(x10, x90), max(x10, x90), color="gold",
                       alpha=0.4, label="ancho 10-90 %")
            ax.legend(fontsize=8)
        ax.set_title(f"{a} → {b}: ancho {_width_text(g)}", fontsize=10)
        ax.set_xlabel(f"Distancia al borde (px): − {a}, + {b}")
        ax.set_ylabel("Intensidad media")
    for ax in axes[3 + len(pairs):]:
        ax.axis("off")

    fig.suptitle(f"Análisis composicional relativo: {path.name}",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# Claves que debe traer el diccionario de pipeline_fraction
_FRACTION_KEYS: tuple[str, ...] = (
    "original", "processed", "mask", "filepath", "background",
)


def pipeline_compositional(
    fraction_dict: dict,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Análisis composicional relativo completo.

    Secuencia: compute_intensity_profiles →
    compute_intensity_gradients → build_phase_intensity_map →
    detect_inhomogeneities → show_compositional_results. Los cálculos
    usan la imagen preprocesada; la figura muestra la original. Se
    usa el mismo fondo que eligió el Módulo A.

    Args:
        fraction_dict: Diccionario de pipeline_fraction (Módulo A).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        El diccionario de entrada más 'intensity_profiles',
        'interface_gradients', 'phase_maps' (mapas float32 del tamaño
        de la imagen) e 'inhomogeneities'.

    Raises:
        ValueError: Si falta alguna clave en fraction_dict.
    """
    missing = [k for k in _FRACTION_KEYS if k not in fraction_dict]
    if missing:
        raise ValueError(
            f"Al diccionario de entrada le falta(n): {', '.join(missing)}."
            " Debe ser el resultado de pipeline_fraction()."
        )
    processed = fraction_dict["processed"]
    mask = fraction_dict["mask"]
    bg = fraction_dict["background"]
    lines = fraction_dict.get("lineas_watershed")

    print("--- B1. Perfiles de intensidad ---")
    profiles = compute_intensity_profiles(processed, mask, bg)
    gradients = compute_intensity_gradients(processed, mask, bg, lines)
    print("--- B2. Mapas e inhomogeneidades ---")
    maps = build_phase_intensity_map(processed, mask, bg)
    inhom = detect_inhomogeneities(processed, mask, bg)
    print("--- B3. Figura ---")
    show_compositional_results(
        fraction_dict["original"], mask, profiles, gradients, inhom,
        fraction_dict["filepath"], maps=maps, output_dir=output_dir,
        show=show,
    )

    print(f"\nResumen composicional: {Path(fraction_dict['filepath']).name}")
    for key, c in profiles["contraste"].items():
        pair = " / ".join(_display_name(n) for n in key.split("-"))
        width = (_width_text(gradients[key]) if key in gradients
                 else "sin interfaz suficiente")
        print(f"  {pair}: contraste {c['indice']:.2f} "
              f"({c['clasificacion']}), ancho de interfaz {width}")
    for key, d in inhom.items():
        print(f"  {_display_name(key)}: {d['n_alta']} zona(s) más "
              f"clara(s) y {d['n_baja']} más oscura(s) que su fase")

    return {
        **fraction_dict,
        "intensity_profiles": profiles,
        "interface_gradients": gradients,
        "phase_maps": maps,
        "inhomogeneities": inhom,
    }
