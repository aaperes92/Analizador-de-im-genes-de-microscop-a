"""Análisis de fracción de fases (estereología) en imágenes TEM.

Etapa 4, Módulo A, del módulo TEM del proyecto "Analizador de Imágenes
de Microscopía". Las fases se llaman siempre "Fase 1", "Fase 2", ...
(de la más oscura a la más clara, sin contar el fondo): nunca se
asignan elementos químicos.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage
from scipy.spatial.distance import cdist

from .loader import DEFAULT_OUTPUT_DIR
from .morphology import _majority_phase
from .segmentation import INTERFACE_LABEL, PHASE_COLORS, _phase_labels

# Tolerancia (en puntos porcentuales) para verificar que las
# fracciones sumen 100 %
SUM_TOLERANCE_PCT: float = 0.01

# Advertencia que acompaña a la fracción de volumen estimada
VOLUME_WARNING: str = (
    "Estimación por el principio de Delesse (fracción de volumen ≈ "
    "fracción de área). Supone una distribución isótropa y que la "
    "imagen es un corte plano. Una imagen TEM es una proyección de todo "
    "el espesor: las partículas se superponen y la fracción de área "
    "tiende a sobreestimar la de volumen."
)


# ====================================================================
# Nombres de las fases (compartido por los módulos de la Etapa 4)
# ====================================================================


def phase_names(
    mask: np.ndarray, background: int | None = None
) -> tuple[int | None, dict[str, int]]:
    """Nombres de la Etapa 4: fondo y Fase 1, Fase 2, ...

    Por defecto el fondo es la fase mayoritaria (mismo criterio que la
    Etapa 3). Las demás se numeran desde 1, de la más oscura a la más
    clara (en la máscara, el orden de las etiquetas es el de
    intensidad).

    Args:
        mask: Máscara de la Etapa 2 (-1 = interfaz, 0, 1, ... = fases).
        background: Etiqueta de la máscara que es el fondo. None para
            usar la fase mayoritaria (por ejemplo, si las partículas
            ocupan más de la mitad de la imagen hay que indicarlo).

    Returns:
        Tupla (etiqueta del fondo o None, {'fase_1': etiqueta, ...}).

    Raises:
        ValueError: Si background no es una etiqueta de la máscara.
    """
    present = _phase_labels(mask)
    if background is None:
        background = _majority_phase(mask)
    elif background not in present:
        raise ValueError(
            f"background={background} no es una fase de la máscara "
            f"(etiquetas presentes: {present})."
        )
    others = [k for k in present if k != background]
    return background, {f"fase_{i}": k
                        for i, k in enumerate(others, start=1)}


def phase_title(name: str) -> str:
    """Convierte 'fase_1' en 'Fase 1' (para títulos y tablas).

    Args:
        name: Clave de la fase.

    Returns:
        Nombre para mostrar.
    """
    return name.replace("fase_", "Fase ")


# ====================================================================
# BLOQUE A1 — Fracciones de área y volumen
# ====================================================================


def compute_area_fractions(
    mask: np.ndarray, background: int | None = None
) -> dict:
    """Fracción de área de cada fase, del fondo y de las interfaces.

    Args:
        mask: Máscara de la Etapa 2 (-1 = interfaz, 0, 1, ... = fases).
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario con:
        - 'fase_N': area_px, area_pct (sobre el área sin fondo ni
          interfaces: las fases suman 100 %), area_pct_con_fondo
          (sobre el total) y etiqueta_mascara.
        - 'interfaces': area_px y area_pct (sobre el total).
        - 'fondo': area_px, area_pct (sobre el total) y
          etiqueta_mascara.
        - 'total_px'.

    Raises:
        ValueError: Si las fracciones no suman 100 % (tolerancia
            SUM_TOLERANCE_PCT).
    """
    total = int(mask.size)
    background, names = phase_names(mask, background)
    n_interface = int(np.sum(mask == INTERFACE_LABEL))
    n_background = (int(np.sum(mask == background))
                    if background is not None else 0)
    n_phases_px = total - n_interface - n_background

    result: dict = {}
    for name, label in names.items():
        area = int(np.sum(mask == label))
        result[name] = {
            "area_px": area,
            "area_pct": 100.0 * area / n_phases_px if n_phases_px else 0.0,
            "area_pct_con_fondo": 100.0 * area / total,
            "etiqueta_mascara": label,
        }
    result["interfaces"] = {
        "area_px": n_interface,
        "area_pct": 100.0 * n_interface / total,
    }
    result["fondo"] = {
        "area_px": n_background,
        "area_pct": 100.0 * n_background / total,
        "etiqueta_mascara": background,
    }
    result["total_px"] = total

    # Verificación: todo suma 100 % y las fases suman 100 % entre sí
    total_pct = (sum(result[n]["area_pct_con_fondo"] for n in names)
                 + result["interfaces"]["area_pct"]
                 + result["fondo"]["area_pct"])
    phases_pct = sum(result[n]["area_pct"] for n in names)
    errors = []
    if abs(total_pct - 100.0) > SUM_TOLERANCE_PCT:
        errors.append(f"fases + interfaces + fondo = {total_pct:.4f} %")
    if names and abs(phases_pct - 100.0) > SUM_TOLERANCE_PCT:
        errors.append(f"fases (sin fondo) = {phases_pct:.4f} %")
    if errors:
        raise ValueError(
            "Las fracciones no suman 100 % (tolerancia "
            f"{SUM_TOLERANCE_PCT} %): " + "; ".join(errors)
        )

    print("Fracciones de área:")
    print(f"  {'':12} {'etiqueta':>8} {'píxeles':>10} "
          f"{'% sin fondo':>11} {'% del total':>11}")
    for name in names:
        r = result[name]
        print(f"  {phase_title(name):12} {r['etiqueta_mascara']:>8} "
              f"{r['area_px']:>10} {r['area_pct']:>11.2f} "
              f"{r['area_pct_con_fondo']:>11.2f}")
    print(f"  {'Interfaces':12} {'-1':>8} {n_interface:>10} {'':>11} "
          f"{result['interfaces']['area_pct']:>11.2f}")
    print(f"  {'Fondo':12} {str(background):>8} {n_background:>10} "
          f"{'':>11} {result['fondo']['area_pct']:>11.2f}")
    print(f"  Total: {total} px = {total_pct:.4f} %")
    if not names:
        print("  Aviso: la máscara tiene una sola fase; no hay fases "
              "para analizar fuera del fondo.")
    return result


def estimate_volume_fractions(area_fractions: dict) -> dict:
    """Agrega la fracción de volumen estimada (principio de Delesse).

    Hay dos lecturas: dentro del material (sin fondo, como pide la
    especificación) y sobre toda la imagen (con fondo). Con una sola
    fase fuera del fondo, la primera vale siempre 100 % y la útil es la
    segunda (la cobertura de las partículas). No modifica el
    diccionario de entrada: devuelve una copia.

    Args:
        area_fractions: Resultado de compute_area_fractions.

    Returns:
        Copia con, en cada fase: vol_frac_estimada (= area_pct / 100),
        vol_frac_pct, vol_frac_estimada_con_fondo (= area_pct_con_fondo
        / 100), vol_frac_pct_con_fondo y advertencia (VOLUME_WARNING).
    """
    result = copy.deepcopy(area_fractions)
    phases = [k for k in result if k.startswith("fase_")]
    print("Fracciones de volumen estimadas:")
    print(f"  {'':8} {'sin fondo':>10} {'sobre el total':>15}")
    for name in phases:
        entry = result[name]
        entry["vol_frac_estimada"] = entry["area_pct"] / 100.0
        entry["vol_frac_pct"] = entry["area_pct"]
        entry["vol_frac_estimada_con_fondo"] = (
            entry["area_pct_con_fondo"] / 100.0
        )
        entry["vol_frac_pct_con_fondo"] = entry["area_pct_con_fondo"]
        entry["advertencia"] = VOLUME_WARNING
        print(f"  {phase_title(name):8} {entry['vol_frac_pct']:>9.2f} % "
              f"{entry['vol_frac_pct_con_fondo']:>13.2f} %")
    if len(phases) == 1:
        print("  Nota: hay una sola fase fuera del fondo, así que 'sin "
              "fondo' vale 100 %; el dato útil es 'sobre el total'.")
    print(f"  ADVERTENCIA: {VOLUME_WARNING}")
    return result


# ====================================================================
# BLOQUE A2 — Densidad numérica
# ====================================================================


def _nearest_neighbor_distances(centroids: np.ndarray) -> np.ndarray:
    """Distancia de cada punto a su vecino más cercano.

    Args:
        centroids: Array (n, 2) de coordenadas (y, x), con n >= 2.

    Returns:
        Array (n,) con la distancia al vecino más cercano.
    """
    distances = cdist(centroids, centroids)
    # La distancia de un punto a sí mismo no cuenta
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def compute_number_density(
    props: dict[str, list[dict]],
    mask: np.ndarray,
    background: int | None = None,
) -> dict:
    """Cantidad de partículas por área y separación entre vecinas.

    props viene de la Etapa 3, con las fases nombradas por su etiqueta
    de la máscara; el resultado usa los nombres de la Etapa 4 (Fase 1,
    Fase 2, ...). Se cuentan todas las regiones, también las que tocan
    el borde. La separación es la distancia de cada centroide al de su
    vecino más cercano de la misma fase. El índice de Clark-Evans
    compara esa separación con la esperada si las partículas
    estuvieran al azar (0.5 / sqrt(densidad)): ≈ 1 al azar, < 1
    agrupadas, > 1 más espaciadas. No corrige el efecto del borde, que
    lo sobreestima un poco.

    Args:
        props: Resultado de extract_region_properties (Etapa 3).
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        Diccionario {'fase_N': {...}} con etiqueta_mascara, n_total,
        densidad_por_px2, densidad_por_1000px2, separacion_media_px,
        separacion_std_px, separacion_esperada_azar_px e
        indice_clark_evans. Sin regiones en props, los valores son None;
        con menos de 2 regiones, la separación es None.
    """
    _, names = phase_names(mask, background)
    total = float(mask.size)
    result: dict[str, dict] = {}

    print("Densidad numérica (todas las regiones, también las del "
          "borde):")
    print(f"  {'':8} {'n':>5} {'dens/1000px²':>12} {'sep. media':>10} "
          f"{'sep. azar':>9} {'Clark-Evans':>11}")
    for name, label in names.items():
        regions = props.get(f"fase_{label}")
        entry: dict = {"etiqueta_mascara": label}
        if regions is None:
            entry.update({k: None for k in (
                "n_total", "densidad_por_px2", "densidad_por_1000px2",
                "separacion_media_px", "separacion_std_px",
                "separacion_esperada_azar_px", "indice_clark_evans",
            )})
            result[name] = entry
            print(f"  {phase_title(name):8} sin datos en props (la Etapa "
                  f"3 no analizó la etiqueta {label})")
            continue

        n = len(regions)
        density = n / total
        entry.update({
            "n_total": n,
            "densidad_por_px2": density,
            "densidad_por_1000px2": 1000.0 * density,
            "separacion_media_px": None,
            "separacion_std_px": None,
            "separacion_esperada_azar_px": None,
            "indice_clark_evans": None,
        })
        if n >= 2:
            centroids = np.array([r["centroide"] for r in regions])
            nearest = _nearest_neighbor_distances(centroids)
            expected = 0.5 / np.sqrt(density)
            entry.update({
                "separacion_media_px": float(nearest.mean()),
                "separacion_std_px": float(nearest.std()),
                "separacion_esperada_azar_px": float(expected),
                "indice_clark_evans": float(nearest.mean() / expected),
            })
        result[name] = entry

        def fmt(value: float | None, spec: str) -> str:
            return "-" if value is None else format(value, spec)

        print(f"  {phase_title(name):8} {n:>5} "
              f"{entry['densidad_por_1000px2']:>12.4f} "
              f"{fmt(entry['separacion_media_px'], '.1f'):>10} "
              f"{fmt(entry['separacion_esperada_azar_px'], '.1f'):>9} "
              f"{fmt(entry['indice_clark_evans'], '.2f'):>11}")
    return result


# ====================================================================
# BLOQUE A3 — Fracción de interfaces
# ====================================================================

# Radio de la dilatación para cruzar la banda de interfaz (2 px)
INTERFACE_REACH: int = 2


def _phase_label_map(
    mask: np.ndarray, background: int | None = None
) -> dict[str, int]:
    """Nombres de la Etapa 4 (con 'fondo') y su etiqueta de la máscara.

    Args:
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.

    Returns:
        {'fondo': etiqueta, 'fase_1': etiqueta, ...}.
    """
    bg, names = phase_names(mask, background)
    return {"fondo": bg, **names} if bg is not None else dict(names)


def _interface_neighbors(
    mask: np.ndarray,
    background: int | None = None,
    watershed_lines: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Qué fases (con nombres de la Etapa 4) llegan a cada interfaz.

    Cada fase y el fondo se dilatan INTERFACE_REACH píxeles: así cruzan
    la banda de interfaz de 2 px y llegan a los píxeles -1 que los
    separan de la fase vecina. Las líneas de watershed, si se pasan, no
    son interfaces entre fases y se excluyen.

    Args:
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.
        watershed_lines: Máscara booleana de las líneas de corte
            (clave 'lineas_watershed' de pipeline_segmentation).

    Returns:
        Tupla (máscara booleana de interfaces entre fases, {nombre:
        máscara booleana de los píxeles de interfaz a los que llega
        esa fase}).
    """
    interface = mask == INTERFACE_LABEL
    if watershed_lines is not None:
        interface &= ~watershed_lines
    size = 2 * INTERFACE_REACH + 1
    reach = {
        name: interface & ndimage.maximum_filter(mask == label, size=size)
        for name, label in _phase_label_map(mask, background).items()
    }
    return interface, reach


def compute_interface_fraction(
    mask: np.ndarray,
    background: int | None = None,
    watershed_lines: np.ndarray | None = None,
) -> dict:
    """Fracción de interfaces y longitud compartida por cada par de fases.

    Un píxel de interfaz al que llegan dos fases (tras dilatarlas
    INTERFACE_REACH px) es parte de la interfaz de ese par; con tres o
    más, es un punto triple y cuenta para cada par. La longitud de un
    par es su cantidad de píxeles dividida por 2 (la banda tiene 2 px
    de ancho). Contar píxeles sobreestima la longitud en los bordes
    diagonales (hasta ~40 %).

    Los contactos entre partículas de una misma fase son las líneas de
    corte del watershed. En la máscara final tienen -1 igual que las
    interfaces, así que solo se pueden medir si se pasan aparte
    (watershed_lines); si no, quedan en None. Las líneas tienen 1 px
    de ancho: su longitud es su cantidad de píxeles.

    Los píxeles de interfaz a los que llega una sola fase (puntas de
    menos de 2 px que la Etapa 2 convirtió enteras en interfaz) se
    informan como sin_clasificar_px.

    Args:
        mask: Máscara de la Etapa 2.
        background: Etiqueta del fondo. None para la fase mayoritaria.
        watershed_lines: Máscara booleana de las líneas de corte
            (clave 'lineas_watershed' de pipeline_segmentation).

    Returns:
        Diccionario con fraccion_total_interfaces (tanto por uno, con
        interfaces y líneas), porcentaje_total_interfaces,
        pares_adyacentes ({'A-B': {'longitud_px', 'fraccion'}}, con
        fracciones que suman 1), contactos_misma_fase ({'fase_N':
        {'longitud_px'}} o None sin watershed_lines) y
        sin_clasificar_px.
    """
    interface, reach = _interface_neighbors(mask, background,
                                            watershed_lines)
    names = list(reach)
    # Cuántas fases llegan a cada píxel de interfaz
    n_reaching = np.zeros(mask.shape, dtype=np.int32)
    for hit in reach.values():
        n_reaching += hit

    pairs: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = int(np.sum(reach[a] & reach[b]))
            if shared:
                pairs[f"{a}-{b}"] = {"longitud_px": shared / 2.0}
    total_length = sum(p["longitud_px"] for p in pairs.values())
    for p in pairs.values():
        p["fraccion"] = (p["longitud_px"] / total_length
                         if total_length else 0.0)
    pairs = dict(sorted(pairs.items(),
                        key=lambda kv: -kv[1]["longitud_px"]))

    # Contactos: cada línea de corte está dentro de una sola fase
    contacts: dict[str, dict] | None = None
    if watershed_lines is not None:
        contacts = {}
        for name, label in _phase_label_map(mask, background).items():
            if name == "fondo":
                continue
            inside = ndimage.maximum_filter(mask == label, size=3)
            contacts[name] = {
                "longitud_px": float(np.sum(watershed_lines & inside)),
            }

    unclassified = int(np.sum(interface & (n_reaching < 2)))
    fraction = float(np.mean(mask == INTERFACE_LABEL))
    result = {
        "fraccion_total_interfaces": fraction,
        "porcentaje_total_interfaces": 100.0 * fraction,
        "pares_adyacentes": pairs,
        "contactos_misma_fase": contacts,
        "sin_clasificar_px": unclassified,
    }

    print(f"Interfaces: {100 * fraction:.2f} % de la imagen "
          "(interfaces entre fases y líneas de watershed)")
    if pairs:
        print(f"  {'par':22} {'longitud (px)':>13} {'fracción':>9}")
        for key, p in pairs.items():
            a, b = key.split("-")
            print(f"  {phase_title(a) + ' / ' + phase_title(b):22} "
                  f"{p['longitud_px']:>13.0f} {p['fraccion']:>9.3f}")
    else:
        print("  No hay fases adyacentes.")
    if contacts is None:
        print("  Contactos entre partículas: no se pueden medir sin "
              "watershed_lines")
    else:
        for name, c in contacts.items():
            print(f"  Contactos entre partículas de {phase_title(name)}: "
                  f"{c['longitud_px']:.0f} px (líneas de watershed)")
    if unclassified:
        print(f"  Sin clasificar: {unclassified} px de interfaz a los que "
              "llega una sola fase (puntas muy finas)")
    return result


# ====================================================================
# BLOQUE A4 — Visualización y pipeline de fracción
# ====================================================================

# Color de las interfaces en la torta y el mapa
INTERFACE_PIE_COLOR: str = "#d0d0d0"
# Cuánto se aclara el fondo en el mapa (0 = color pleno, 1 = blanco)
BACKGROUND_FADE: float = 0.7
# Con esta cantidad de regiones o menos se numeran en el mapa
MAX_LABELED_REGIONS: int = 30


def _label_color(label: int) -> str:
    """Color de una etiqueta de la máscara (igual que en la Etapa 2)."""
    return PHASE_COLORS[label % len(PHASE_COLORS)]


def show_fraction_results(
    mask: np.ndarray,
    area_fracs: dict,
    vol_fracs: dict,
    density: dict,
    filepath: str | Path,
    interfaces: dict | None = None,
    props: dict[str, list[dict]] | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Muestra y guarda la figura del análisis de fracción.

    Paneles: (1) torta de fracciones de área sobre el total (fondo,
    fases e interfaces); (2) barras con la fracción de volumen
    estimada dentro del material y sobre el total; (3) mapa de fases
    con la densidad de cada fase en la leyenda (y el número de cada
    región si hay MAX_LABELED_REGIONS o menos); (4) tabla con todas las
    métricas. Se guarda como '<nombre>_fracciones.png'.

    Args:
        mask: Máscara de la Etapa 2.
        area_fracs: Resultado de compute_area_fractions.
        vol_fracs: Resultado de estimate_volume_fractions.
        density: Resultado de compute_number_density.
        filepath: Ruta del archivo de origen (para título y nombre).
        interfaces: Resultado de compute_interface_fraction (opcional,
            agrega una fila a la tabla).
        props: Resultado de extract_region_properties (opcional, para
            numerar las regiones en el mapa).
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        Ruta del archivo PNG guardado.
    """
    path = Path(filepath)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_fracciones.png"

    phases = [k for k in area_fracs if k.startswith("fase_")]
    bg = area_fracs["fondo"]["etiqueta_mascara"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_pie, ax_bar, ax_map, ax_tab = axes.ravel()

    # Panel 1: torta sobre el total de la imagen
    labels, values, colors = [], [], []
    if bg is not None:
        labels.append("Fondo")
        values.append(area_fracs["fondo"]["area_pct"])
        colors.append(_label_color(bg))
    for name in phases:
        labels.append(phase_title(name))
        values.append(area_fracs[name]["area_pct_con_fondo"])
        colors.append(_label_color(area_fracs[name]["etiqueta_mascara"]))
    labels.append("Interfaces")
    values.append(area_fracs["interfaces"]["area_pct"])
    colors.append(INTERFACE_PIE_COLOR)
    ax_pie.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, wedgeprops={"edgecolor": "white"},
               pctdistance=0.8)
    ax_pie.set_title("Fracción de área sobre el total de la imagen")

    # Panel 2: fracción de volumen estimada, dos lecturas
    y = np.arange(len(phases))
    inside = [vol_fracs[n]["vol_frac_pct"] for n in phases]
    total = [vol_fracs[n]["vol_frac_pct_con_fondo"] for n in phases]
    ax_bar.barh(y - 0.2, inside, height=0.4, color="tab:blue",
                label="dentro del material (sin fondo)")
    ax_bar.barh(y + 0.2, total, height=0.4, color="tab:orange",
                label="sobre el total de la imagen")
    for yi, a, t in zip(y, inside, total):
        ax_bar.text(a, yi - 0.2, f" {a:.1f} %", va="center")
        ax_bar.text(t, yi + 0.2, f" {t:.1f} %", va="center")
    ax_bar.set_yticks(y, [phase_title(n) for n in phases])
    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Fracción de volumen estimada (%)")
    ax_bar.set_title("Volumen estimado (Delesse: volumen ≈ área)")
    ax_bar.legend(loc="lower right")
    ax_bar.invert_yaxis()

    # Panel 3: mapa de fases (fondo atenuado, interfaces en blanco)
    rgb = np.ones(mask.shape + (3,))
    if bg is not None:
        faded = np.array(to_rgb(_label_color(bg)))
        rgb[mask == bg] = faded + (1 - faded) * BACKGROUND_FADE
    for name in phases:
        label = area_fracs[name]["etiqueta_mascara"]
        rgb[mask == label] = to_rgb(_label_color(label))
        d = density.get(name, {})
        texto = f"{phase_title(name)}"
        if d.get("n_total") is not None:
            texto += (f": n={d['n_total']}, "
                      f"{d['densidad_por_1000px2']:.4f}/1000px²")
            if d.get("indice_clark_evans") is not None:
                texto += f", R={d['indice_clark_evans']:.2f}"
        ax_map.plot([], [], "s", color=_label_color(label), label=texto)
        regions = (props or {}).get(f"fase_{label}", [])
        if 0 < len(regions) <= MAX_LABELED_REGIONS:
            for r in regions:
                ax_map.text(r["centroide"][1], r["centroide"][0],
                            str(r["id"]), fontsize=7, ha="center",
                            va="center")
    ax_map.imshow(rgb, interpolation="nearest")
    ax_map.set_title("Fases (fondo atenuado) y densidad numérica")
    ax_map.legend(loc="lower right", fontsize=8)
    ax_map.axis("off")

    # Panel 4: tabla con todas las métricas
    ax_tab.axis("off")
    header = ["Fase", "Área %\nsin fondo", "Área %\ntotal",
              "Vol. % est.\nsin fondo", "n", "Dens./\n1000 px²",
              "Separación\nmedia (px)", "R"]

    def fmt(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    rows = []
    for name in phases:
        a, v, d = area_fracs[name], vol_fracs[name], density.get(name, {})
        rows.append([
            phase_title(name), f"{a['area_pct']:.2f}",
            f"{a['area_pct_con_fondo']:.2f}", f"{v['vol_frac_pct']:.2f}",
            fmt(d.get("n_total"), "d"),
            fmt(d.get("densidad_por_1000px2"), ".4f"),
            fmt(d.get("separacion_media_px"), ".1f"),
            fmt(d.get("indice_clark_evans"), ".2f"),
        ])
    rows.append(["Fondo", "-", f"{area_fracs['fondo']['area_pct']:.2f}",
                 "-", "-", "-", "-", "-"])
    rows.append(["Interfaces", "-",
                 f"{area_fracs['interfaces']['area_pct']:.2f}",
                 "-", "-", "-", "-", "-"])
    table = ax_tab.table(cellText=rows, colLabels=header, loc="center",
                         cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    notas = ["R = índice de Clark-Evans (≈1 al azar, <1 agrupadas, "
             ">1 espaciadas)."]
    if interfaces:
        pares = interfaces["pares_adyacentes"]
        if pares:
            key, p = next(iter(pares.items()))
            a, b = key.split("-")
            notas.append(f"Par con más interfaz: {phase_title(a)} / "
                         f"{phase_title(b)} ({p['fraccion']:.0%}, "
                         f"{p['longitud_px']:.0f} px).")
        if interfaces["contactos_misma_fase"] is not None:
            contactos = sum(c["longitud_px"] for c in
                            interfaces["contactos_misma_fase"].values())
            notas.append(f"Contactos entre partículas: {contactos:.0f} px.")
    ax_tab.text(0.5, 0.02, "\n".join(notas), ha="center", va="bottom",
                fontsize=9, transform=ax_tab.transAxes)
    ax_tab.set_title("Métricas por fase")

    fig.suptitle(f"Fracción de fases: {path.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura guardada en: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# Claves que debe traer el diccionario de pipeline_morphology
_MORPHOLOGY_KEYS: tuple[str, ...] = ("mask", "props", "filepath")


def pipeline_fraction(
    morphology_dict: dict,
    background: int | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Análisis de fracción completo de una imagen ya analizada.

    Secuencia: compute_area_fractions → estimate_volume_fractions →
    compute_number_density → compute_interface_fraction (con las
    líneas de watershed si vienen en la entrada) →
    show_fraction_results.

    Args:
        morphology_dict: Diccionario de pipeline_morphology (Etapa 3),
            con al menos 'mask', 'props' y 'filepath'.
        background: Etiqueta del fondo. None para la fase mayoritaria.
        output_dir: Carpeta donde guardar la figura. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra la figura con plt.show().

    Returns:
        El diccionario de entrada más 'area_fractions',
        'volume_fractions', 'number_density', 'interface_fraction' y
        'background' (etiqueta del fondo usada).

    Raises:
        ValueError: Si falta alguna clave en morphology_dict.
    """
    missing = [k for k in _MORPHOLOGY_KEYS if k not in morphology_dict]
    if missing:
        raise ValueError(
            f"Al diccionario de entrada le falta(n): {', '.join(missing)}."
            " Debe ser el resultado de pipeline_morphology()."
        )
    mask = morphology_dict["mask"]
    props = morphology_dict["props"]
    filepath = morphology_dict["filepath"]
    bg, _ = phase_names(mask, background)

    print("--- A1. Fracciones de área y volumen ---")
    area_fractions = compute_area_fractions(mask, bg)
    volume_fractions = estimate_volume_fractions(area_fractions)
    print("--- A2. Densidad numérica ---")
    density = compute_number_density(props, mask, bg)
    print("--- A3. Interfaces ---")
    interfaces = compute_interface_fraction(
        mask, bg, morphology_dict.get("lineas_watershed")
    )
    print("--- A4. Figura ---")
    show_fraction_results(mask, area_fractions, volume_fractions,
                          density, filepath, interfaces=interfaces,
                          props=props, output_dir=output_dir, show=show)

    print(f"\nResumen de fracción: {Path(filepath).name} (fondo = "
          f"etiqueta {bg}, {area_fractions['fondo']['area_pct']:.1f} %)")
    for name in [k for k in area_fractions if k.startswith("fase_")]:
        a, d = area_fractions[name], density[name]
        n_txt = "-" if d["n_total"] is None else str(d["n_total"])
        r_txt = ("-" if d["indice_clark_evans"] is None
                 else f"{d['indice_clark_evans']:.2f}")
        print(f"  {phase_title(name)}: {a['area_pct_con_fondo']:.2f} % de "
              f"la imagen ({a['area_pct']:.2f} % sin fondo) | "
              f"n = {n_txt} | R = {r_txt}")

    return {
        **morphology_dict,
        "area_fractions": area_fractions,
        "volume_fractions": volume_fractions,
        "number_density": density,
        "interface_fraction": interfaces,
        "background": bg,
    }
