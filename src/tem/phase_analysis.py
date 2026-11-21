"""Pipeline maestro de la Etapa 4: análisis de fases.

Encadena los tres módulos de la Etapa 4 del módulo TEM del proyecto
"Analizador de Imágenes de Microscopía": fracción (fraction.py),
composicional relativo (compositional.py) y espacial (spatial.py). Las
fases se llaman siempre "Fase 1", "Fase 2", ... y nunca se asignan
elementos químicos.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

from pathlib import Path

from .compositional import pipeline_compositional
from .fraction import phase_title, pipeline_fraction
from .spatial import pipeline_spatial


def _print_report(result: dict) -> None:
    """Imprime el reporte final de la Etapa 4.

    Args:
        result: Resultado de pipeline_spatial (con todo acumulado).
    """
    areas = result["area_fractions"]
    volumes = result["volume_fractions"]
    phases = [k for k in areas if k.startswith("fase_")]
    line = "=" * 72
    print(f"\n{line}\nREPORTE DE LA ETAPA 4: {Path(result['filepath']).name}"
          f"\n{line}")
    print(f"Fondo: etiqueta {result['background']} "
          f"({areas['fondo']['area_pct']:.2f} % de la imagen); "
          f"interfaces {areas['interfaces']['area_pct']:.2f} %")

    print("\n1. Fracciones de área y volumen estimado (Delesse)")
    print(f"   {'fase':8} {'área % total':>12} {'área % sin fondo':>16} "
          f"{'vol. % est. total':>17} {'sin fondo':>9}")
    for name in phases:
        a, v = areas[name], volumes[name]
        print(f"   {phase_title(name):8} {a['area_pct_con_fondo']:>12.2f} "
              f"{a['area_pct']:>16.2f} {v['vol_frac_pct_con_fondo']:>17.2f} "
              f"{v['vol_frac_pct']:>9.2f}")
    print("   (volumen estimado suponiendo isotropía y un corte plano; una "
          "imagen TEM es una proyección)")

    print("\n2. Distribución espacial")
    print(f"   {'fase':8} {'n':>4} {'RDF':>12} {'densidad':>30} "
          f"{'Hopkins':>14} {'grupos':>6}")
    for name in phases:
        r = result["rdf"][name]
        d = result["density_map"][name]
        c = result["clustering"][name]
        h = ("-" if c["hopkins_statistic"] is None
             else f"{c['hopkins_statistic']:.2f}")
        groups = ("-" if c["n_clusters_dbscan"] is None
                  else c["n_clusters_dbscan"])
        print(f"   {phase_title(name):8} {r['n']:>4} {r['clasificacion']:>12}"
              f" {d['clasificacion']:>30} "
              f"{h + ' ' + c['hopkins_clasificacion']:>14} {groups:>6}")

    print("\n3. Inhomogeneidades (zonas más claras / más oscuras que su "
          "fase)")
    for name, d in result["inhomogeneities"].items():
        title = phase_title(name).replace("fondo", "Fondo")
        print(f"   {title:8} {d['n_alta']} clara(s), {d['n_baja']} "
              f"oscura(s) — {d['area_anomala_pct']:.2f} % del área")

    print("\n4. Gradientes espaciales significativos (Bonferroni)")
    found = False
    for name in phases:
        for text in result["spatial_gradients"][name][
                "gradientes_significativos"]:
            print(f"   {phase_title(name)}: {text}")
            found = True
    if not found:
        print("   Ninguno.")
    print(line)


def pipeline_phase_analysis(
    morphology_dict: dict,
    background: int | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Análisis de fases completo: fracción, composicional y espacial.

    Secuencia: pipeline_fraction → pipeline_compositional →
    pipeline_spatial, y al final un reporte con las fracciones de área
    y volumen, la distribución espacial, las inhomogeneidades y los
    gradientes significativos de cada fase.

    Args:
        morphology_dict: Diccionario de pipeline_morphology (Etapa 3).
        background: Etiqueta del fondo. None para la fase mayoritaria.
            Los Módulos B y C usan el mismo fondo que el A.
        output_dir: Carpeta donde guardar las figuras. Por defecto,
            data/samples/ en la raíz del proyecto.
        show: Si es True, muestra las figuras con plt.show().

    Returns:
        El diccionario de pipeline_spatial, que acumula todo lo de las
        Etapas 1 a 4.
    """
    print("=" * 72 + "\nETAPA 4 — Módulo A: fracción de fases\n" + "=" * 72)
    fraction = pipeline_fraction(morphology_dict, background=background,
                                 output_dir=output_dir, show=show)
    print("=" * 72 + "\nETAPA 4 — Módulo B: análisis composicional relativo"
          "\n" + "=" * 72)
    compositional = pipeline_compositional(fraction, output_dir=output_dir,
                                           show=show)
    print("=" * 72 + "\nETAPA 4 — Módulo C: análisis espacial\n" + "=" * 72)
    spatial = pipeline_spatial(compositional, output_dir=output_dir,
                               show=show)
    _print_report(spatial)
    return spatial
