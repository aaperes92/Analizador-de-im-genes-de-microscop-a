"""Módulo TEM del Analizador de Imágenes de Microscopía."""

# Etapa 1: carga y preprocesamiento
from .loader import (
    load_tem_image,
    pipeline_tem,
    preprocess_tem_image,
    show_preprocessing_results,
)

# Etapa 2: segmentación de fases
from .segmentation import (
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
)

# Etapa 3: análisis morfológico
from .morphology import (
    analyze_orientation,
    analyze_shape,
    analyze_size_distribution,
    extract_region_properties,
    pipeline_morphology,
    plot_orientation_rose,
    show_morphology_results,
)

# Etapa 4: análisis de fases (fracción, composicional y espacial)
from .fraction import (
    compute_area_fractions,
    compute_interface_fraction,
    compute_number_density,
    estimate_volume_fractions,
    pipeline_fraction,
    show_fraction_results,
)
from .compositional import (
    build_phase_intensity_map,
    compute_intensity_gradients,
    compute_intensity_profiles,
    detect_inhomogeneities,
    pipeline_compositional,
    show_compositional_results,
)
from .spatial import (
    compute_density_map,
    compute_rdf,
    detect_clustering,
    detect_spatial_gradients,
    pipeline_spatial,
    show_spatial_results,
)
from .phase_analysis import pipeline_phase_analysis

__all__ = [
    # Etapa 1
    "load_tem_image",
    "preprocess_tem_image",
    "show_preprocessing_results",
    "pipeline_tem",
    # Etapa 2
    "analyze_histogram",
    "detect_n_phases",
    "select_threshold_method",
    "apply_segmentation",
    "clean_mask",
    "separate_touching_regions",
    "detect_interfaces",
    "compute_phase_stats",
    "show_segmentation_results",
    "pipeline_segmentation",
    # Etapa 3
    "extract_region_properties",
    "analyze_size_distribution",
    "analyze_shape",
    "analyze_orientation",
    "plot_orientation_rose",
    "show_morphology_results",
    "pipeline_morphology",
    # Etapa 4 — Módulo A: fracción
    "compute_area_fractions",
    "estimate_volume_fractions",
    "compute_number_density",
    "compute_interface_fraction",
    "show_fraction_results",
    "pipeline_fraction",
    # Etapa 4 — Módulo B: composicional
    "compute_intensity_profiles",
    "compute_intensity_gradients",
    "build_phase_intensity_map",
    "detect_inhomogeneities",
    "show_compositional_results",
    "pipeline_compositional",
    # Etapa 4 — Módulo C: espacial
    "compute_rdf",
    "compute_density_map",
    "detect_clustering",
    "detect_spatial_gradients",
    "show_spatial_results",
    "pipeline_spatial",
    # Etapa 4 — pipeline maestro
    "pipeline_phase_analysis",
]
