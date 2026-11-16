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
]
