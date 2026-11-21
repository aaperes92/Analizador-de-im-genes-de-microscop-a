"""Módulo SEM del Analizador de Imágenes de Microscopía."""

# Carga: lectura, metadata, barra de datos y ROI
from .carga import (
    cargar_imagen,
    guardar_roi,
    leer_metadata,
    mostrar_imagen_con_histograma,
    recortar_barra_datos,
    seleccionar_roi,
)

# Preprocesamiento: suavizado, sombras, ecualización y tanh
from .preprocesamiento import (
    corregir_sombras,
    ecualizar,
    explorar_preprocesamiento,
    normalizar_tanh,
    suavizar,
)

__all__ = [
    "cargar_imagen",
    "corregir_sombras",
    "ecualizar",
    "explorar_preprocesamiento",
    "guardar_roi",
    "leer_metadata",
    "mostrar_imagen_con_histograma",
    "normalizar_tanh",
    "recortar_barra_datos",
    "seleccionar_roi",
    "suavizar",
]
