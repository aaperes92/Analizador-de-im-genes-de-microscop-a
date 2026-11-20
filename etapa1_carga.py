"""
Etapa 1 del pipeline SEM: carga de imagen, metadata, visualización, selección y guardado del ROI.

Uso:
    python etapa1_carga.py <ruta_imagen> [--salida CARPETA] [--sufijo SUFIJO]

Ejemplo:
    python etapa1_carga.py imagenes/SEM/BSE/BSE_1.tif
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from analizador_sem.carga import (
    cargar_imagen,
    guardar_roi,
    leer_metadata,
    mostrar_imagen_con_histograma,
    recortar_barra_datos,
    seleccionar_roi,
)


def main() -> int:
    """Ejecuta la Etapa 1 sobre la imagen pasada por línea de comandos. Retorna el código de salida."""
    parser = argparse.ArgumentParser(description="Etapa 1 SEM: carga, metadata y selección de ROI.")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen SEM (.tif)")
    parser.add_argument(
        "--salida", type=Path, default=Path("mascaras_roi"),
        help="Carpeta raíz donde guardar los ROIs (por defecto: mascaras_roi)",
    )
    parser.add_argument(
        "--sufijo", default=None,
        help='Sufijo del ROI, ej. "_1". Por defecto se usa el primer "_N" libre.',
    )
    args = parser.parse_args()
    # Evita errores de codificación al imprimir acentos/emojis en consolas Windows redirigidas.
    sys.stdout.reconfigure(errors="replace")

    # 1. Cargar imagen, leer metadata y quitar la barra de datos.
    try:
        imagen_completa = cargar_imagen(args.imagen)
    except FileNotFoundError as error:
        print(f"❌ {error}")
        return 1
    metadata = leer_metadata(args.imagen)
    imagen = recortar_barra_datos(imagen_completa, metadata)

    print(f"Imagen: {args.imagen.name}")
    print(f"  Fabricante: {metadata.get('fabricante', 'desconocido')}")
    print(f"  Shape original: {imagen_completa.shape} -> sin barra de datos: {imagen.shape}")
    if "tamano_pixel_m" in metadata:
        print(f"  Tamaño de píxel: {metadata['tamano_pixel_m'] * 1e9:.3g} nm")

    # 2. Visualización diagnóstica (cerrar la ventana para continuar).
    mostrar_imagen_con_histograma(imagen, titulo=args.imagen.name, metadata=metadata)
    print("Cerrá la ventana del histograma para pasar a la selección del ROI...")
    plt.show()

    # 3. Selección interactiva del ROI.
    print("Seleccioná el ROI y presioná Enter (o cerrá la ventana) para confirmar.")
    coordenadas = seleccionar_roi(imagen, titulo=args.imagen.name)
    if coordenadas is None:
        print("⚠️ No se seleccionó ningún ROI; no se guardó nada.")
        return 0
    print(f"  ROI (x1, y1, x2, y2): {coordenadas}")

    # 4. Guardado.
    try:
        carpeta = guardar_roi(
            imagen, args.imagen.name, coordenadas, args.salida, metadata, sufijo=args.sufijo
        )
    except (ValueError, FileExistsError) as error:
        print(f"❌ {error}")
        return 1
    print(f"✅ ROI guardada en: {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
