"""Análisis TEM completo: preprocesamiento, segmentación y morfología.

Corre las tres etapas del módulo TEM sobre una imagen o sobre todas las
imágenes de una carpeta.

Cómo usarlo (desde la raíz del proyecto):

    1. Editá la sección CONFIGURACIÓN de abajo (la ruta y las opciones).
    2. Corré:
           uv run python scripts/analizar_tem.py
       o pasá la ruta directamente, sin editar el archivo:
           uv run python scripts/analizar_tem.py "C:\\ruta\\imagen.dm3"

Con una imagen: muestra en pantalla las figuras de cada etapa (cerrá
cada ventana para pasar a la siguiente) e imprime todos los
resultados.
Con una carpeta: procesa todas las imágenes sin abrir ventanas, imprime
una tabla y guarda un resumen en CSV.

Todas las figuras se guardan en data/samples/ (o en CARPETA_SALIDA).

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import csv
import io
import sys
import time
from pathlib import Path

# ====================================================================
# CONFIGURACIÓN — editá estos valores
# ====================================================================

# Imagen o carpeta a analizar (.dm3, .mrc, .tif, .tiff, .png, .jpg)
RUTA = (
    r"C:\Users\Daniel\Downloads\TEM-20261116T174039Z-1-001\TEM"
    r"\campo-claro-HR\prueba1.dm3"
)

# Etapa 1 — filtro de ruido:
#   'auto', 'gaussian', 'median', 'bilateral' o 'wiener'
FILTRO = "auto"

# Etapa 2 — método de umbralización:
#   'auto', 'otsu', 'gmm', 'triangle' (campo oscuro) o 'adaptive'
UMBRAL = "auto"

# Etapa 3 — fases a medir: None = todas menos la mayoritaria (fondo).
#   Ejemplos: [0] o [1, 2]
FASES = None

# Mostrar las figuras en pantalla (solo con una imagen)
MOSTRAR_FIGURAS = True

# Carpeta donde guardar figuras y CSV: None = data/samples/
CARPETA_SALIDA = None

# ====================================================================
# (No hace falta tocar nada de acá para abajo)
# ====================================================================

# Permite importar el paquete tem desde src/
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tem import (  # noqa: E402
    pipeline_morphology,
    pipeline_segmentation,
    pipeline_tem,
)
from tem.loader import DEFAULT_OUTPUT_DIR, SUPPORTED_FORMATS  # noqa: E402


def analizar(path: Path, mostrar: bool) -> dict:
    """Corre las tres etapas sobre una imagen.

    Args:
        path: Ruta a la imagen.
        mostrar: Si es True, muestra las figuras y todos los mensajes.

    Returns:
        Diccionario con los resultados de cada etapa ('etapa1',
        'etapa2', 'etapa3') y el tiempo total en segundos ('segundos').
    """
    inicio = time.perf_counter()
    salida = sys.stdout if mostrar else io.StringIO()
    with contextlib.redirect_stdout(salida):
        print("\n" + "=" * 70)
        print(f"ETAPA 1 — Carga y preprocesamiento: {path.name}")
        print("=" * 70)
        etapa1 = pipeline_tem(path, method=FILTRO,
                              output_dir=CARPETA_SALIDA, show=mostrar)

        print("\n" + "=" * 70)
        print("ETAPA 2 — Segmentación de fases")
        print("=" * 70)
        etapa2 = pipeline_segmentation(etapa1, method=UMBRAL,
                                       output_dir=CARPETA_SALIDA,
                                       show=mostrar)

        print("\n" + "=" * 70)
        print("ETAPA 3 — Análisis morfológico")
        print("=" * 70)
        etapa3 = pipeline_morphology(etapa2, phases=FASES,
                                     output_dir=CARPETA_SALIDA,
                                     show=mostrar)
    return {"etapa1": etapa1, "etapa2": etapa2, "etapa3": etapa3,
            "segundos": time.perf_counter() - inicio}


def filas_resumen(path: Path, r: dict) -> list[dict]:
    """Arma una fila de resumen por fase analizada.

    Args:
        path: Ruta a la imagen.
        r: Resultado de analizar().

    Returns:
        Lista de diccionarios (una fila por fase de la Etapa 3).
    """
    e2, e3 = r["etapa2"], r["etapa3"]
    filas = []
    for fase, size in e3["size_stats"].items():
        orient = e3["orientation_stats"][fase]
        ajuste = size["ajuste_lognormal"]
        filas.append({
            "archivo": path.name,
            "filtro": r["etapa1"]["method_used"],
            "n_fases": e2["n_phases"],
            "umbral": e2["threshold_method"],
            "fase": fase,
            "area_fase_pct": round(e2["stats"][fase]["area_pct"], 2),
            "n_regiones": size["n_regiones"],
            "area_media_px": _redondear(size["area_media_px"]),
            "area_mediana_px": _redondear(size["area_mediana_px"]),
            "lognormal_p": (_redondear(ajuste["ks_pvalue"], 3)
                            if ajuste else ""),
            "forma": e3["shape_stats"][fase]["forma_predominante"] or "",
            "indice_S": _redondear(orient["indice_orden_S"], 2),
            "orientacion": orient["clasificacion"] or "",
            "segundos": round(r["segundos"], 1),
        })
    return filas


def _redondear(valor: float | None,
               decimales: int = 0) -> int | float | str:
    """Redondea un valor (entero si decimales es 0) o devuelve '' si es
    None."""
    if valor is None:
        return ""
    return int(round(valor)) if decimales == 0 else round(valor, decimales)


def imprimir_tabla(filas: list[dict]) -> None:
    """Imprime el resumen como tabla.

    Args:
        filas: Filas de filas_resumen().
    """
    print(f"\n{'archivo':36} {'umbral':>8} {'fase':7} {'área%':>6} "
          f"{'n':>4} {'media px':>9} {'forma':>10} {'S':>5}  orientación")
    for f in filas:
        print(f"{f['archivo']:36} {f['umbral']:>8} {f['fase']:7} "
              f"{f['area_fase_pct']:>6} {f['n_regiones']:>4} "
              f"{str(f['area_media_px']):>9} {f['forma']:>10} "
              f"{str(f['indice_S']):>5}  {f['orientacion']}")


def guardar_csv(filas: list[dict], nombre: str) -> Path:
    """Guarda el resumen en un CSV (se abre bien con Excel).

    Args:
        filas: Filas de filas_resumen().
        nombre: Nombre base del archivo.

    Returns:
        Ruta del CSV.
    """
    carpeta = Path(CARPETA_SALIDA) if CARPETA_SALIDA else DEFAULT_OUTPUT_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{nombre}_resumen_tem.csv"
    # utf-8-sig, ';' y coma decimal para que Excel en español lo abra
    # con los números como números
    con_coma = [
        {k: str(v).replace(".", ",") if isinstance(v, float) else v
         for k, v in fila.items()}
        for fila in filas
    ]
    with destino.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0]),
                                delimiter=";")
        writer.writeheader()
        writer.writerows(con_coma)
    return destino


def main() -> None:
    """Analiza la imagen o la carpeta configurada."""
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(RUTA)
    if not ruta.exists():
        print(f"No existe: {ruta}\nRevisá RUTA en la configuración.")
        sys.exit(1)

    if ruta.is_file():
        resultado = analizar(ruta, mostrar=MOSTRAR_FIGURAS)
        filas = filas_resumen(ruta, resultado)
        print("\n" + "=" * 70)
        print(f"RESUMEN ({resultado['segundos']:.1f} s)")
        print("=" * 70)
        imprimir_tabla(filas)
        if filas:
            print(f"\nCSV: {guardar_csv(filas, ruta.stem)}")
        print(f"Figuras: {CARPETA_SALIDA or DEFAULT_OUTPUT_DIR}")
        return

    imagenes = sorted(p for p in ruta.iterdir()
                      if p.suffix.lower() in SUPPORTED_FORMATS)
    if not imagenes:
        print(f"No hay imágenes soportadas en {ruta}")
        sys.exit(1)
    print(f"Analizando {len(imagenes)} imágenes de {ruta} "
          "(sin abrir ventanas)...")
    filas: list[dict] = []
    for i, imagen in enumerate(imagenes, start=1):
        print(f"  [{i}/{len(imagenes)}] {imagen.name}", end="", flush=True)
        try:
            resultado = analizar(imagen, mostrar=False)
        except Exception as error:  # una imagen mala no frena el resto
            print(f"  → ERROR: {error}")
            continue
        filas.extend(filas_resumen(imagen, resultado))
        print(f"  ({resultado['segundos']:.1f} s)")
    imprimir_tabla(filas)
    if filas:
        print(f"\nCSV: {guardar_csv(filas, ruta.name)}")
    print(f"Figuras: {CARPETA_SALIDA or DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
