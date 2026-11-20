# Etapa 1 SEM: carga de imagen, metadata y selección de ROI

Esta etapa toma una imagen SEM (`.tif`), lee su metadata de adquisición, quita la barra de datos,
muestra un diagnóstico (imagen + histograma), permite seleccionar una región de interés (ROI) y la
guarda para las etapas siguientes (procesamiento y segmentación).

**Alcance:** solo imágenes SEM. TEM y MOKE se tratan en otros módulos.

---

## 1. Uso

Desde la raíz del repositorio, con el Python del entorno virtual:

```bash
.venv\Scripts\python.exe etapa1_carga.py imagenes/SEM/BSE/BSE_1.tif
```

(o activar antes el entorno con `.venv\Scripts\Activate.ps1` y usar `python`).

| Opción | Descripción | Por defecto |
|---|---|---|
| `imagen` | Ruta a la imagen SEM | (obligatorio) |
| `--salida` | Carpeta raíz donde se guardan los ROIs | `mascaras_roi` |
| `--sufijo` | Sufijo de la versión del ROI (ej. `_zona_grano`) | primer `_N` libre |

Pasos que ejecuta el script:

1. Carga la imagen, lee la metadata y recorta la barra de datos. Imprime un resumen en consola.
2. Abre la ventana de **imagen + histograma**. Cerrarla para continuar.
3. Abre el **selector de ROI**: arrastrar un rectángulo con el botón izquierdo (se puede ajustar),
   y presionar **Enter** o cerrar la ventana para confirmar.
4. Guarda el ROI e imprime la carpeta de salida. Si no se seleccionó nada, no guarda y lo avisa.

Las funciones también se pueden usar desde otro script:

```python
from pathlib import Path
from analizador_sem.carga import (
    cargar_imagen, leer_metadata, recortar_barra_datos,
    mostrar_imagen_con_histograma, seleccionar_roi, guardar_roi,
)

ruta = Path("imagenes/SEM/BSE/BSE_1.tif")
metadata = leer_metadata(ruta)
imagen = recortar_barra_datos(cargar_imagen(ruta), metadata)
```

---

## 2. Estructura

```
analizador_sem/
    __init__.py
    carga.py            # funciones de la Etapa 1
etapa1_carga.py         # script orquestador (línea de comandos)
requirements.txt        # dependencias
docs/etapa1_carga.md    # este documento
mascaras_roi/           # salida (ignorada por git)
```

Dependencias: `numpy`, `scipy`, `matplotlib`, `scikit-image`, `tifffile`, `exifread`.
**No se usa OpenCV (`cv2`)**: todo el procesamiento de imágenes va por `scikit-image`.

---

## 3. Funciones (`analizador_sem/carga.py`)

Todas tienen docstring y type hints. Ninguna modifica arrays in-place: siempre retornan arrays nuevos.

### `cargar_imagen(ruta) -> np.ndarray`
- Lee con `skimage.io.imread`.
- RGB/RGBA → escala de grises (`rgb2gray`).
- Convierte a **float32 en [0, 1]** con `img_as_float32`, que reescala según el rango del dtype
  (uint8 ÷ 255, uint16 ÷ 65535), **sin estirar el contraste**. Por eso una imagen FEI cuyo máximo
  es 65280 llega a 0.996 y no a 1.0: así las imágenes siguen siendo comparables entre sí.
- Lanza `FileNotFoundError` si el archivo no existe y `ValueError` si no es 2D ni RGB(A).

### `leer_metadata(ruta) -> dict`
Retorna la metadata de adquisición con **claves y unidades iguales para todos los fabricantes**:

| Clave | Unidad / formato |
|---|---|
| `fabricante` | `'FEI'`, `'Zeiss'` o `'EXIF'` |
| `ancho_px`, `alto_px` | píxeles (imagen completa) |
| `alto_util_px` | filas sin barra de datos (solo FEI) |
| `tamano_pixel_m` | metros por píxel |
| `ancho_campo_um` | campo de visión horizontal, µm |
| `magnificacion` | número (ej. `2020.0`) |
| `voltaje_kV` | kV |
| `distancia_trabajo_mm` | mm |
| `detector` | texto (ej. `'ETD'`, `'BSD4 A'`) |
| `fecha_adquisicion` | `'AAAA-MM-DD HH:MM:SS'` |
| `crudo` | diccionario original completo del fabricante |

Orden de lectura: FEI → Zeiss (ambos con `tifffile`) → EXIF estándar (`exifread`) como respaldo.
Nunca lanza excepción por metadata ausente: retorna `{}` e imprime un warning. También avisa si no
hay tamaño de píxel (las medidas quedarían en píxeles).

### `recortar_barra_datos(imagen, metadata) -> np.ndarray`
- **FEI:** recorta a `alto_util_px` (dato de la metadata).
- **Zeiss:** la metadata no indica la barra, así que se **detecta automáticamente** (ver sección 4.3).
- Sin información ni barra detectada: retorna una copia sin recortar.

### `mostrar_imagen_con_histograma(imagen, titulo, metadata) -> plt.Figure`
- Izquierda: imagen (`cmap='gray'`, `vmin=0`, `vmax=1`) con **barra de escala en µm** calculada con
  el tamaño de píxel (longitud redonda 1/2/5 × 10ⁿ de ~20% del ancho).
- Derecha: histograma de 256 bins en [0, 1].
- Título con nombre, shape, dtype y min/max; debajo, una línea con la metadata.
- Retorna la figura (se puede guardar con `fig.savefig`). No llama a `plt.show()`: lo hace el script.

### `seleccionar_roi(imagen, titulo) -> tuple | None`
- Ventana con `RectangleSelector`; bloquea hasta Enter o cerrar la ventana.
- Retorna `(x1, y1, x2, y2)` ordenado (x1 < x2, y1 < y2) y recortado a los bordes de la imagen,
  o `None` si no se seleccionó nada. Los clics sin área se ignoran.
- Lanza `RuntimeError` si matplotlib está en un backend sin ventanas (ej. `Agg`).

### `guardar_roi(imagen, nombre_archivo, coordenadas, ruta_mascaras, metadata, sufijo) -> Path`
Guarda en `ruta_mascaras/<nombre>/<nombre><sufijo>/`:

| Archivo | Contenido | Uso |
|---|---|---|
| `roi_<nombre><sufijo>.npy` | recorte float32 en [0, 1] | entrada de las etapas siguientes |
| `roi_<nombre><sufijo>.tif` | recorte uint8 | solo para visualizar |
| `roi_<nombre><sufijo>.json` | origen, coordenadas, shapes y metadata (sin `crudo`) | calibración en unidades reales |

- Sin `sufijo`, usa el primer `_N` libre: **nunca sobrescribe** un ROI anterior.
- Lanza `ValueError` si las coordenadas no forman un rectángulo válido dentro de la imagen, y
  `FileExistsError` si se pasa un sufijo explícito que ya existe.

---

## 4. Hallazgos sobre las imágenes SEM

### 4.1 La metadata no es EXIF estándar
Las imágenes no traen los campos EXIF habituales (exposición, fecha, magnificación). La información
está en **tags TIFF propietarios** de cada fabricante:

| Equipo | Tag TIFF | Formato | Lector |
|---|---|---|---|
| FEI/Thermo (Inspect S, detector ETD) | 34682 (`0x877A`) | texto tipo INI (`[Beam] HV=...`) | `tifffile.fei_metadata` |
| Zeiss SmartSEM (Crossbeam 340, BSD4) | 34118 (`0x8546`) | binario | `tifffile.sem_metadata` |

`exifread` muestra esos tags como datos crudos sin interpretar; por eso se usa `tifffile` (ya
instalado como backend de scikit-image) y `exifread` queda como respaldo.

### 4.2 Tamaño de píxel en Zeiss
Zeiss guarda dos valores:
- `ap_pixel_size` ("Pixel Size"): referido a un barrido de 1024 px de ancho. **No se usa.**
- `ap_image_pixel_size` ("Image Pixel Size"): el de la imagen guardada. **Es el correcto.**

Verificación: `tamaño de píxel × ancho = campo de visión` coincide con error ≤ 0,4 % en las 12
imágenes. Usar `ap_pixel_size` habría duplicado (2048 px) o cuadruplicado (4096 px) todas las medidas.

En FEI la magnificación no se guarda directamente; se calcula como `MagCanvasRealWidth / HFW`.

### 4.3 Barra de datos incrustada
Ambos equipos incrustan la barra de datos (escala, kV, magnificación) en las filas inferiores de la
imagen. Si no se quita, sus píxeles saturados deforman el histograma y el umbral de segmentación.

| Equipo | Barra | Cómo se recorta |
|---|---|---|
| FEI | negra con texto blanco; 59 filas en 943 (119 en 1887) | `[Image] ResolutionY` de la metadata |
| Zeiss | blanca con texto negro; 138 filas en 1536, 280 en 3072 (~9 %) | detección automática |

Detección Zeiss: en el 20 % inferior de la imagen se busca la primera fila 100 % blanca cuya fila
anterior sea mayormente no blanca (el borde superior de la barra). Las filas de texto dentro de la
barra siguen siendo mayormente blancas y no generan falsos bordes. Si no se encuentra el borde, no
se recorta nada.

### 4.4 Formatos heterogéneos
Las imágenes llegan como uint8 con paleta (Zeiss), uint16 (FEI) y una RGB. `cargar_imagen` las
lleva a todas a float32 2D en [0, 1].

---

## 5. Cambios respecto del notebook anterior (`backup-codigo-viejo.ipynb`)

| Antes | Ahora |
|---|---|
| `cv2.imread` | `skimage.io.imread` + `img_as_float32` |
| Sin metadata | `leer_metadata` (FEI, Zeiss, EXIF) |
| La barra de datos quedaba en la imagen | `recortar_barra_datos` |
| Sin histograma | `mostrar_imagen_con_histograma` |
| `%matplotlib widget` + diccionario global de coordenadas | `seleccionar_roi` bloqueante que retorna las coordenadas |
| Celdas sueltas de guardado; `nombre_archivo.stem` sobre un `str` (error) | `guardar_roi` |
| `Image.fromarray` sin importar PIL (error) | `skimage.io.imsave` |
| `sufijo = "_1"` a mano | sufijo automático sin sobrescribir |
| Solo `.npy` (sin calibración) | `.npy` + `.tif` + `.json` con tamaño de píxel |
| Nombres de imagen fijos en el código | ruta por línea de comandos |

Los notebooks no se modificaron.

---

## 6. Verificación realizada

- `cargar_imagen`: las 12 `.tif` SEM cargan como float32 2D en [0, 1]; ruta inexistente → `FileNotFoundError`.
- `leer_metadata`: las 12 imágenes dan fabricante, tamaño de píxel, kV, WD, detector y fecha;
  PNG sin metadata → `{}` con warning.
- `recortar_barra_datos`: FEI 943 → 884 filas; Zeiss 1536 → 1398 y 3072 → 2792; sin barra no recorta.
- `mostrar_imagen_con_histograma`: figuras generadas para las 12 imágenes sin warnings; revisión
  visual de una Zeiss y una FEI (escala correcta).
- `seleccionar_roi`: probado con arrastres simulados (normal, invertido, fuera de límites, clic sin área).
- `guardar_roi`: archivos correctos, recorte idéntico al original, sufijo automático, errores esperados.
- `etapa1_carga.py`: ejecución completa con la selección simulada.

**Pendiente:** probar las ventanas interactivas con el mouse en uso real.
