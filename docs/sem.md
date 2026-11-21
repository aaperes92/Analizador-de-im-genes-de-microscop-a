# Módulo SEM

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Mauricio Romano (mauricio.romano@ib.edu.ar).

El módulo `sem` trabaja con imágenes de microscopía electrónica de barrido (SEM):

1. **[Carga](#carga)**: lee la imagen, interpreta su metadata de adquisición, quita la barra
   de datos y permite seleccionar y guardar una región de interés (ROI).
   Código: [`src/sem/carga.py`](../src/sem/carga.py).
2. **[Preprocesamiento](#preprocesamiento)**: mejora la ROI antes de segmentarla (suavizado,
   corrección de sombras, ecualización de contraste y normalización tanh).
   Código: [`src/sem/preprocesamiento.py`](../src/sem/preprocesamiento.py).

Al final hay un [glosario](#glosario) con los términos técnicos.

---

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz del proyecto:

```bash
uv sync
```

Librerías que usa el módulo SEM:

| Librería | Para qué se usa |
|---|---|
| `numpy` | Arrays y cálculo numérico |
| `scikit-image` | Lectura de imágenes y todo el procesamiento (filtros, morfología, ecualización) |
| `tifffile` | Lectura de la metadata propietaria de los TIFF de FEI y Zeiss |
| `exifread` | Lectura de metadata EXIF estándar (respaldo) |
| `matplotlib` | Figuras y ventanas interactivas (selector de ROI, sliders) |

El módulo no usa OpenCV.

## Organización del código

```
src/sem/
    __init__.py           expone las funciones públicas del módulo
    carga.py              lectura, metadata, barra de datos y ROI
    preprocesamiento.py   suavizado, sombras, ecualización y tanh
scripts/
    sem_carga.py          script de línea de comandos para la carga
data/sem/roi/             ROIs guardadas (no se versionan en git)
```

Las funciones se pueden importar desde el paquete:

```python
import sys
sys.path.insert(0, "src")          # desde la raíz del proyecto

from sem import cargar_imagen, leer_metadata, recortar_barra_datos, suavizar
```

Todas las funciones tienen docstring con sus parámetros, retorno y errores: esa es la
referencia completa. Este documento explica qué hace cada una y cómo funciona.

Convenciones comunes a todo el módulo:

- Las imágenes se manejan como arrays 2D `float32`.
- Ninguna función modifica su entrada: siempre retornan un array nuevo.
- Las coordenadas de una ROI son `(x1, y1, x2, y2)` en píxeles, con `x` = columna e `y` = fila.

---

## Imágenes de entrada

### Formatos

Cualquier formato que lea `scikit-image` (`.tif`, `.png`, `.jpg`, ...). El formato esperado
es el `.tif` que guarda el microscopio, porque es el único que conserva la metadata.

Se aceptan imágenes en escala de grises (uint8, uint16, con o sin paleta) y RGB/RGBA, que se
convierten a escala de grises.

### Metadata de adquisición

La metadata de los microscopios SEM no está en los campos EXIF estándar sino en **tags TIFF
propietarios** de cada fabricante:

| Fabricante | Tag TIFF | Formato | Cómo se lee |
|---|---|---|---|
| FEI / Thermo Fisher | 34682 | texto tipo INI (`[Beam] HV=...`) | `tifffile` (`fei_metadata`) |
| Zeiss (SmartSEM) | 34118 | binario | `tifffile` (`sem_metadata`) |
| Otros | EXIF estándar | — | `exifread` (solo ancho, alto y fecha) |

El dato más importante es el **tamaño de píxel**: permite pasar de píxeles a unidades reales.
Una imagen sin metadata se puede procesar igual, pero las medidas quedan en píxeles.

### Barra de datos

Ambos fabricantes incrustan una barra de datos (escala, voltaje, magnificación) en las filas
inferiores de la imagen. Hay que quitarla antes de analizar, porque sus píxeles saturados
deforman el histograma.

| Fabricante | Barra | Cómo se detecta |
|---|---|---|
| FEI | fondo negro, texto blanco | La metadata indica las filas útiles (`[Image] ResolutionY`) |
| Zeiss | fondo blanco, texto negro | No figura en la metadata: se detecta en la imagen (ver [Recorte de la barra de datos](#recorte-de-la-barra-de-datos)) |

---

## Carga

### Uso

```bash
uv run python scripts/sem_carga.py <ruta_imagen> [--salida CARPETA] [--sufijo SUFIJO]
```

| Opción | Descripción | Por defecto |
|---|---|---|
| `imagen` | Ruta a la imagen SEM | (obligatorio) |
| `--salida` | Carpeta raíz donde se guardan las ROIs | `data/sem/roi` |
| `--sufijo` | Sufijo que identifica la ROI (ej. `_zona_grano`) | primer `_N` libre |

El script:

1. Carga la imagen, lee la metadata, quita la barra de datos e imprime un resumen.
2. Abre una ventana con la imagen y su histograma. Se cierra para continuar.
3. Abre el selector de ROI: se arrastra un rectángulo con el botón izquierdo (se puede ajustar)
   y se confirma con **Enter** o cerrando la ventana.
4. Guarda la ROI. Si no se seleccionó nada, no guarda y lo avisa.

### Salidas

Cada ROI se guarda en `<salida>/<nombre_imagen>/<nombre_imagen><sufijo>/`:

| Archivo | Contenido | Uso |
|---|---|---|
| `roi_<nombre><sufijo>.npy` | recorte `float32` en [0, 1] | entrada del preprocesamiento |
| `roi_<nombre><sufijo>.tif` | recorte `uint8` | solo para visualizar |
| `roi_<nombre><sufijo>.json` | imagen de origen, coordenadas, tamaños y metadata | calibración en unidades reales |

Sin `--sufijo` se usa el primer `_N` libre, así que nunca se sobrescribe una ROI anterior.

### Funcionamiento

#### Lectura y normalización — `cargar_imagen`

Lee la imagen con `skimage.io.imread`, convierte RGB/RGBA a escala de grises y la lleva a
`float32` en [0, 1] con `img_as_float32`.

La normalización **depende del tipo de dato, no del contenido**: uint8 se divide por 255 y
uint16 por 65535. No se estira el contraste. Así, dos imágenes del mismo equipo siguen siendo
comparables entre sí, y una imagen que no usa todo el rango del detector no llega a 1.0.

#### Metadata — `leer_metadata`

Intenta, en orden, FEI, Zeiss y EXIF, y retorna los valores con **claves y unidades comunes**
a todos los fabricantes:

| Clave | Unidad / formato |
|---|---|
| `fabricante` | `'FEI'`, `'Zeiss'` o `'EXIF'` |
| `ancho_px`, `alto_px` | píxeles (imagen completa, con barra) |
| `alto_util_px` | filas sin barra de datos (solo FEI) |
| `tamano_pixel_m` | metros por píxel |
| `ancho_campo_um` | campo de visión horizontal, µm |
| `magnificacion` | número (ej. `2020.0`) |
| `voltaje_kV` | kV |
| `distancia_trabajo_mm` | mm |
| `detector` | texto (ej. `'ETD'`, `'BSD4 A'`) |
| `fecha_adquisicion` | `'AAAA-MM-DD HH:MM:SS'` |
| `crudo` | diccionario original completo del fabricante |

Detalles de interpretación:

- **Tamaño de píxel en Zeiss.** Zeiss guarda dos valores: `ap_pixel_size`, referido a un
  barrido de 1024 px de ancho, y `ap_image_pixel_size`, referido a la imagen guardada. Se usa
  el segundo, que cumple `tamaño de píxel × ancho = campo de visión`. Usar el primero daría
  medidas 2 o 4 veces mayores en imágenes de 2048 o 4096 px.
- **Magnificación en FEI.** FEI no la guarda; se calcula como `MagCanvasRealWidth / HFW`
  (ancho de pantalla de referencia sobre campo de visión).
- **Voltaje.** FEI lo guarda en volts y Zeiss en kV; se unifica a kV.

Nunca lanza una excepción por metadata ausente o ilegible: retorna `{}` y avisa. También avisa
si no encontró el tamaño de píxel.

#### Recorte de la barra de datos — `recortar_barra_datos`

- **FEI:** se queda con las primeras `alto_util_px` filas.
- **Zeiss:** busca, en el 20 % inferior de la imagen, la primera fila completamente blanca
  cuya fila anterior sea mayormente no blanca: ese es el borde superior de la barra. Las filas
  de texto dentro de la barra siguen siendo mayormente blancas y no generan falsos bordes.
  Si no encuentra el borde, no recorta.

#### Visualización — `mostrar_imagen_con_histograma`

Figura con la imagen (escala de grises fija en [0, 1], sin estirar el contraste) y su
histograma de 256 niveles. Si hay tamaño de píxel, agrega una barra de escala de longitud
redonda (1, 2 o 5 × 10ⁿ µm) de alrededor del 20 % del ancho. Debajo muestra la metadata.
Retorna la figura; no la abre (eso lo hace `plt.show()`).

#### Selección de ROI — `seleccionar_roi`

Ventana con un `RectangleSelector` de matplotlib. Bloquea hasta Enter o hasta cerrar la
ventana, y retorna `(x1, y1, x2, y2)` ordenado y recortado a los bordes de la imagen, o `None`
si no se seleccionó nada. Requiere un backend de matplotlib con ventanas (por ejemplo TkAgg).

#### Guardado — `guardar_roi`

Recorta la imagen y escribe los tres archivos descritos en [Salidas](#salidas). El `.json` no
incluye la metadata `crudo` del fabricante.

---

## Preprocesamiento

Prepara la ROI para la segmentación. El orden de los pasos es:

```
ROI -> suavizar -> corregir_sombras -> ecualizar -> normalizar_tanh
```

```python
roi_suave       = suavizar(roi, sigma=1.0)
roi_sin_sombras = corregir_sombras(roi_suave, metodo="blur_resta", fraccion=0.03)
roi_ecualizada  = ecualizar(roi_sin_sombras, metodo="clahe", clip_limit=0.01)
roi_procesada   = normalizar_tanh(roi_ecualizada, factor=2.0)
```

### Funcionamiento

#### Suavizado — `suavizar`

Filtro gaussiano (`skimage.filters.gaussian`) para reducir el ruido de alta frecuencia.
`sigma` es el desvío del filtro en píxeles; debe ser bastante menor que las estructuras que se
quieren conservar. Como referencia, un filtro de caja de lado `k` equivale a una gaussiana de
`sigma ≈ k / √12`.

#### Corrección de sombras — `corregir_sombras`

Corrige la iluminación no uniforme. Hay dos métodos:

- **`blur_resta`**: estima el fondo con un gaussiano ancho y lo resta. Adecuado cuando la
  iluminación varía de forma gradual. El ancho del fondo es
  `sigma_fondo = fraccion × √(alto × ancho) / √12`, es decir, equivale a un filtro de caja de
  lado `fraccion × √(alto × ancho)`. La salida queda centrada en 0 (con valores negativos).
  Con `fraccion` chica actúa como un filtro pasa-altos fuerte: realza la textura fina pero
  iguala regiones grandes de distinto brillo.
- **`white_tophat`**: resta a la imagen su apertura morfológica con un disco de radio `radio`.
  Adecuado cuando el fondo tiene cambios bruscos. Se conserva lo que es **más chico que el
  disco** y se elimina lo más grande, así que el radio debe superar la estructura más grande
  que se quiere conservar. El disco se descompone en cruces (`decomposition="crosses"`), una
  aproximación muy cercana al disco exacto y mucho más rápida. La salida es ≥ 0.

#### Ecualización — `ecualizar`

Primero reescala la entrada a [0, 1] (acepta valores negativos, como la salida de
`blur_resta`) y después aplica uno de tres métodos:

- **`clahe`**: ecualización adaptativa por regiones con límite de recorte
  (`skimage.exposure.equalize_adapthist`). `clip_limit` controla cuánto se realza el contraste
  local. El rango útil es bajo: por encima de ~0.02 el recorte deja de actuar y el resultado
  ya no cambia.
- **`global`**: ecualización del histograma de toda la imagen.
- **`rescale`**: solo el reescalado a [0, 1].

La salida es `float32` en [0, 1].

#### Normalización tanh — `normalizar_tanh`

```
salida = tanh(factor × (x − mediana) / max|x − mediana|)
```

Centra la imagen en su mediana y aplica una tangente hiperbólica: comprime los extremos y
expande los tonos cercanos a la mediana. Con `factor` alto los valores se acercan a ±1 y la
imagen se aproxima a una binaria; con `factor` bajo el resultado es casi lineal.

**La salida está en (−1, 1), no en [0, 1].** La mediana queda en 0, así que el signo indica
si un píxel está por encima o por debajo de la intensidad típica.

#### Exploración interactiva — `explorar_preprocesamiento`

Ventana con 4 sliders (sigma, fracción de sombras o radio del tophat, `clip_limit` y factor
tanh) que muestra en tiempo real la ROI original, la corregida, la ecualizada, la normalizada
y sus histogramas. Cada slider recalcula solo desde su paso en adelante. Al cerrar imprime los
valores elegidos y los retorna en un diccionario.

---

## Glosario

| Término | Significado |
|---|---|
| **SEM** | Microscopía electrónica de barrido |
| **SE / BSE** | Detector de electrones secundarios (contraste topográfico) / retrodispersados (contraste por número atómico) |
| **ROI** | Región de interés: el recorte de la imagen que se analiza |
| **Tamaño de píxel** | Distancia real que representa un píxel (m/píxel) |
| **HFW** | Ancho horizontal del campo de visión |
| **WD** | Distancia de trabajo entre la lente y la muestra |
| **Barra de datos** | Franja inferior de la imagen con escala y datos de adquisición |
| **Histograma** | Cantidad de píxeles en cada nivel de intensidad |
| **CLAHE** | Ecualización adaptativa del histograma con límite de contraste |
| **Apertura / tophat** | Operaciones morfológicas: la apertura elimina estructuras brillantes más chicas que el disco; el tophat es la imagen menos su apertura |
| **Backend de matplotlib** | Motor que dibuja las figuras; uno "con ventanas" (TkAgg, QtAgg) es necesario para las herramientas interactivas |
