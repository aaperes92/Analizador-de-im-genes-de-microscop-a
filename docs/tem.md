# Módulo TEM

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Daniel Romero (daniel.romero@ib.edu.ar).

El módulo `tem` trabaja con imágenes de microscopía electrónica de
transmisión (TEM) en cuatro etapas:

1. **[Etapa 1 — Carga y preprocesamiento](#etapa-1--carga-y-preprocesamiento)**:
   lee la imagen, la normaliza a [0, 1] y reduce el ruido.
   Código: [`src/tem/loader.py`](../src/tem/loader.py).
2. **[Etapa 2 — Segmentación de fases](#etapa-2--segmentación-de-fases)**:
   detecta cuántas fases de intensidad hay, las separa, limpia las
   regiones y calcula estadísticas por fase.
   Código: [`src/tem/segmentation.py`](../src/tem/segmentation.py).
3. **[Etapa 3 — Análisis morfológico](#etapa-3--análisis-morfológico)**:
   mide tamaño, forma y orientación de cada región segmentada.
   Código: [`src/tem/morphology.py`](../src/tem/morphology.py).
4. **[Etapa 4 — Análisis de fases](#etapa-4--análisis-de-fases)**:
   fracciones de área y volumen, densidad, interfaces, intensidad
   relativa por fase y distribución espacial de las partículas.
   Código: [`src/tem/fraction.py`](../src/tem/fraction.py),
   [`compositional.py`](../src/tem/compositional.py),
   [`spatial.py`](../src/tem/spatial.py) y
   [`phase_analysis.py`](../src/tem/phase_analysis.py).

Tests: [`tests/tem/`](../tests/tem/) (153 tests, `uv run pytest`).

**Análisis completo en un paso**: [`scripts/analizar_tem.py`](../scripts/analizar_tem.py)
corre las cuatro etapas sobre una imagen o una carpeta, muestra las
8 figuras y guarda un resumen en CSV con las métricas de todas las
etapas. Se configura editando la sección CONFIGURACIÓN del archivo
(ruta, filtro, umbral, fases y `FONDO`) o pasando la ruta:

```bash
uv run python scripts/analizar_tem.py "RUTA\imagen.dm3"
```

---

# Etapa 1 — Carga y preprocesamiento

## Instalación

El proyecto usa [uv](https://docs.astral.sh/uv/). Desde la raíz:

```bash
uv sync
```

Esto crea `.venv` con Python 3.14 e instala las dependencias de
`uv.lock`, entre ellas `ncempy` (para `.dm3`) y `mrcfile` (para `.mrc`).

## Uso rápido

Desde un notebook en `notebooks/`:

```python
import sys
sys.path.insert(0, "../src")  # para poder importar el módulo tem

from tem import pipeline_tem

resultado = pipeline_tem("ruta/a/imagen.dm3")
resultado["method_used"]  # filtro aplicado, p. ej. 'gaussian'
```

`pipeline_tem` carga la imagen, la preprocesa, muestra la figura de
comparación y la guarda en `data/samples/<nombre>_preprocessing.png`.

Paso a paso, con control de cada etapa:

```python
from tem import (
    load_tem_image, preprocess_tem_image, show_preprocessing_results,
)

original = load_tem_image("imagen.dm3")
procesada = preprocess_tem_image(original, method="median")
show_preprocessing_results(original, procesada, "imagen.dm3", show=False)
```

Correr los tests:

```bash
uv run pytest
```

---

## Formatos soportados

| Extensión | Librería | Notas |
|---|---|---|
| `.tif`, `.tiff`, `.jpg`, `.png` | scikit-image | Color (RGB/RGBA) → escala de grises |
| `.dm3` | ncempy | Formato de Gatan DigitalMicrograph |
| `.mrc` | mrcfile | Se abre en modo `permissive` (tolera encabezados irregulares) |

- Las extensiones se aceptan en mayúscula o minúscula.
- Si el archivo trae una **pila** de cuadros (3D), se usa el primero y
  se avisa con un mensaje.
- Si el archivo no existe se lanza `FileNotFoundError`; si la extensión
  no está soportada, `ValueError` con la lista de formatos válidos.

## Normalización por percentiles

Toda imagen se lleva a **float32 en [0, 1]**. En lugar de usar el
mínimo y el máximo, se toman los **percentiles 0.1 y 99.9** como 0 y 1
y se recorta lo que queda afuera.

**Por qué:** unos pocos píxeles extremos (píxeles calientes, bordes de
la rejilla) estiran la escala y comprimen el contraste del resto. En
las imágenes de prueba, `prueba6.dm3` tenía píxeles extremos que
estiraban su rango 4.8 veces: con mínimo y máximo parecía la imagen más
limpia del grupo cuando en realidad era la más ruidosa (tiene ~3.5 veces
menos cuentas que las demás).

Una imagen constante devuelve un array de ceros.

## Filtros

Los parámetros por defecto están pensados para imágenes normalizadas a
[0, 1].

| Método | Implementación | Parámetros | Cuándo conviene |
|---|---|---|---|
| `gaussian` | `skimage.filters.gaussian` | σ = 1.0 | Ruido bajo. Conserva franjas de red en imágenes HR. |
| `median` | `skimage.filters.median` | disco de radio 2 | Ruido impulsivo (sal y pimienta, píxeles calientes). |
| `bilateral` | `skimage.restoration.denoise_bilateral` | `sigma_color` = 0.05, `sigma_spatial` = 2 | Ruido moderado; conserva bordes de partículas. |
| `wiener` | `scipy.signal.wiener` | ventana 5×5 | Ruido alto; se adapta a la varianza local. |
| `auto` | — | — | Elige uno de los anteriores (ver abajo). |

El resultado siempre se recorta a [0, 1] y se devuelve en float32 con
el mismo shape que la entrada.

> **Atención con imágenes HR:** en `prueba13.dm3` se ven franjas de red
> que `gaussian` y `median` conservan, pero **`bilateral` borra por
> completo**. En imágenes de alta resolución conviene revisar qué filtro
> se aplicó.

## Modo `auto`

Decide en dos pasos:

1. **Ruido impulsivo.** Cuenta los píxeles que están en un extremo
   (≤ 0.02 o ≥ 0.98) **y** difieren en más de 0.3 de la mediana de sus
   vecinos. Si superan el 0.2 % de la imagen → `median`.
2. **Nivel de ruido.** Calcula la dispersión robusta (MAD × 1.4826) de
   los gradientes Sobel horizontal y vertical. Se usa la MAD y no la
   varianza porque los bordes de las partículas ocupan pocos píxeles y
   casi no la afectan: mide sobre todo el ruido.

| Dispersión de los gradientes | Filtro |
|---|---|
| < 0.04 | `gaussian` |
| 0.04 – 0.09 | `bilateral` |
| > 0.09 | `wiener` |

Siempre imprime el valor medido y la justificación de la elección.

### Calibración

Los umbrales se ajustaron con imágenes sintéticas (partículas + ruido
gaussiano) y con **13 imágenes reales de campo claro HR** (2048×2048,
`.dm3`):

| Imágenes | Dispersión | Filtro elegido |
|---|---|---|
| 12 de 13 | 0.010 – 0.022 | `gaussian` |
| `prueba6.dm3` (poca señal) | 0.052 | `bilateral` |

Ninguna mostró ruido impulsivo (máximo 0.0024 % de píxeles, umbral
0.2 %).

---

## Referencia de funciones

### `load_tem_image(filepath) -> np.ndarray`

Carga una imagen y la normaliza a [0, 1].

- **filepath** (`str | Path`): ruta al archivo.
- **Devuelve:** array 2D float32 en [0, 1].
- **Imprime:** nombre del archivo, formato, shape y dtype.
- **Errores:** `FileNotFoundError`, `ValueError`.

### `preprocess_tem_image(image, method='auto') -> np.ndarray`

Reduce el ruido con el filtro indicado.

- **image** (`np.ndarray`): imagen 2D en [0, 1].
- **method** (`str`): `'gaussian'`, `'median'`, `'bilateral'`,
  `'wiener'` o `'auto'`.
- **Devuelve:** array float32 en [0, 1], mismo shape.
- **Errores:** `ValueError` si el método no existe o la imagen no es 2D.

### `show_preprocessing_results(original, processed, filepath, output_dir=None, show=True) -> Path`

Figura de 4 paneles: original, procesada, histograma de la original
(azul) e histograma de la procesada (rojo).

- **output_dir**: carpeta de destino; por defecto `data/samples/` en
  la raíz del proyecto (se crea si no existe).
- **show**: si es `True`, abre la figura con `plt.show()`.
- **Devuelve:** ruta del PNG guardado (`<nombre>_preprocessing.png`).

### `pipeline_tem(filepath, method='auto', output_dir=None, show=True) -> dict`

Encadena las tres funciones anteriores.

- **Devuelve:** `{'original', 'processed', 'filepath', 'method_used'}`.
  En modo `auto`, `method_used` es el filtro que realmente se aplicó.
- **Errores:** los de `load_tem_image` y `ValueError` si el método no
  es válido (se valida antes de leer el archivo).

---

## Constantes ajustables

Al comienzo de `src/tem/loader.py`:

| Constante | Valor | Qué controla |
|---|---|---|
| `NORM_PERCENTILES` | (0.1, 99.9) | Percentiles que se llevan a 0 y 1 |
| `GAUSSIAN_SIGMA` | 1.0 | σ del filtro gaussiano |
| `MEDIAN_RADIUS` | 2 | Radio del disco de la mediana |
| `BILATERAL_SIGMA_COLOR` | 0.05 | Tolerancia de intensidad del bilateral |
| `BILATERAL_SIGMA_SPATIAL` | 2.0 | Alcance espacial del bilateral |
| `WIENER_WINDOW` | 5 | Lado de la ventana de Wiener |
| `AUTO_IMPULSE_FRACTION` | 0.002 | Fracción de píxeles impulsivos para elegir `median` |
| `AUTO_GRAD_LOW` | 0.04 | Límite ruido bajo / moderado |
| `AUTO_GRAD_HIGH` | 0.09 | Límite ruido moderado / alto |
| `DEFAULT_OUTPUT_DIR` | `data/samples/` | Carpeta de las figuras |

---

## Limitaciones y próximos pasos

- Los umbrales del modo `auto` se calibraron con campo claro HR. En 9
  imágenes de campo oscuro también eligió `gaussian`. Falta verificarlos
  con otras técnicas (difracción, baja magnificación).
- En una imagen HR con más ruido que `prueba6`, el modo `auto` elegiría
  `bilateral` o `wiener`, que pueden borrar las franjas de red.
- `bilateral` es lento en imágenes de 2048×2048 (~12 s contra ~3 s de
  los demás filtros).
- El tamaño de píxel que traen los `.dm3` todavía no se usa; servirá
  para la barra de escala y para medir partículas en las próximas
  etapas.
- La figura reduce la imagen a ~470 px de lado, así que el efecto del
  filtro no se aprecia; un recorte a tamaño real ayudaría.

---

# Etapa 2 — Segmentación de fases

Separa la imagen en **fases de intensidad** (zonas con niveles de gris
parecidos) y mide cada una. **No asigna elementos químicos**: solo
detecta y separa las fases.

## Uso rápido

```python
from tem import pipeline_tem, pipeline_segmentation

pre = pipeline_tem("imagen.dm3", show=False)        # Etapa 1
seg = pipeline_segmentation(pre)                     # Etapa 2

seg["n_phases"]          # número de fases
seg["threshold_method"]  # método aplicado: 'otsu', 'gmm', 'triangle'...
seg["stats"]["fase_0"]   # área, intensidad y regiones de la fase 0
seg["mask"]              # máscara de etiquetas (ver abajo)
```

La figura se guarda en `data/samples/<nombre>_segmentation.png`.
Para forzar un método: `pipeline_segmentation(pre, method="gmm")`.
Métodos válidos: `auto`, `otsu`, `gmm`, `triangle`, `adaptive`.

## La máscara de etiquetas

| Valor | Significado |
|---|---|
| `-1` | Interfaz entre fases (2 px de ancho) o línea de corte del watershed |
| `0` | Fase más oscura |
| `1`, `2`, … | Fases siguientes, de más oscura a más clara |

No hay una etiqueta de "fondo": el fondo es la fase que corresponda por
intensidad (la más clara en campo claro, la más oscura en campo oscuro).

## Cómo funciona, bloque por bloque

### Bloque 1 — Cuántas fases hay (`analyze_histogram`, `detect_n_phases`)

1. Histograma de 256 bins, suavizado (σ = 2 bins). Los bins 0 y 255 se
   reemplazan por sus vecinos: la normalización por percentiles acumula
   ahí miles de píxeles que generarían picos falsos.
2. Picos con prominencia mínima del 5 % de la altura máxima (se adapta
   a cada imagen) y al menos 8 bins de distancia.
3. **Fusión de picos**: se eliminan uno a uno los picos que no son una
   fase real, hasta que todos cumplan:
   - **masa**: su tramo del histograma tiene ≥ 5 % de los píxeles;
   - **valle**: el valle con cada vecino baja a ≤ 75 % del pico más bajo.
4. Con 0 o 1 pico se asume el mínimo de 2 fases.

La entropía de Shannon del histograma se imprime como dato, pero no
decide. Se probó confirmar el número de fases con el BIC de un GMM y
**no sirve**: con 200 000 píxeles elige siempre el máximo de
componentes.

**Por qué 75 % y no 70 %**: en `prueba13.dm3` (partícula oscura sobre
fondo claro) el valle entre ambas queda al 71 %; con 70 % la partícula
se fusionaba con el fondo.

### Bloque 2 — Dónde cortar (`select_threshold_method` y métodos)

| Método | Función | Cuándo lo elige `auto` |
|---|---|---|
| `triangle` | `threshold_triangle_method` | 2 fases y asimetría del histograma > 1.5 (campo oscuro) |
| `otsu` | `threshold_otsu_multi` | Hasta 3 fases (resto de los casos) |
| `gmm` | `threshold_gmm` | Más de 3 fases |
| `adaptive` | `threshold_adaptive_local` | Nunca: solo si se pide |

- **GMM** se entrena con 200 000 píxeles al azar (semilla fija: siempre
  da el mismo resultado). Cada umbral es el punto medio entre las
  medias de dos componentes vecinas.
- **Umbral local adaptativo**: bloque del 10 % del lado menor (205 px
  en 2048×2048). Siempre da 2 fases. **No se elige automáticamente**:
  la medida de iluminación de la especificación (desvío de medias en
  bloques de 64×64) supera 0.15 en 6 de 13 imágenes de campo claro por
  culpa de partículas grandes, no de la iluminación. Se probó también
  un plano ajustado sobre el fondo y no detectaba un gradiente
  agregado artificialmente. La medida se sigue calculando y, si
  supera 0.15, se imprime una advertencia.

#### Campo claro y campo oscuro

En **campo oscuro** casi toda la imagen es fondo oscuro y unas pocas
partículas brillan mucho. El histograma tiene un pico grande abajo y
una cola larga hacia arriba, y **Otsu falla**: una sola partícula
saturada se lleva el umbral y las demás quedan como fondo
(`Image_0039-tlt_0.92-rot_330.41.dm3`: 1 partícula de ~20). El
**método del triángulo** corta donde termina el pico del fondo.

La **asimetría** (skewness) del histograma distingue los dos casos:

| Tipo | Imágenes | Asimetría |
|---|---|---|
| Campo claro HR | 13 | −1.81 a +0.83 |
| Campo oscuro | 9 | +1.75 a +13.1 |

Por eso el límite `DARKFIELD_SKEW = 1.5`. Sale de solo 22 imágenes: con
otras muestras puede necesitar ajuste. El triángulo **no** sirve para
campo claro (en `prueba13.dm3` tomaría solo el 1.8 % del área).

### Bloque 3 — Regiones (`apply_segmentation`, `clean_mask`, `separate_touching_regions`, `detect_interfaces`)

- **`apply_segmentation`**: etiqueta cada píxel según el tramo de
  umbrales en que cae. Acepta también una máscara (umbral adaptativo).
- **`clean_mask`**: por fase, elimina regiones y huecos de menos de
  0.01 % de la imagen (419 px en 2048×2048) y suaviza bordes con una
  apertura de radio 1. Los píxeles que quedan sin fase pasan a la fase
  vecina. Una fase nunca desaparece. (El 0.1 % de la especificación,
  4194 px, borraba nanopartículas reales.)
- **`separate_touching_regions`**: separa con watershed las regiones de
  una fase que se tocan.
  - No procesa la fase mayoritaria (fondo o matriz).
  - **Detección adaptativa**: cada región se queda con su núcleo (los
    píxeles a más del 50 % de su distancia máxima al borde); si hay 2 o
    más núcleos, había un cuello. La erosión fija de radio 2 de la
    especificación solo detectaba cuellos de ~5 px o menos.
  - **No corta regiones de más del 5 % de la imagen**: son zonas (por
    ejemplo, espesores en `prueba7.dm3`), no partículas. Sin este
    límite, una zona de `prueba7` se partía en 31 pedazos.
  - Las líneas de corte se marcan con `-1`.
- **`detect_interfaces`**: un píxel es interfaz si algún vecino (de 8)
  es de otra fase. Marca un píxel a cada lado: 2 px de ancho.

### Bloque 4 — Estadísticas (`compute_phase_stats`)

Por fase: `area_px`, `area_pct` (sobre el área sin interfaces: las
fases suman 100 %), `intensidad_media`, `intensidad_std`,
`n_regiones`, `tamanio_medio_px`, `tamanio_min_px`, `tamanio_max_px` y
`n_fragmentos`. Además `interfaces` con `area_px` y `area_pct` (sobre
el área total).

**Fragmentos**: marcar las interfaces deja islotes de 1 a 418 px donde
una fase tenía puntas finas. No cuentan como regiones (se informan en
`n_fragmentos`), pero su área sí se suma. Sin esto, una fase de
`prueba7` contaba 106 regiones en lugar de 4.

Las regiones se cuentan con vecindad de 4 píxeles. Los tamaños están
**en píxeles**.

### Bloque 5 — Figura (`show_segmentation_results`)

Paneles: imagen original; máscara coloreada (interfaces en blanco) con
barra de colores y leyenda; histograma con cada fase en su color y los
umbrales; una imagen por fase sobre fondo negro. La grilla se adapta al
número de fases.

### Bloque 6 — Pipeline (`pipeline_segmentation`)

Encadena todo y devuelve `original`, `processed`, `mask`, `n_phases`,
`threshold_method`, `stats`, `filepath`, `thresholds` (umbrales
usados; `None` con `adaptive`) y `lineas_watershed` (máscara booleana
de las líneas de corte del watershed: en `mask` tienen `-1` igual que
las interfaces y no se pueden separar; la Etapa 4 las usa para medir
los contactos entre partículas). Valida el método y las claves del
diccionario de entrada antes de procesar.

## Resultados con las imágenes de prueba

**Campo claro HR** (13 imágenes, ~10 s cada una con ambas etapas):

| Imagen | Fases | Método | Área % por fase (regiones) |
|---|---|---|---|
| prueba1 (nanopartículas) | 2 | otsu | 12.0 (302) · 88.0 (8) |
| prueba7 (espesores) | 4 | gmm | 23.9 (1) · 35.8 (4) · 25.8 (1) · 14.4 (2) |
| prueba10 | 3 | otsu | 19.8 (9) · 30.8 (19) · 49.5 (13) |
| prueba13 (partícula grande) | 2 | otsu | 39.0 (7) · 61.0 (17) |
| Las otras 9 | 2 | otsu | — |

**Campo oscuro** (9 imágenes): todas eligen `triangle`.

| Imagen | Otsu (antes) | Triángulo (ahora) |
|---|---|---|
| Image_0039 rot_330 | 0.3 % · 1 partícula | 4.3 % · 39 regiones |
| Image_0039 rot_340 | 2.2 % · 21 | 5.2 % · 48 |
| img_0033 | 3.3 % · 34 | 6.5 % · 47 |
| S041C_0027/0029/0030 | Solo la parte brillante | Partícula completa |

## Scripts de prueba por bloque

En [`scripts/segmentacion/`](../scripts/segmentacion/). Se corren desde
la raíz del proyecto con una imagen (abre una figura) o una carpeta
(imprime una tabla):

```bash
uv run python scripts/segmentacion/bloque1_fases.py RUTA
```

| Script | Qué muestra |
|---|---|
| `bloque1_fases.py` | Histograma, picos que quedan y fusionados, vista previa de fases |
| `bloque2_umbrales.py` | Otsu, GMM, triángulo y umbral local lado a lado |
| `bloque3_regiones.py` | La máscara tras cada paso y un recorte a tamaño real |
| `bloque4_estadisticas.py` | Área por fase y distribución de tamaños de regiones |
| `bloque5_figura.py` | La figura final (con carpeta: genera todas sin abrir ventanas) |
| `bloque6_pipeline.py` | Etapas 1 y 2 completas; acepta un método: `RUTA gmm` |

## Constantes ajustables (`src/tem/segmentation.py`)

| Constante | Valor | Qué controla |
|---|---|---|
| `HIST_SMOOTH_SIGMA` | 2.0 | Suavizado del histograma (bins) |
| `PEAK_MIN_PROMINENCE` | 0.05 | Prominencia mínima de un pico (fracción del máximo) |
| `PEAK_MIN_DISTANCE` | 8 | Distancia mínima entre picos (bins) |
| `PEAK_MIN_MASS` | 0.05 | Masa mínima para que un pico sea fase |
| `VALLEY_MAX_RATIO` | 0.75 | Valle máximo para no fusionar dos picos |
| `MIN_PHASES` | 2 | Número mínimo de fases |
| `OTSU_MAX_PHASES` | 3 | Hasta cuántas fases se usa Otsu |
| `DARKFIELD_SKEW` | 1.5 | Asimetría a partir de la cual se usa el triángulo |
| `GMM_MAX_SAMPLES` | 200 000 | Píxeles para entrenar el GMM y medir la asimetría |
| `ADAPTIVE_BLOCK_FRACTION` | 0.10 | Bloque del umbral local (fracción del lado) |
| `ILLUMINATION_WARNING` | 0.15 | Medida de iluminación que dispara la advertencia |
| `CLEAN_MIN_FRACTION` | 0.0001 | Tamaño mínimo de región y de fragmento (fracción de la imagen) |
| `CLEAN_OPENING_RADIUS` | 1 | Radio de la apertura que suaviza bordes |
| `TOUCH_CORE_FRACTION` | 0.5 | Núcleo para detectar regiones que se tocan |
| `WATERSHED_MIN_DISTANCE` | 10 | Distancia mínima entre marcadores (px) |
| `WATERSHED_MAX_REGION` | 0.05 | Regiones más grandes no se cortan (fracción de la imagen) |

## Limitaciones conocidas

- **El número de fases sale solo del histograma**: no usa la
  información espacial. Una fase evidente en la imagen puede tener un
  valle poco marcado. Conviene revisarlo con `bloque1_fases.py`.
- **La iluminación despareja no se detecta.** En `prueba13.dm3` una
  esquina más oscura queda en la fase de la partícula.
- **Cortes del watershed en nanopartículas**: con la detección
  adaptativa, `prueba1` y `prueba2` pasaron de ~180 a ~300 regiones.
  Falta confirmar con las imágenes que no se corte de más (si corta de
  más, bajar `TOUCH_CORE_FRACTION`).
- **Campo oscuro**: las partículas muy tenues (casi al nivel del ruido)
  quedan afuera en parte, y dos partículas grandes que se tocan
  (`S041C_0025`) quedan unidas porque superan el 5 % del área.
- **Calibración**: todos los criterios se ajustaron con 22 imágenes
  (13 de campo claro HR y 9 de campo oscuro).
- **Unidades**: los tamaños están en píxeles; los `.dm3` traen el
  tamaño de píxel (por ejemplo 0.0024 µm en campo claro, 0.09 nm en
  campo oscuro) pero todavía no se usa.

---

# Etapa 3 — Análisis morfológico

Mide **tamaño, forma y orientación** de cada región que dejó la
segmentación. Todas las medidas están **en píxeles** y el módulo **no
asume nada sobre qué material es cada fase**.

## Uso rápido

```python
from tem import pipeline_tem, pipeline_segmentation, pipeline_morphology

pre = pipeline_tem("imagen.dm3", show=False)          # Etapa 1
seg = pipeline_segmentation(pre, show=False)           # Etapa 2
mor = pipeline_morphology(seg)                         # Etapa 3

mor["props"]["fase_0"][0]          # medidas de la primera región
mor["size_stats"]["fase_0"]        # estadísticas de tamaño
mor["shape_stats"]["fase_0"]["forma_predominante"]
mor["orientation_stats"]["fase_0"]["indice_orden_S"]
```

Genera dos figuras en `data/samples/`: `<nombre>_orientacion.png`
(rosa de orientaciones) y `<nombre>_morfologia.png`.

Para elegir las fases: `pipeline_morphology(seg, phases=[1])`.

## Qué regiones se analizan

- **Fases**: todas menos la **mayoritaria** (fondo o matriz). Funciona
  igual en campo claro (partículas oscuras) y en campo oscuro
  (partículas brillantes). Con `phases` se eligen a mano.
- **Regiones**: se separan con vecindad de 4 píxeles y se ignoran los
  fragmentos de menos de 419 px (0.01 % de la imagen), con el mismo
  criterio que las estadísticas de la Etapa 2.
- **Regiones que tocan el borde**: tienen área y forma truncadas. Se
  marcan con `toca_borde`, se dibujan (punteadas) pero **no entran**
  en las estadísticas de tamaño, forma y orientación.

## Bloque 1 — Propiedades por región (`extract_region_properties`)

Por región, con `skimage.measure.regionprops`: `area_px`,
`perimetro_px`, `eje_mayor_px`, `eje_menor_px`, `orientacion_deg`,
`centroide` (y, x), `bbox`, `solidez`, `excentricidad` y `toca_borde`.

**Convención de ángulos**: `orientacion_deg` va de 0° a 180°, con
**0° = horizontal** y 90° = vertical, medido contra reloj tal como se
ve la imagen. scikit-image usa otra convención (ángulo respecto del eje
vertical, entre −90° y 90°): se convierte con
`(grados + 90) mod 180`.

## Bloque 2 — Tamaño y forma

### `analyze_size_distribution`

Media, mediana, desvío, mínimo, máximo, IQR y percentiles 10/25/75/90
del área, y un **ajuste log-normal** (`scipy.stats.lognorm` con `loc`
fijo en 0) evaluado con el test de Kolmogorov-Smirnov
(`buen_ajuste` si p > 0.05). Con menos de 5 regiones no se ajusta.

**Cómo leer `buen_ajuste`** (es orientativo):
- El p-valor es optimista: los parámetros se ajustan con los mismos
  datos que se testean.
- Con cientos de regiones el test detecta cualquier diferencia chica.
- **La distribución está cortada en 419 px**: la limpieza de la Etapa
  2 elimina las regiones más chicas. Por eso falla el ajuste en
  `prueba1.dm3` (293 partículas, p = 0.007): faltan las más chicas.

### `analyze_shape`

Circularidad (4π·área/perímetro²), excentricidad (media, percentiles y
conteo en 5 rangos), solidez y relación de aspecto (eje mayor / eje
menor). La forma predominante se decide con los valores medios y estas
reglas, en orden:

| Condición | Forma |
|---|---|
| circularidad > 0.85 | `circular` |
| relación de aspecto > 2.0 | `elongada` |
| solidez < 0.80 | `irregular` |
| si no | `compacta` |

`conteo_formas` aplica las mismas reglas a **cada región**: un
promedio puede esconder una mezcla (en `prueba1`: 192 compactas, 46
elongadas, 32 circulares y 23 irregulares).

**La circularidad se recorta a 1**: el perímetro digital subestima el
real y en regiones chicas la fórmula da más de 1. Un círculo perfecto
da 1.16 con r = 5 px, 0.98 con r = 10 y 0.93 con r = 40.

## Bloque 3 — Orientación

### `analyze_orientation`

- Se usan solo las regiones con **excentricidad ≥ 0.5**: las casi
  circulares no tienen eje mayor definido.
- Ángulo medio y desvío con **estadística circular de período 180°**
  (`scipy.stats.circmean`/`circstd` entre 0 y π): 170° y 10° promedian
  0°, no 90°.
- **Índice de orden** S = √(⟨cos 2θ⟩² + ⟨sin 2θ⟩²): 0 al azar, 1 todas
  alineadas. `orientado` si S > 0.5, si no `aleatorio`.
- **Con menos de 10 regiones** la clasificación es `insuficiente`: al
  azar, S vale en promedio ≈ 0.89/√n, así que con pocas regiones sale
  alto por casualidad. Se informa como `S_esperado_aleatorio`:

| Regiones | S esperado al azar |
|---|---|
| 3 | 0.51 |
| 5 | 0.40 |
| 10 | 0.28 |
| 50 | 0.13 |

### `plot_orientation_rose`

Una rosa polar por fase con 18 sectores de 10°, duplicada en
180°–360° por simetría, con el color de la fase y una línea punteada en
el ángulo medio.

## Bloque 4 — Figura (`show_morphology_results`)

Paneles: elipses ajustadas sobre la imagen (color por fase, punteadas
si tocan el borde); mapa de calor del área de cada región (escala
logarítmica: las áreas van de 419 px a más de un millón); mapa de
calor de la excentricidad; y por fase, el histograma de áreas con la
log-normal (**roja** si el ajuste es bueno, **gris punteada** si no) y
líneas de media (rojo) y mediana (verde).

## Bloque 5 — Pipeline (`pipeline_morphology`)

Encadena todo, imprime un resumen por fase y devuelve `original`,
`processed`, `mask`, `props`, `size_stats`, `shape_stats`,
`orientation_stats`, `filepath` y `n_phases`.

## Resultados con las imágenes de prueba

Las tres etapas tardan entre 12 y 22 s por imagen.

| Imagen | Regiones | Área media (px) | Forma | S | Orientación |
|---|---|---|---|---|---|
| prueba1 | 293 | 1 496 | compacta | 0.10 | aleatorio |
| prueba2 | 292 | 3 230 | compacta | 0.17 | aleatorio |
| prueba9 | 205 | 2 591 | compacta | 0.17 | aleatorio |
| prueba11 | 198 | 5 449 | compacta | 0.05 | aleatorio |
| prueba14 | 15 | 1 307 | irregular | 0.71 | orientado* |
| Image_0039 rot_330 (oscuro) | 35 | 4 250 | compacta | 0.16 | aleatorio |
| Image_0039 rot_340 (oscuro) | 45 | 3 435 | irregular | 0.06 | aleatorio |
| img_0032 (oscuro) | 23 | 7 716 | compacta | 0.66 | orientado* |
| img_0033 (oscuro) | 45 | 5 735 | compacta | 0.21 | aleatorio |

\* Ver limitaciones: la orientación detectada viene de la textura de la
imagen, no de las partículas.

Con pocas regiones (`prueba13`, `prueba7`, las `S041C`) no hay
estadística útil: la orientación queda `insuficiente` o la fase sin
regiones.

## Scripts de prueba por bloque

En [`scripts/morfologia/`](../scripts/morfologia/), con una imagen
(abre figuras) o una carpeta (imprime una tabla):

| Script | Qué muestra |
|---|---|
| `bloque1_propiedades.py` | Tabla de regiones; centroides y bounding boxes sobre la imagen |
| `bloque2_tamano_forma.py` | Áreas con la log-normal, circularidad vs. aspecto, solidez, excentricidad |
| `bloque3_orientacion.py` | Rosa de orientaciones; `--sintetico` compara elipses alineadas y al azar |
| `bloque4_figura.py` | La figura morfológica completa |
| `bloque5_pipeline.py` | Las tres etapas desde el archivo; acepta fases: `RUTA 1` o `RUTA 0,2` |

## Constantes ajustables (`src/tem/morphology.py`)

| Constante | Valor | Qué controla |
|---|---|---|
| `LOGNORM_MIN_REGIONS` | 5 | Mínimo de regiones para ajustar la log-normal |
| `KS_PVALUE_OK` | 0.05 | p-valor para `buen_ajuste` |
| `CIRCULAR_MIN_CIRCULARITY` | 0.85 | Circularidad para `circular` |
| `ELONGATED_MIN_ASPECT` | 2.0 | Relación de aspecto para `elongada` |
| `IRREGULAR_MAX_SOLIDITY` | 0.80 | Solidez para `irregular` |
| `ORIENTATION_MIN_ECCENTRICITY` | 0.5 | Excentricidad mínima para la orientación |
| `ORIENTED_MIN_S` | 0.5 | S para `orientado` |
| `ORIENTATION_MIN_REGIONS` | 10 | Mínimo de regiones para clasificar la orientación |
| `ROSE_BINS` | 18 | Sectores de la rosa (de 10°) |

El tamaño mínimo de región es `CLEAN_MIN_FRACTION` de
`segmentation.py`, compartido con la Etapa 2.

## Limitaciones conocidas

- **Fase mayoritaria**: si las partículas ocupan más de la mitad de la
  imagen, la fase "excluida" son ellas. En `S041C_0025.dm3` se
  analiza el vacío; hay que pasar `phases=[1]`.
- **La orientación mide formas, no materiales**: en `prueba14.dm3` las
  regiones chicas siguen las franjas de la red cristalina y en
  `img_0032.dm3` un rayado diagonal del fondo; ambas dan `orientado`
  aunque las partículas no estén alineadas.
- **Partículas en el borde**: quedan fuera de las estadísticas. En
  imágenes con pocas partículas grandes (`S041C`) casi no queda nada
  para medir.
- **Distribución truncada** en 419 px por la limpieza de la Etapa 2
  (afecta la media, la mediana y el ajuste log-normal).
- **Unidades**: todo en píxeles.

---

# Etapa 4 — Análisis de fases

Tres módulos que se encadenan: **fracción** (cuánto hay de cada fase),
**composicional relativo** (cómo es su intensidad) y **espacial** (cómo
se distribuyen sus partículas). Todo en píxeles.

## Uso rápido

```python
from tem import (pipeline_tem, pipeline_segmentation,
                 pipeline_morphology, pipeline_phase_analysis)

pre = pipeline_tem("imagen.dm3", show=False)          # Etapa 1
seg = pipeline_segmentation(pre, show=False)           # Etapa 2
mor = pipeline_morphology(seg, show=False)             # Etapa 3
res = pipeline_phase_analysis(mor)                     # Etapa 4

res["area_fractions"]["fase_1"]      # fracciones de la Fase 1
res["clustering"]["fase_1"]          # Hopkins y DBSCAN
res["inhomogeneities"]["fondo"]      # zonas anómalas del fondo
```

Imprime un reporte final (fracciones, distribución espacial,
inhomogeneidades y gradientes) y guarda tres figuras en
`data/samples/`: `<nombre>_fracciones.png`, `_composicional.png` y
`_espacial.png`. Cada módulo también se puede correr solo:
`pipeline_fraction` → `pipeline_compositional` → `pipeline_spatial`;
cada uno devuelve lo que recibe más sus resultados.

Si las partículas ocupan más de la mitad de la imagen, hay que indicar
el fondo: `pipeline_phase_analysis(mor, background=0)` (y en la Etapa 3,
las fases que no son fondo). En `analizar_tem.py`: `FONDO = 0`.

## Los nombres de las fases

- **Fondo**: la fase mayoritaria (o la indicada con `background`).
- **Fase 1, Fase 2, …**: las demás, de la más oscura a la más clara.
  Cada una guarda su `etiqueta_mascara` para cruzarla con las Etapas 2
  y 3 (que nombran las fases por etiqueta: en `prueba1.dm3` la
  `fase_0` de la Etapa 3 es la `fase_1` de la Etapa 4).
- **Nunca se asignan elementos químicos.**

## Módulo A — Fracción de fases (`fraction.py`)

### A1 — Fracciones de área y volumen

- `area_pct`: sobre el área sin fondo ni interfaces (las fases suman
  100 %). `area_pct_con_fondo`: sobre toda la imagen. Si las sumas no
  dan 100 % (tolerancia 0.01 %) se lanza `ValueError`.
- **Volumen (principio de Delesse)**: fracción de volumen ≈ fracción
  de área. Hay dos lecturas: `vol_frac_estimada` (dentro del material,
  como pide la especificación) y `vol_frac_estimada_con_fondo` (sobre
  toda la imagen). **Con una sola fase fuera del fondo** —20 de las 22
  imágenes de prueba— la primera vale siempre 100 % y la útil es la
  segunda (en `prueba1`, las partículas cubren el 11.6 %).
- **Advertencia**: Delesse supone un **corte** plano de una
  microestructura isótropa. Una imagen TEM es una **proyección** de todo
  el espesor: las partículas se superponen y la fracción de área tiende
  a sobreestimar la de volumen.

### A2 — Densidad numérica

Cantidad de partículas por área (también cada 1000 px²) y
**separación al vecino más cercano** (no el promedio de todas las
distancias entre pares, que depende sobre todo del tamaño de la imagen).
El **índice de Clark-Evans** R compara esa separación con la esperada
al azar (0.5/√densidad): ≈ 1 al azar, < 1 agrupadas, > 1 espaciadas.
Una grilla da R = 2.0; puntos al azar dan ~1.1 porque no se corrige el
efecto del borde.

### A3 — Interfaces

Los pares de fases adyacentes se detectan dilatando cada fase 2 px para
cruzar la banda de interfaz. La longitud de un par es su cantidad de
píxeles / 2 (la banda tiene 2 px); contar píxeles sobreestima los
bordes diagonales hasta ~40 %. Los **contactos entre partículas** de una
misma fase son las líneas del watershed: se miden con
`lineas_watershed` de la Etapa 2 (sin ellas, quedan en `None`; antes de
guardarlas aparte, deducirlas de la máscara daba 10 733 px en
`prueba6` cuando eran 133).

## Módulo B — Composicional relativo (`compositional.py`)

> **La intensidad TEM no es composición.** Depende del espesor, de la
> difracción y del enfoque. Este módulo compara intensidades relativas
> entre fases **dentro de una misma imagen** (normalizada en la Etapa
> 1): no sirve para comparar imágenes ni para identificar materiales.

### B1 — Perfiles de intensidad

- Por fase (incluido el fondo, así siempre hay al menos un par):
  histograma de 100 bins, media, mediana, moda, desvío, asimetría,
  curtosis (en exceso) e IQR.
- **Contraste** |mA − mB| / (mA + mB): alto (> 0.3), medio, bajo
  (< 0.1).
- **Perfil de interfaz**: la intensidad promediada según la distancia
  con signo al borde entre dos fases (−20 a +20 px), es decir, el perfil
  perpendicular promediado a lo largo de toda la interfaz. Se corrige el
  corrimiento de 1.5 px que introduce la banda de interfaz. Se miden el
  **ancho de transición** (10 %–90 % del cambio; coincide con 2.56·σ
  dentro de ~1 px en bordes sintéticos), la pendiente máxima del perfil
  y el percentil 95 del gradiente de Sobel.
- **`meseta_alcanzada`**: si el perfil sigue cambiando en algún extremo,
  el ancho está limitado por la ventana y se informa como
  **"≥ X px (sin meseta)"**. En las imágenes reales pasa casi siempre:
  los bordes de las partículas son graduales (se afinan hacia el borde)
  y solo 4 pares de campo claro dieron un ancho confiable (9–17 px).

### B2 — Mapas e inhomogeneidades

- Mapas de intensidad y de **z-score** por fase (respecto de su propia
  media y desvío) y un mapa global.
- **Inhomogeneidades**: con millones de píxeles, el ruido solo ya
  supera |z| > 2.5 en ~1 % de ellos. Por eso se suaviza dentro de cada
  fase (σ = 2 px) y solo cuentan las zonas conectadas de ≥ 419 px. Con
  ruido puro da 0 zonas.
- En la práctica encuentra zonas que se apartan de su fase, no
  necesariamente defectos: el **halo claro de Fresnel** alrededor de las
  partículas (`prueba13`), **huecos de vacío** dentro de un fondo que
  mezcla vacío y película de soporte (`prueba1`), núcleos más gruesos.

## Módulo C — Espacial (`spatial.py`)

Cuatro medidas complementarias. Con menos de 10 partículas, todas dan
**'insuficiente'**.

| Medida | Qué responde | Referencia al azar |
|---|---|---|
| **RDF g(r)** | ¿Hay más vecinas a distancia r que al azar? | 40 repeticiones al azar en la misma ventana (corrige el borde) |
| **Mapa de densidad** | ¿Hay zonas con más partículas? | CV de 200 distribuciones al azar |
| **Hopkins + DBSCAN** | ¿Están agrupadas? ¿Dónde están los grupos? | Hopkins: H ≈ 0.5 al azar |
| **Gradientes** | ¿Cambia algo de un lado al otro de la imagen? | Regresión con corrección de Bonferroni |

### C1 — RDF

g(r) hasta 1/4 del lado de la imagen. Como las partículas tienen tamaño,
g ≈ 0 por debajo de su diámetro aunque estén agrupadas: se clasifica con
el promedio de g **entre 1 y 3 diámetros típicos**, comparado con la
franja (percentiles 2.5–97.5) que dan las repeticiones al azar. Una
grilla de paso a da picos en a, a·√2, 2a…

### C2 — Mapa de densidad

Grilla automática (~5 partículas por celda; con 32×32 fijo y 300
partículas, cada celda tendría 0.3 y el CV saldría alto aunque estén al
azar). El CV se clasifica con los límites de la especificación (0.3 y
0.7) solo si supera el percentil 97.5 del CV al azar; si no, la
clasificación es **'uniforme (compatible con azar)'**.

### C3 — Hopkins y DBSCAN

Dos cambios respecto de la especificación, comprobados con patrones
sintéticos:

- **Hopkins con distancias al cuadrado** (H = Σu² / (Σu² + Σw²),
  variante de Lawson y Jurs para 2D). Con distancias simples una grilla
  daba H = 0.28 ('aleatorio'); ahora 0.14 ('regular'). Se promedian 20
  muestras.
- **min_samples de DBSCAN según el azar**: percentil 99 de las vecinas
  esperadas al azar dentro de eps (Poisson), más 1. Con n // 20 no se
  encontraban grupos en `prueba1` (300 partículas → min 15) y sí en
  puntos al azar. eps = 2 × distancia media al vecino más cercano.

### C4 — Gradientes espaciales

10 franjas en cada dirección; en cada una, partículas, área media (sin
las del borde) e intensidad media de la fase. Recta con
`scipy.stats.linregress`. Son 6 tests por fase: un gradiente es
**significativo solo si p < 0.05/6** (Bonferroni); sin la corrección,
~1 de cada 4 fases mostraría uno por casualidad (en `prueba1`, la
densidad horizontal da p = 0.029 y no pasa). **Un gradiente de
intensidad puede ser de espesor o de iluminación**, no de la fase
(`prueba7`).

## Resultados con las imágenes de prueba

| Imagen | Cobertura | Clark-Evans R | RDF | Densidad | Hopkins H | DBSCAN |
|---|---|---|---|---|---|---|
| prueba1 | 11.6 % | 0.79 | clustering | mod. heterogénea | 0.89 clustering | 10 grupos |
| prueba2 | 24.0 % | 1.02 | clustering | mod. heterogénea | 0.78 clustering | 6 grupos |
| prueba4 | 15.7 % | 0.96 | clustering | mod. heterogénea | 0.74 aleatorio | 3 grupos |
| img_0033 (oscuro) | 6.4 % | 0.53 | clustering | mod. heterogénea | 0.95 clustering | 3 grupos |
| prueba7 (3 fases) | 23.6 / 25.5 / 14.2 % | — | insuficiente | insuficiente | insuficiente | — |

En `prueba1` coinciden las cuatro medidas. `prueba4` es un caso de
frontera: la RDF lo ve agrupado y Hopkins (0.74) queda justo debajo de 0.75.

## Scripts de prueba por bloque

En [`scripts/fases/`](../scripts/fases/), con una imagen (abre figuras),
una carpeta (imprime una tabla) y, en los del Módulo C,
`--sintetico` (patrones conocidos: grilla, al azar, grupos):

| Script | Qué muestra |
|---|---|
| `a1_fracciones.py` | Fracciones de área y volumen (acepta `FONDO`) |
| `a2_densidad.py` | Centroides unidos a su vecino más cercano, Clark-Evans |
| `a3_interfaces.py` | Interfaces coloreadas por par y contactos entre partículas |
| `a4_pipeline.py` | Figura de fracciones (Módulo A completo) |
| `b1_perfiles.py` | Histogramas por fase y perfiles de interfaz |
| `b2_inhomogeneidades.py` | Mapa de z-score y zonas anómalas |
| `b3_pipeline.py` | Figura composicional (Módulo B completo) |
| `c1_rdf.py` | g(r) con la franja del azar |
| `c2_densidad.py` | Mapa de densidad sobre la imagen |
| `c3_clustering.py` | Grupos de DBSCAN y Hopkins |
| `c4_gradientes.py` | Valores por franja y rectas |
| `c5_pipeline.py` | Figura espacial (Módulo C completo) |

## Constantes ajustables

| Constante | Valor | Qué controla |
|---|---|---|
| `fraction.SUM_TOLERANCE_PCT` | 0.01 | Tolerancia de la suma de fracciones (%) |
| `fraction.INTERFACE_REACH` | 2 | Dilatación para cruzar la banda de interfaz (px) |
| `compositional.CONTRAST_HIGH` / `_LOW` | 0.3 / 0.1 | Límites del contraste |
| `compositional.INTERFACE_HALF_WIDTH` | 20 | Ventana del perfil de interfaz (px a cada lado) |
| `compositional.PLATEAU_MAX_SLOPE_RATIO` | 0.1 | Pendiente máxima en los extremos para "meseta" |
| `compositional.INHOMOGENEITY_SIGMA` | 2.0 | Suavizado antes de buscar inhomogeneidades (px) |
| `compositional.INHOMOGENEITY_Z` | 2.5 | Umbral de z-score |
| `spatial.MIN_PARTICLES` | 10 | Mínimo de partículas para clasificar |
| `spatial.RDF_SHORT_RANGE` | (1, 3) | Rango de la RDF para clasificar (diámetros) |
| `spatial.DENSITY_PER_CELL` | 5 | Partículas por celda con `grid_size='auto'` |
| `spatial.CV_UNIFORM` / `CV_HETEROGENEOUS` | 0.3 / 0.7 | Límites del CV |
| `spatial.HOPKINS_POWER` | 2 | Potencia de las distancias en H |
| `spatial.HOPKINS_CLUSTERED` / `_REGULAR` | 0.75 / 0.25 | Límites de H |
| `spatial.DBSCAN_EPS_FACTOR` | 2.0 | eps = factor × vecino más cercano medio |
| `spatial.DBSCAN_RANDOM_PERCENTILE` | 0.99 | Percentil de Poisson para min_samples |
| `spatial.GRADIENT_STRIPS` | 10 | Franjas por dirección |
| `spatial.GRADIENT_ALPHA` / `_N_TESTS` | 0.05 / 6 | Bonferroni: p < 0.05/6 |

## Limitaciones conocidas

- **Partículas partidas** (anillos en campo oscuro, cortes del
  watershed): varias regiones muy juntas se ven como un grupo e inflan
  la RDF y Hopkins. En campo oscuro conviene mirar el pico de g(r) a
  distancias menores que el tamaño de partícula.
- **Fase mayoritaria = partículas**: si cubren más de la mitad de la
  imagen hay que indicar el fondo (`background` / `FONDO`).
- **Intensidad ≠ composición**: los gradientes y las inhomogeneidades de
  intensidad pueden ser de espesor, iluminación u ópticos (Fresnel).
- **Casos de frontera**: métodos distintos pueden no coincidir cerca
  de sus límites (`prueba4`).
- **Bordes de la imagen**: Clark-Evans y Hopkins no corrigen el borde
  (lo sobreestiman un poco); la RDF y el mapa de densidad sí.
- **Pocas partículas**: con menos de 10, la distribución espacial no se
  clasifica.
- **Unidades**: todo en píxeles.
