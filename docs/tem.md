# Módulo TEM

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Daniel Romero (daniel.romero@ib.edu.ar).

El módulo `tem` trabaja con imágenes de microscopía electrónica de
transmisión (TEM) en dos etapas:

1. **[Etapa 1 — Carga y preprocesamiento](#etapa-1--carga-y-preprocesamiento)**:
   lee la imagen, la normaliza a [0, 1] y reduce el ruido.
   Código: [`src/tem/loader.py`](../src/tem/loader.py).
2. **[Etapa 2 — Segmentación de fases](#etapa-2--segmentación-de-fases)**:
   detecta cuántas fases de intensidad hay, las separa, limpia las
   regiones y calcula estadísticas por fase.
   Código: [`src/tem/segmentation.py`](../src/tem/segmentation.py).

Tests: [`tests/tem/`](../tests/tem/) (71 tests, `uv run pytest`).

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
`threshold_method`, `stats`, `filepath` y `thresholds` (umbrales
usados; `None` con `adaptive`). Valida el método y las claves del
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
