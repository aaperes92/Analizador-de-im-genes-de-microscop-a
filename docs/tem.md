# Módulo TEM: carga y preprocesamiento (Etapa 1)

Parte del proyecto **Analizador de Imágenes de Microscopía**.
Autor del módulo: Daniel Romero (daniel.romero@ib.edu.ar).

El módulo `tem` lee imágenes de microscopía electrónica de transmisión
(TEM), las normaliza a un rango común y reduce su ruido con un filtro
elegido a mano o de forma automática. Es la base para las etapas
siguientes (segmentación y medición).

Código: [`src/tem/loader.py`](../src/tem/loader.py).
Tests: [`tests/tem/test_loader.py`](../tests/tem/test_loader.py).

---

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

- Los umbrales del modo `auto` se calibraron con un solo tipo de imagen
  (campo claro HR). Conviene verificarlos con otras técnicas (campo
  oscuro, difracción, baja magnificación).
- En una imagen HR con más ruido que `prueba6`, el modo `auto` elegiría
  `bilateral` o `wiener`, que pueden borrar las franjas de red.
- `bilateral` es lento en imágenes de 2048×2048 (~12 s contra ~3 s de
  los demás filtros).
- El tamaño de píxel que traen los `.dm3` todavía no se usa; servirá
  para la barra de escala y para medir partículas en las próximas
  etapas.
- La figura reduce la imagen a ~470 px de lado, así que el efecto del
  filtro no se aprecia; un recorte a tamaño real ayudaría.
