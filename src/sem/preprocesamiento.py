"""Preprocesamiento de imágenes SEM: mejora de la ROI antes de segmentarla.

Pipeline: suavizado -> corrección de sombras -> ecualización (CLAHE) -> normalización tanh.
Cada paso es una función pura: recibe un array y retorna uno nuevo, sin modificar la entrada.

Parte del módulo SEM del proyecto "Analizador de Imágenes de Microscopía".

Autor: Mauricio Romano (mauricio.romano@ib.edu.ar)
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.widgets import Slider

from .carga import _BACKENDS_SIN_VENTANA


def suavizar(imagen: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica un suavizado gaussiano a la imagen para reducir ruido de alta frecuencia.

    Reemplaza al `cv2.blur(roi, (3, 3))` del notebook anterior. Un filtro de caja de lado k
    equivale aproximadamente a una gaussiana de sigma = k / sqrt(12) (k=3 -> sigma ~ 0.87).

    Parameters
    ----------
    imagen : np.ndarray
        Array 2D float32 normalizado (0-1).
    sigma : float
        Desviación estándar del kernel gaussiano en píxeles.
        Valores típicos: 0.5 (suave) a 3.0 (agresivo). 0 devuelve una copia sin cambios.
        Default: 1.0.

    Returns
    -------
    np.ndarray
        Imagen suavizada, nueva, con el mismo shape y dtype que la entrada.

    Raises
    ------
    ValueError
        Si `sigma` es negativo.
    """
    if sigma < 0:
        raise ValueError(f"sigma debe ser >= 0, se recibió {sigma}")
    if sigma == 0:
        return imagen.copy()
    suavizada = ski.filters.gaussian(imagen, sigma=sigma, preserve_range=True)
    return suavizada.astype(imagen.dtype, copy=False)


_METODOS_SOMBRAS = ("blur_resta", "white_tophat")


def corregir_sombras(
    imagen: np.ndarray, metodo: str = "blur_resta", fraccion: float = 0.03, radio: int = 50
) -> np.ndarray:
    """
    Corrige la iluminación no uniforme (sombras locales) de la imagen.

    Ofrece dos métodos:
    - 'blur_resta': técnica del equipo. Estima el fondo con un blur gaussiano ancho
      y lo resta a la imagen. Rápido y efectivo para gradientes suaves de iluminación.
      Usar cuando la iluminación varía de forma gradual a lo largo de la imagen.
    - 'white_tophat': morfología matemática. Resta a la imagen su apertura con un disco,
      que actúa como fondo. Más robusto ante variaciones abruptas.
      Usar cuando el fondo tiene cambios bruscos y las estructuras de interés son
      más brillantes que su entorno.

    Parameters
    ----------
    imagen : np.ndarray
        Array 2D float32 normalizado (0-1).
    metodo : str
        'blur_resta' o 'white_tophat'. Default: 'blur_resta'.
    fraccion : float
        Solo para 'blur_resta'. Tamaño del fondo como fracción del lado medio de la imagen
        (sqrt(alto*ancho)). Se usa sigma_fondo = fraccion * sqrt(alto*ancho) / sqrt(12), que
        equivale al filtro de caja del notebook anterior. Default: 0.03.
    radio : int
        Solo para 'white_tophat'. Radio del disco en píxeles. Debe ser MAYOR que la estructura
        más grande que se quiere conservar: lo que sea más chico que el disco se conserva y
        lo más grande se considera fondo y se elimina. Default: 50.
        El disco se descompone en cruces (`decomposition='crosses'`), una aproximación muy
        cercana al disco exacto pero ~35 veces más rápida (0.5 s vs 17 s con radio 50 en
        una ROI de ~850x850 px).

    Returns
    -------
    np.ndarray
        Imagen nueva float32 con la iluminación corregida, mismo shape.
        'blur_resta' da valores centrados en 0 (pueden ser negativos; es lo esperado).
        'white_tophat' da valores >= 0.

    Raises
    ------
    ValueError
        Si `metodo` no es válido, si `fraccion` no está en (0, 1] o si `radio` < 1.
    """
    if metodo not in _METODOS_SOMBRAS:
        raise ValueError(f"metodo debe ser uno de {_METODOS_SOMBRAS}, se recibió {metodo!r}")

    if metodo == "blur_resta":
        if not 0 < fraccion <= 1:
            raise ValueError(f"fraccion debe estar en (0, 1], se recibió {fraccion}")
        sigma_fondo = fraccion * np.sqrt(imagen.size) / np.sqrt(12)
        fondo = ski.filters.gaussian(imagen, sigma=sigma_fondo, preserve_range=True)
        corregida = imagen - fondo
    else:
        if radio < 1:
            raise ValueError(f"radio debe ser >= 1, se recibió {radio}")
        disco = ski.morphology.disk(int(radio), decomposition="crosses")
        corregida = ski.morphology.white_tophat(imagen, disco)

    return corregida.astype(np.float32, copy=False)


_METODOS_ECUALIZACION = ("clahe", "global", "rescale")


def ecualizar(imagen: np.ndarray, metodo: str = "clahe", clip_limit: float = 0.03) -> np.ndarray:
    """
    Ecualiza el contraste de la imagen para mejorar la visibilidad de estructuras.

    Acepta entradas con cualquier rango, incluidos valores negativos como la salida de
    corregir_sombras('blur_resta'): antes de ecualizar se reescala min-max a [0, 1].
    La escala absoluta se pierde; solo importa el orden relativo de intensidades.

    Parameters
    ----------
    imagen : np.ndarray
        Array 2D float32 (típicamente la salida de corregir_sombras).
    metodo : str
        - 'clahe': ecualización adaptativa con límite de recorte (recomendado para SEM)
        - 'global': ecualización global del histograma
        - 'rescale': reescalado de intensidad al rango completo [0, 1]
        Default: 'clahe'.
    clip_limit : float
        Solo para 'clahe'. Límite de recorte del histograma local, en (0, 1].
        ~0.01: poco realce. ~0.1: realce agresivo. Default: 0.03.

    Returns
    -------
    np.ndarray
        Imagen nueva float32 en el rango [0, 1].

    Raises
    ------
    ValueError
        Si `metodo` no es válido o `clip_limit` no está en (0, 1].
    """
    if metodo not in _METODOS_ECUALIZACION:
        raise ValueError(
            f"metodo debe ser uno de {_METODOS_ECUALIZACION}, se recibió {metodo!r}"
        )
    if metodo == "clahe" and not 0 < clip_limit <= 1:
        raise ValueError(f"clip_limit debe estar en (0, 1], se recibió {clip_limit}")

    # Imagen uniforme: no hay contraste que ecualizar.
    if np.ptp(imagen) == 0:
        return np.zeros_like(imagen, dtype=np.float32)

    reescalada = ski.exposure.rescale_intensity(
        imagen.astype(np.float32), in_range="image", out_range=(0.0, 1.0)
    )
    if metodo == "clahe":
        resultado = ski.exposure.equalize_adapthist(reescalada, clip_limit=clip_limit)
    elif metodo == "global":
        resultado = ski.exposure.equalize_hist(reescalada)
    else:
        resultado = reescalada
    return resultado.astype(np.float32, copy=False)


def normalizar_tanh(imagen: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Aplica normalización no lineal mediante tangente hiperbólica para realzar contraste medio.

    La imagen se centra en su mediana y se escala por su máxima desviación, de modo que
    queda en [-1, 1] antes de aplicar tanh. Así la tanh comprime simétricamente los
    extremos (muy oscuros y muy brillantes) y expande los tonos cercanos a la mediana.

        salida = tanh(factor * (x - mediana) / (max|x - mediana| + 1e-8))

    Parameters
    ----------
    imagen : np.ndarray
        Array 2D float32, en cualquier rango (por ejemplo la salida de ecualizar, en [0, 1],
        o de corregir_sombras, con negativos).
    factor : float
        Ganancia aplicada antes de la tanh. Debe ser > 0.
        factor < 1: casi lineal (resultado parecido a la entrada).
        factor > 1: satura los extremos hacia ±1 y aumenta el contraste en los tonos medios;
        valores altos se acercan a una imagen binaria.
        Default: 1.0 (los extremos llegan a ±tanh(1) ≈ ±0.76).

    Returns
    -------
    np.ndarray
        Imagen nueva float32 en el rango (-1, 1), centrada en 0 (la mediana queda en 0).
        IMPORTANTE para la binarización: el rango NO es [0, 1]. El signo indica si un píxel
        está por encima o por debajo de la mediana.

    Raises
    ------
    ValueError
        Si `factor` <= 0.
    """
    if factor <= 0:
        raise ValueError(f"factor debe ser > 0, se recibió {factor}")
    centrada = imagen.astype(np.float32) - np.median(imagen)
    escala = np.max(np.abs(centrada)) + 1e-8
    return np.tanh(factor * centrada / escala).astype(np.float32, copy=False)


def explorar_preprocesamiento(roi: np.ndarray, metodo_sombras: str = "blur_resta") -> dict:
    """
    Ventana interactiva para explorar y ajustar los parámetros de preprocesamiento.

    Muestra el pipeline en tiempo real al mover los sliders:
    fila 1: ROI original | sin sombras | post-CLAHE
    fila 2: post-tanh | histogramas de cada paso (superpuestos, normalizados a [0, 1]) | sliders

    Sliders:
    - Sigma blur: suavizado gaussiano (0.5 a 5.0, inicial 1.0)
    - Fracción sombras (0.01 a 0.10, inicial 0.03) si metodo_sombras='blur_resta',
      o Radio tophat (5 a 150 px, inicial 50) si metodo_sombras='white_tophat'
    - CLAHE clip (0.001 a 0.03, inicial 0.01)
    - Factor tanh (0.5 a 3.0, inicial 1.0)

    La función bloquea hasta que se cierra la ventana (o se presiona Enter). Al cerrar,
    imprime los parámetros en consola con el formato para copiar al pipeline.

    Parameters
    ----------
    roi : np.ndarray
        ROI 2D float32 normalizada (0-1).
    metodo_sombras : str
        'blur_resta' o 'white_tophat': define qué controla el segundo slider.

    Returns
    -------
    dict
        Parámetros finales: {'sigma', 'metodo_sombras', 'fraccion' o 'radio',
        'clip_limit', 'factor_tanh'}, para usarlos directamente desde el script.

    Raises
    ------
    ValueError
        Si `metodo_sombras` no es válido.
    RuntimeError
        Si matplotlib usa un backend sin ventanas.
    """
    if metodo_sombras not in _METODOS_SOMBRAS:
        raise ValueError(
            f"metodo_sombras debe ser uno de {_METODOS_SOMBRAS}, se recibió {metodo_sombras!r}"
        )
    if plt.get_backend().lower() in _BACKENDS_SIN_VENTANA:
        raise RuntimeError(
            f"El backend de matplotlib '{plt.get_backend()}' no abre ventanas; "
            "usá uno interactivo (por ejemplo TkAgg) para explorar los parámetros."
        )

    fig, _sliders, parametros = _preparar_explorador(roi, metodo_sombras)
    plt.show(block=True)  # _sliders debe seguir vivo mientras la ventana esté abierta
    plt.close(fig)

    print("── Parámetros seleccionados ──")
    for nombre, valor in parametros.items():
        texto = f"'{valor}'" if isinstance(valor, str) else (f"{valor:g}")
        print(f"{nombre:<15}= {texto}")
    return dict(parametros)


# Orden de los pasos y el parámetro que controla cada uno (para recalcular solo lo necesario).
_PASOS = (
    ("suave", "sigma"),
    ("Sin sombras", "sombras"),
    ("CLAHE", "clip_limit"),
    ("tanh", "factor_tanh"),
)


def _procesar(
    roi: np.ndarray, parametros: dict, previo: dict | None = None, cambio: str | None = None
) -> dict[str, np.ndarray]:
    """
    Corre el pipeline y retorna la imagen de cada paso.

    Si se pasa `previo` (resultado anterior) y `cambio` (parámetro modificado: 'sigma',
    'sombras', 'clip_limit' o 'factor_tanh'), reutiliza los pasos anteriores a ese
    parámetro y recalcula solo desde ahí.
    """
    claves = [parametro for _, parametro in _PASOS]
    desde = claves.index(cambio) if (previo is not None and cambio in claves) else 0
    pasos = dict(previo) if desde > 0 else {"Original": roi}

    if desde <= 0:
        pasos["suave"] = suavizar(roi, parametros["sigma"])
    if desde <= 1:
        if parametros["metodo_sombras"] == "blur_resta":
            pasos["Sin sombras"] = corregir_sombras(
                pasos["suave"], "blur_resta", fraccion=parametros["fraccion"]
            )
        else:
            pasos["Sin sombras"] = corregir_sombras(
                pasos["suave"], "white_tophat", radio=parametros["radio"]
            )
    if desde <= 2:
        pasos["CLAHE"] = ecualizar(pasos["Sin sombras"], "clahe", clip_limit=parametros["clip_limit"])
    pasos["tanh"] = normalizar_tanh(pasos["CLAHE"], factor=parametros["factor_tanh"])
    return pasos


def _preparar_explorador(roi: np.ndarray, metodo_sombras: str):
    """
    Crea la figura con los 4 sliders. Retorna (fig, sliders, parametros), donde `parametros`
    se actualiza con cada cambio. Separada de `explorar_preprocesamiento` para poder
    probarla sin abrir ventanas.
    """
    parametros = {"sigma": 1.0, "metodo_sombras": metodo_sombras}
    if metodo_sombras == "blur_resta":
        parametros["fraccion"] = 0.03
        slider_sombras = ("fraccion", "Fracción sombras", 0.01, 0.10, 0.03, None)
    else:
        parametros["radio"] = 50
        slider_sombras = ("radio", "Radio tophat (px)", 5, 150, 50, 1)
    parametros.update({"clip_limit": 0.01, "factor_tanh": 1.0})

    fig = plt.figure(figsize=(15, 9))
    grilla = fig.add_gridspec(2, 3)
    ejes = {
        "Original": fig.add_subplot(grilla[0, 0]),
        "Sin sombras": fig.add_subplot(grilla[0, 1]),
        "CLAHE": fig.add_subplot(grilla[0, 2]),
        "tanh": fig.add_subplot(grilla[1, 0]),
    }
    ax_hist = fig.add_subplot(grilla[1, 1])
    grilla_sliders = grilla[1, 2].subgridspec(4, 1, hspace=1.2)

    pasos = _procesar(roi, parametros)
    rangos = {"Original": (0, 1), "CLAHE": (0, 1), "tanh": (-1, 1)}
    imagenes = {}
    for nombre, ax in ejes.items():
        vmin, vmax = rangos.get(nombre, (pasos[nombre].min(), pasos[nombre].max()))
        imagenes[nombre] = ax.imshow(pasos[nombre], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(nombre)
        ax.set_axis_off()

    # 64 bins: menos que los ~100 niveles de una ROI que viene de uint8 (evita el "peine").
    bins = np.linspace(0, 1, 65)
    centros = (bins[:-1] + bins[1:]) / 2
    lineas = {}
    for nombre in ejes:
        (lineas[nombre],) = ax_hist.plot(
            centros, _histograma_normalizado(pasos[nombre], bins), label=nombre
        )
    ax_hist.set_title("Histogramas (cada paso normalizado a [0, 1])", fontsize=10)
    ax_hist.set_xlim(0, 1)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    definiciones = [
        ("sigma", "Sigma blur", 0.5, 5.0, 1.0, None),
        slider_sombras,
        ("clip_limit", "CLAHE clip", 0.001, 0.03, 0.01, None),
        ("factor_tanh", "Factor tanh", 0.5, 3.0, 1.0, None),
    ]
    sliders = {}
    for i, (clave, etiqueta, minimo, maximo, inicial, paso) in enumerate(definiciones):
        ax_slider = fig.add_subplot(grilla_sliders[i])
        ax_slider.set_title(etiqueta, fontsize=10, loc="left")
        sliders[clave] = Slider(ax_slider, "", minimo, maximo, valinit=inicial, valstep=paso)

    def actualizar(clave: str) -> None:
        slider = sliders[clave]
        nuevo = int(slider.val) if clave == "radio" else float(slider.val)
        if parametros[clave] == nuevo:
            return
        parametros[clave] = nuevo
        cambio = "sombras" if clave in ("fraccion", "radio") else clave
        nuevas = _procesar(roi, parametros, previo=pasos, cambio=cambio)
        pasos.update(nuevas)
        for nombre, imagen in imagenes.items():
            imagen.set_data(nuevas[nombre])
            if nombre not in rangos:
                imagen.set_clim(nuevas[nombre].min(), nuevas[nombre].max())
        for nombre, linea in lineas.items():
            linea.set_ydata(_histograma_normalizado(nuevas[nombre], bins))
        ax_hist.relim()
        ax_hist.autoscale_view(scalex=False)
        fig.canvas.draw_idle()

    for clave, slider in sliders.items():
        slider.on_changed(lambda _valor, clave=clave: actualizar(clave))

    def al_presionar_tecla(evento) -> None:
        if evento.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", al_presionar_tecla)
    fig.suptitle("Ajustá los parámetros · Enter o cerrar la ventana para confirmar", fontsize=11)
    return fig, sliders, parametros


def _histograma_normalizado(valores: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Histograma (densidad) de los valores reescalados min-max a [0, 1]."""
    rango = np.ptp(valores)
    reescalados = (valores - valores.min()) / rango if rango > 0 else np.zeros_like(valores)
    conteos, _ = np.histogram(reescalados, bins=bins, density=True)
    return conteos
