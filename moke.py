"""Detección del dominio interior en imágenes MOKE.

Uso básico:

    import skimage as ski
    from dominio_moke import detectar_dominio, mostrar_dominio

    img = ski.io.imread("7.png")
    res = detectar_dominio(img)
    mostrar_dominio(img, res)
    print(res["centro"], res["radio"])

Para una carpeta entera (figuras + tabla medidas.csv):

    from dominio_moke import analizar_carpeta

    filas = analizar_carpeta("evolucion_dominio", escala=0.5, unidad="nm")
"""

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as P
from scipy import ndimage as ndi
from skimage import filters, io, measure, morphology, segmentation, transform


def detectar_dominio(img, semilla=None, referencia=None, escala=4, sigma=12,
                     margen=80, suavizado=24, orden=2):
    """Detecta el dominio interior (más oscuro que el fondo) de una imagen MOKE.

    Parámetros (las distancias van en píxeles de la imagen original):
      img        : imagen en escala de grises (matriz 2D)
      semilla    : (fila, columna) de un punto dentro del dominio. Si no se da,
                   se busca la mancha oscura más grande.
      referencia : imagen de la misma zona SIN dominio (opcional). Si se da,
                   se usa como fondo en lugar de estimarlo.
      escala     : se trabaja con la imagen reducida `escala` veces (más rápido)
      sigma      : suavizado para quitar ruido
      margen     : franja que se descarta junto al borde de la apertura
      suavizado  : tamaño de las irregularidades del borde que se eliminan
      orden      : orden del polinomio que modela la iluminación de fondo

    Devuelve un diccionario con:
      contorno  : puntos (fila, columna) del borde del dominio
      centro    : (fila, columna) del círculo ajustado al contorno
      radio     : radio de ese círculo
      area      : área del dominio en píxeles
      contraste : diferencia relativa dominio/fondo (-0.12 = 12 % más oscuro)
      ruido     : variación relativa del fondo (para comparar con el contraste)
      mascara   : matriz de True/False del tamaño de la imagen
    """
    chica = transform.rescale(img.astype(float), 1 / escala, anti_aliasing=True)
    disco = morphology.disk(suavizado // escala)

    # 1) Apertura: la zona iluminada (el "círculo grande" de afuera)
    apertura = chica > filters.threshold_otsu(chica)
    etiquetas = measure.label(apertura)
    mayor = np.bincount(etiquetas.ravel())[1:].argmax() + 1
    apertura = ndi.binary_fill_holes(etiquetas == mayor)
    # Nos alejamos del borde de la apertura, que está desenfocado
    interior = morphology.erosion(apertura, morphology.disk(margen // escala))

    # 2) Suavizar para quitar ruido
    suave = filters.gaussian(chica, sigma / escala)

    # 3) Iluminación de fondo
    filas, cols = np.mgrid[: chica.shape[0], : chica.shape[1]]
    if referencia is not None:
        ref = transform.rescale(referencia.astype(float), 1 / escala, anti_aliasing=True)
        fondo_img = filters.gaussian(ref, sigma / escala)
    else:
        # Polinomio 2D ajustado "por arriba": en cada vuelta, los píxeles que
        # quedan debajo del ajuste (dominio, suciedad) pesan muy poco, así el
        # polinomio sigue solo al fondo claro.
        V = P.polyvander2d(filas / chica.shape[0] - 0.5, cols / chica.shape[1] - 0.5, [orden, orden])
        Vi, zi = V[interior], suave[interior]
        peso = np.ones(zi.size)
        for _ in range(15):
            coef, *_ = np.linalg.lstsq(Vi * peso[:, None], zi * peso, rcond=None)
            peso = np.where(zi < Vi @ coef, 0.2, 1.0)
        fondo_img = V @ coef
    # `plana`: ~0 en el fondo y negativa donde la imagen es más oscura
    plana = np.zeros_like(suave)
    plana[interior] = suave[interior] / fondo_img[interior] - 1

    # 4) Zona de partida: alrededor de la semilla o, si no hay semilla, la
    #    mancha oscura más grande que no toca el borde
    if semilla is not None:
        zona = np.hypot(filas - semilla[0] / escala, cols - semilla[1] / escala) <= suavizado / escala
        zona &= interior
    else:
        oscuro = interior & (plana < filters.threshold_otsu(plana[interior]))
        etiquetas = segmentation.clear_border(measure.label(morphology.opening(oscuro, disco)), mask=interior)
        zona = etiquetas == np.bincount(etiquetas.ravel(), minlength=2)[1:].argmax() + 1
    if not zona.any():
        raise ValueError("No se encontró el dominio. Probá indicar una semilla dentro del dominio.")

    # 5) Se repite 3 veces: medir los niveles de gris del dominio y del fondo,
    #    poner el umbral a mitad de camino y quedarse con la mancha oscura que
    #    contiene a la zona de partida. Cada vuelta parte de la mancha anterior.
    fondo = np.percentile(plana[interior], 95)
    for _ in range(3):
        nivel = np.median(plana[zona])
        for _ in range(3):
            umbral = (fondo + nivel) / 2
            fondo = np.median(plana[interior & (plana > umbral)])
        umbral = (fondo + nivel) / 2
        etiquetas = measure.label(morphology.opening(interior & (plana < umbral), disco))
        cuenta = np.bincount(etiquetas[zona], minlength=2)[1:]
        if cuenta.sum() == 0:
            raise ValueError("La semilla no cae en una zona oscura. Probá con otro punto.")
        zona = ndi.binary_fill_holes(etiquetas == cuenta.argmax() + 1)
    dominio = morphology.closing(zona, disco)
    ruido = 1.4826 * np.median(np.abs(plana[interior & (plana > umbral)] - fondo))

    # Avisos cuando el resultado es dudoso
    if abs(nivel - fondo) < 3 * ruido:
        warnings.warn("El contraste es bajo comparado con el ruido: revisá el resultado.")
    if (dominio & ~morphology.erosion(interior, morphology.disk(2))).any():
        warnings.warn("El dominio llega al borde de la apertura: el contorno puede estar cortado.")

    # 6) Contorno (el "edge") y círculo que mejor lo ajusta (mínimos cuadrados)
    contorno = max(measure.find_contours(dominio.astype(float), 0.5), key=len)
    contorno = (contorno + 0.5) * escala - 0.5  # a píxeles de la imagen original
    f, c = contorno[:, 0], contorno[:, 1]
    A = np.column_stack([2 * f, 2 * c, np.ones_like(f)])
    (f0, c0, k), *_ = np.linalg.lstsq(A, f**2 + c**2, rcond=None)

    return {
        "contorno": contorno,
        "centro": (float(f0), float(c0)),
        "radio": float(np.sqrt(k + f0**2 + c0**2)),
        "area": int(dominio.sum()) * escala**2,
        "contraste": float(nivel - fondo),
        "ruido": float(ruido),
        "mascara": transform.resize(dominio, img.shape, order=0),
    }


def mostrar_dominio(img, res, ax=None):
    """Muestra la imagen con el contorno (rojo) y el círculo ajustado (celeste)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    # La imagen se ve casi negra si no se estira el contraste
    claro = img[img > filters.threshold_otsu(img)]
    ax.imshow(img, cmap="gray", vmin=np.percentile(claro, 1), vmax=np.percentile(claro, 99))
    ax.plot(res["contorno"][:, 1], res["contorno"][:, 0], "r-", lw=1, label="contorno")
    f0, c0 = res["centro"]
    ax.add_patch(plt.Circle((c0, f0), res["radio"], fill=False, color="cyan", ls="--", label="círculo ajustado"))
    ax.plot(c0, f0, "c+")
    ax.set_title(f"centro = ({f0:.0f}, {c0:.0f})   radio = {res['radio']:.0f} px")
    ax.legend(loc="upper right")
    ax.axis("off")
    return ax


def analizar_carpeta(carpeta, salida="resultados", escala=1.0, unidad="px"):
    """Detecta el dominio en todas las imágenes .png de una carpeta.

    Los archivos deben llamarse con números (0.png, 1.png, ...) y el dominio
    debe crecer desde el mismo lugar: la última imagen da la semilla.

      carpeta : carpeta con las imágenes
      salida  : carpeta donde se guardan una figura por imagen y medidas.csv
      escala  : tamaño de un píxel, en la unidad indicada
      unidad  : nombre de esa unidad, para los títulos de la tabla ("nm", "um")

    Devuelve las filas de la tabla (la primera son los títulos).
    """
    carpeta, salida = Path(carpeta), Path(salida)
    salida.mkdir(exist_ok=True)
    archivos = sorted(carpeta.glob("*.png"), key=lambda f: int(f.stem))

    # La última imagen (dominio más grande) da un punto de adentro para las demás
    semilla = detectar_dominio(io.imread(archivos[-1]))["centro"]

    filas = [["archivo", "fila_centro_px", "col_centro_px", "radio_px",
              f"radio_{unidad}", f"area_{unidad}2", "contraste", "dudoso"]]
    for f in archivos:
        img = io.imread(f)
        try:
            res = detectar_dominio(img, semilla=semilla)
        except ValueError as error:
            print(f.name, "-> no se detectó dominio:", error)
            continue

        ax = mostrar_dominio(img, res)
        ax.figure.savefig(salida / f"dominio_{f.stem}.png", dpi=150, bbox_inches="tight")
        plt.show()

        dudoso = abs(res["contraste"]) < 3 * res["ruido"]
        filas.append([f.name, round(res["centro"][0], 1), round(res["centro"][1], 1),
                      round(res["radio"], 1), round(escala * res["radio"], 1),
                      round(escala**2 * res["area"], 1), round(res["contraste"], 3), dudoso])

    with open(salida / "medidas.csv", "w", newline="") as tabla:
        csv.writer(tabla).writerows(filas)
    return filas
