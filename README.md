## Titulo
"[Analizador de imágenes de Microscopía]"

## Overview

EL programa contiene herramientas para analizar imágenes obtenidas mediante técnicas de microscopía, incluyendo TEM, SEM y Microscopía MOKE.
Comprende funciones para procesar imágenes, segmentarlas y extraer información relevante sobre la composición de las muestras. 
Además, permite seleccionar funciones en dependencia si estas analizando imágenes de MOKE, o de TEM/SEM.

#Requeriments
El código se desarrollará en Python y puede requerir los siguientes paquetes:

numpy
scipy
matplotlib
scikit-image
exifread

puedes instalar los paquetes necesarios ejecutando el siguiente comando en tu terminal o línea de comandos:

pip install numpy scipy matplotlib scikit-image

# Input data
- Las entradas del programa son imágenes de microscopía en formatos estándar (por ejemplo, .tif) de datos experimentales.
- Es importante conocer que tipo de imagen se está analizando, ya que el programa ofrece diferentes funciones según si se trata de imágenes de MOKE o de TEM/SEM.
- Para el caso de composiciones habría que entrarle la composición de la muestra, por ejemplo: Fe y Ni.

# Output data
- Los resultados del análisis se presentan en forma de gráficos y tablas, que incluyen ajustes de curvas
- Concentraciones relativas de los elementos presentes en la muestra, obtenidas a partir de las imágenes de microscopía y los datos experimentales.
- Imágenes segmentadas y procesadas, que muestren cada elemento por separado.
- Para imágenes de MOKE, se pueden obtener mapas de contraste magnético y tamaños de dominio.

# Additional information

## Author
- [Mauricio Romano]
- [Daniel Romero]
- [Angel A. Perez]

## Contact
Para consultas relacionadas con el conjunto de datos, puede comunicarse con los autores a través de los siguientes correos electrónicos:  
- [mauricio.romano@ib.edu.ar]
- [daniel.romero@ib.edu.ar]
- [angel.perez@ib.edu.ar]
