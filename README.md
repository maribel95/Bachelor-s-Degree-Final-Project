# TFG-LIME

=======================                  C A S T E L L A N O                    =========================

Trabajo final de grado. Explicación LIME en redes neuronales para reconocimiento facial.

Las redes neuronales especializadas en reconocimiento facial se están extendiendo en todas las tecnologías y ámbitos de la vida cotidiana. Sin embargo, los modelos son percibidos como cajas negras cuyo funcionamiento interno mantiene cierto misterio. Por lo tanto, ha cobrado un gran interés poder entender el por qué de sus resultados y cuál es el razonamiento que aplican.
En el presente proyecto se trabaja con una técnica de explicabilidad innovativa: Local Interpretable Model-agnostic Explanations (LIME). Ofrece explicaciones indivi- duales para cada muestra que ayudan a entender por qué un modelo da una predicción. Este trabajo emplea un enfoque diferente para el ámbito de la biometría facial. Funda- mentándose en el cálculo de distancias entre imágenes, se obtienen explicaciones de las regiones faciales relevantes sin necesidad de encorsetarse en tareas de clasificación.
Esto se consigue utilizando un planteamiento basado en la distancia del coseno en- tre los vectores de características obtenidos de cada imágen. Ahora se utilizan métricas de distancia, resultando en una puntuación que indica la similitud entre las imágenes. Se sigue un proceso de manipulación de los resultados hasta generar mapas de calor que resumen los rasgos faciales más identificativos.
Los resultados obtenidos muestran ciertas divergencias entre las redes, sobre todo aquellas con mayor diferencia en el número de capas. Por lo general todas indicaron gran fijación en la zona de la nariz. Algunos modelos remarcaron algunos rasgos más que otros y el área de reconocimiento también fluctuaba según la profundidad. Los modelos mostraron también ciertas diferencias por etnia y sexo.
Se podría aprovechar este nuevo enfoque basado en la distancia del coseno para impulsar nuevos estudios relativos a la biometría facial. Una idea podría ser forzar redes a fijarse en rasgos concretos y ver su desempeño. Otra, la importancia de la calidad de imágen para el éxito en la clasificación. También se pueden combinar redes para diferentes tipos de detección. Estas son algunas de las muchas ideas que se pueden explorar en un futuro.

# Flujo de trabajo


Este trabajo consiste en el análisis de redes neuronales para poder entender en qué se fijan para reconocer caras. Para ello, se aplica el método LIME. La idea consiste en obtener explicaciones en varios ejemplos de sujetos individuales. Con una muestra lo suficientemente grande, se puede alcanzar una mejor comprensión del funcionamiento global del modelo.
El objetivo principal es llevar a cabo un estudio estadístico y analizar cómo fun- cionan las redes neuronales especializadas en el reconocimiento facial. Es sugerente explorar si las conclusiones a las que llegan estas redes tienen cierto criterio y no surgen del azar; si cada una se fija en los mismos rasgos, cuáles son los más importantes y los posibles sesgos de preentrenamiento que puedan subyacer.
Para poder poder llevar a cabo el estudio, se ha seguido un flujo de trabajo ordenado y estructurado. Lo primero es decidir qué modelos utilizar y qué base de datos elegir. Luego se procede a aplicar las explicaciones individuales de LIME de todos y cada uno de los sujetos. Se obtienen las máscaras mediante dicho método, utilizando el nuevo enfoque basado en la similitud del coseno y se consiguen las regiones de la cara con mayor importancia para cada persona. Se normalizan las máscaras obtenidas. Para ello, se aplican una serie de landmarks que detectan mismos puntos en cada cara; posteriormente se conectan y alinean. En particular, se detectan los siguientes rasgos anatómicos: cejas, ojos, nariz, labios y el contorno. Las transformaciones colocan los rasgos en las mismas regiones de píxeles para posteriormente obtener mapas de calor. En resumen, se obtienen las máscaras de mayor importancia, se normalizan y a continuación se realizan mapas de calor que muestren aquellas zonas de la cara con mayor influencia para las redes para el reconocimiento. Finalmente, se realizan varios experimentos que muestran estos mapas de calor según diferentes enfoques. Y a través de la divergencia Kullback–Leibler, se obtienen los dendrogramas relativos para sacar conclusiones de todo el conjunto de datos obtenidos.

Las tareas a realizar son las siguientes: 

- [X] Creación máscaras LIME.
- [X] Normalización caras y máscaras.
- [X] Creación mapas de calor.
- [X] Creación gráficos dendrogramas.

En la siguiente imagen queda reflejado todo el proceso a seguir:
<img width="985" alt="Captura de pantalla 2023-11-15 a las 11 10 21" src="https://github.com/maribel95/TFG-LIME/assets/61268027/2aaab47a-3507-454a-a035-5dd53ff84f70">

### Explicaciones LIME
Por lo general, los espacios de características con los que los modelos de aprendizaje profundo hacen frente, presentan límites no lineales y de alta complejidad. Para dar una explicación a estos modelos, se utiliza LIME, que es una técnica de explicabilidad novedosa que apareció por primera vez en 2016 y cuya intención es esclarecer las predicciones de cualquier clasificador. La idea es construir un modelo local simple y fácil de interpretar sobre predicciones particulares. A partir de diversas muestras representativas, se extrapolan los resultados a un plano más general, dando así una explicación global del modelo .

- Lo primero que se hace es generar varias muestras perturbadas a su alrededor. Estas muestras presentan ligeras variaciones respecto a la original, por lo que todas se ubican en un espacio dimensional muy cercano.
- A continuación, se utiliza una función kernel que asigne pesos al subconjunto de muestras perturbadas, según la distancia respecto a la original. De esta manera, se pueden saber qué muestras son más similares a la instancia inicial.
- El siguiente paso es entrenar un modelo simple. Este genera un resultado para el subconjunto de datos particular, dando una explicación interpretable, aunque muy específica. Entre las opciones del modelo local se encuentran los de regresión lineal, árboles de decisión o aquellos basados en reglas.
- Finalmente, se analizan los coeficientes producidos por el modelo, mostrando aquellos factores que contribuyen más en la predicción. Por lo tanto, se obtienen las características más importantes que definen la instancia original.

En el contexto de la clasificación de imágenes, la porción de vecindad se genera a través de varias muestras con diferentes combinaciones de regiones ocluidas. Luego se analiza la importancia que tienen para la imagen original. De manera que si el parche tapaba una parte importante, esto afecta directamente a la predicción de la clase. Se utiliza un vector binario que indica la presencia o ausencia de esos parches en las nuevas imágenes distorsionadas.

El proceso de tratamiento de datos es el siguiente: 
<img width="975" alt="Captura de pantalla 2023-11-15 a las 10 44 46" src="https://github.com/maribel95/TFG-LIME/assets/61268027/45675563-14aa-4dcd-91e3-611d4ffda69e">

Y el proceso de tratamiento de imágenes es este:
<img width="985" alt="Captura de pantalla 2023-11-15 a las 10 48 04" src="https://github.com/maribel95/TFG-LIME/assets/61268027/9805cdbf-22c0-4bbd-99ef-5e989485a4f2">


### Normalización imágenes

Una vez obtenidas las explicaciones LIME, el siguiente paso es la normalización para alinear y estandarizar las caras. Se busca reducir la variabilidad en las poses y en las expresiones faciales, de manera que todas las imágenes mantengan una misma orientación y alineación.

En esta fase se deben realizar las siguientes tareas:
- [X] Detección facial. El primer paso para la normalización es aplicar un detector de caras que focalice la persona e ignore objetos que puedan suponer una distracción a la ahora de aplicar landmarks. En este caso se utiliza el detector de caras Haarcascade de la librería de OpenCV de Python.
- [X] Detección Landmarks. La identificación exacta de puntos de referencia dentro de la cara es el segundo paso para normalizar. Se aplican detectores de referencias faciales que colocan los rasgos en una plantilla uniforme e idéntica para todos los demás rostros.
Se utiliza un enfoque que involucra un descriptor que detecta 68 landmarks faciales. Se adapta a cada cara distinta, centrándose en los puntos más representativos: boca, ceja derecha e izquierda, ojo derecho e izquierdo, nariz y mandíbula.
- [X] Creación malla mediante triangularización Delaunay. El último paso es la creación de una malla formada por triángulos que cumplan la propiedad de Delaunay y que conecte los puntos de referencia de la cara. Los landmarks actúan como puntos de control, son los conectores de los vértices y es alrededor de estos donde se construyen los triángulos.

Los resultados serían los siguientes:

<img width="985" alt="Captura de pantalla 2023-11-15 a las 10 48 04" src="https://github.com/maribel95/TFG-LIME/assets/61268027/11e75800-0183-487b-ab27-5e0b0be0d276">


# Recursos del proyecto:

Dentro de la carpeta del proyecto principal deben estar estas tres carpetas, que no están subidas en el repositorio debido a contar con un peso excesivo de memoria.
- Models
- resources
- Datos

En principio odo este material adicional se puede encontrar y descargar fácilmente en este enlace de drive: https://drive.google.com/drive/folders/1FrBMYl6eoBmATRG4wk3BWuTouRE761b8?usp=sharing

A continuación se muestra cómo se han obtenido de manera individual cada recurso:

### Models

Los modelos se descargaron de este repositorio de github: https://github.com/deepinsight/insightface/tree/master/model_zoo
Se escogieron las ResNet entrenadas con ArcFace. En la imágen se pueden ver los modelos escogidos. Todas menos R100 Casia y R100 MS1MV2. 
<img width="849" alt="Captura de pantalla 2023-08-29 a las 16 15 39" src="https://github.com/maribel95/TFG-LIME/assets/61268027/cd569c30-6cad-4af6-92e4-0a7b1a5a55c5">


### resources

El primer recurso que se necesita es el reconocer de caras haarcascades. Se puede descargar en el siguiente enlace y sencillamente se coloca dentro de la carpeta resources.
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

El segundo que se necesita es el archivo de detección de 68 landmarks de la cara:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat



### Datos

Los datos fueron proporcionados por el investigador Fernando Alonso Fernández. Básicamente es un subconjunto de la base de datos VGG2Face Pertenecen a esta carpeta drive: https://drive.google.com/drive/folders/1cu7eGTg2zqQPEyreqvt4BGLWGca5MNB9

<img width="751" alt="Captura de pantalla 2023-08-29 a las 16 33 34" src="https://github.com/maribel95/TFG-LIME/assets/61268027/bff05eee-ef16-4d04-9834-3816a2661020">

Aunque cabe recalcar que se hicieron una serie de adaptaciones para el presente proyecto. Subieron un directorio todas las carpetas de los individuos y se eliminó la carpeta test. Así que la estructura para acceder a las carpetas de cada uno de los usuarios(formato nxxxxxx) queda así.


<img width="130" alt="Captura de pantalla 2023-08-29 a las 16 35 43" src="https://github.com/maribel95/TFG-LIME/assets/61268027/6fc52846-c282-4fec-9f0b-740fec8d580b">






