# TFG-LIME

=======================                  C A S T E L L A N O                    =========================

Trabajo final de grado. Explicación LIME en redes neuronales para reconocimiento facial.

Las redes neuronales especializadas en reconocimiento facial se están extendiendo en todas las tecnologías y ámbitos de la vida cotidiana. Sin embargo, los modelos son percibidos como cajas negras cuyo funcionamiento interno mantiene cierto misterio. Por lo tanto, ha cobrado un gran interés poder entender el por qué de sus resultados y cuál es el razonamiento que aplican.
En el presente proyecto se trabaja con una técnica de explicabilidad innovativa: Local Interpretable Model-agnostic Explanations (LIME). Ofrece explicaciones indivi- duales para cada muestra que ayudan a entender por qué un modelo da una predicción. Este trabajo emplea un enfoque diferente para el ámbito de la biometría facial. Funda- mentándose en el cálculo de distancias entre imágenes, se obtienen explicaciones de las regiones faciales relevantes sin necesidad de encorsetarse en tareas de clasificación.
Esto se consigue utilizando un planteamiento basado en la distancia del coseno en- tre los vectores de características obtenidos de cada imágen. Ahora se utilizan métricas de distancia, resultando en una puntuación que indica la similitud entre las imágenes. Se sigue un proceso de manipulación de los resultados hasta generar mapas de calor que resumen los rasgos faciales más identificativos.
Los resultados obtenidos muestran ciertas divergencias entre las redes, sobre todo aquellas con mayor diferencia en el número de capas. Por lo general todas indicaron gran fijación en la zona de la nariz. Algunos modelos remarcaron algunos rasgos más que otros y el área de reconocimiento también fluctuaba según la profundidad. Los modelos mostraron también ciertas diferencias por etnia y sexo.
Se podría aprovechar este nuevo enfoque basado en la distancia del coseno para impulsar nuevos estudios relativos a la biometría facial. Una idea podría ser forzar redes a fijarse en rasgos concretos y ver su desempeño. Otra, la importancia de la calidad de imágen para el éxito en la clasificación. También se pueden combinar redes para diferentes tipos de detección. Estas son algunas de las muchas ideas que se pueden explorar en un futuro.

Dentro de la carpeta del proyecto principal deben estar estas tres carpetas, que no están subidas en el repositorio debido a contar con un peso excesivo de memoria.
- Models
- resources
- Datos

En principio odo este material adicional se puede encontrar y descargar fácilmente en este enlace de drive: https://drive.google.com/drive/folders/1FrBMYl6eoBmATRG4wk3BWuTouRE761b8?usp=sharing

A continuación se muestra cómo se han obtenido de manera individual cada recurso:

# Models

Los modelos se descargaron de este repositorio de github: https://github.com/deepinsight/insightface/tree/master/model_zoo
Se escogieron las ResNet entrenadas con ArcFace. En la imágen se pueden ver los modelos escogidos. Todas menos R100 Casia y R100 MS1MV2. 
<img width="849" alt="Captura de pantalla 2023-08-29 a las 16 15 39" src="https://github.com/maribel95/TFG-LIME/assets/61268027/cd569c30-6cad-4af6-92e4-0a7b1a5a55c5">


# resources

El primer recurso que se necesita es el reconocer de caras haarcascades. Se puede descargar en el siguiente enlace y sencillamente se coloca dentro de la carpeta resources.
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

El segundo que se necesita es el archivo de detección de 68 landmarks de la cara:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat



# Datos

Los datos fueron proporcionados por el investigador Fernando Alonso Fernández. Básicamente es un subconjunto de la base de datos VGG2Face Pertenecen a esta carpeta drive: https://drive.google.com/drive/folders/1cu7eGTg2zqQPEyreqvt4BGLWGca5MNB9

<img width="751" alt="Captura de pantalla 2023-08-29 a las 16 33 34" src="https://github.com/maribel95/TFG-LIME/assets/61268027/bff05eee-ef16-4d04-9834-3816a2661020">

Aunque cabe recalcar que se hicieron una serie de adaptaciones para el presente proyecto. Subieron un directorio todas las carpetas de los individuos y se eliminó la carpeta test. Así que la estructura para acceder a las carpetas de cada uno de los usuarios(formato nxxxxxx) queda así.


<img width="130" alt="Captura de pantalla 2023-08-29 a las 16 35 43" src="https://github.com/maribel95/TFG-LIME/assets/61268027/6fc52846-c282-4fec-9f0b-740fec8d580b">

# Flujo de trabajo


Este trabajo consiste en el análisis de redes neuronales para poder entender en qué se fijan para reconocer caras. Para ello, se aplica el método LIME. La idea consiste en obtener explicaciones en varios ejemplos de sujetos individuales. Con una muestra lo suficientemente grande, se puede alcanzar una mejor comprensión del funcionamiento global del modelo.
El objetivo principal es llevar a cabo un estudio estadístico y analizar cómo fun- cionan las redes neuronales especializadas en el reconocimiento facial. Es sugerente explorar si las conclusiones a las que llegan estas redes tienen cierto criterio y no surgen del azar; si cada una se fija en los mismos rasgos, cuáles son los más importantes y los posibles sesgos de preentrenamiento que puedan subyacer.
Para poder poder llevar a cabo el estudio, se ha seguido un flujo de trabajo ordenado y estructurado. Lo primero es decidir qué modelos utilizar y qué base de datos elegir. Luego se procede a aplicar las explicaciones individuales de LIME de todos y cada uno de los sujetos. Se obtienen las máscaras mediante dicho método, utilizando el nuevo enfoque basado en la similitud del coseno y se consiguen las regiones de la cara con mayor importancia para cada persona. Se normalizan las máscaras obtenidas. Para ello, se aplican una serie de landmarks que detectan mismos puntos en cada cara; posteriormente se conectan y alinean. En particular, se detectan los siguientes rasgos anatómicos: cejas, ojos, nariz, labios y el contorno. Las transformaciones colocan los rasgos en las mismas regiones de píxeles para posteriormente obtener mapas de calor. En resumen, se obtienen las máscaras de mayor importancia, se normalizan y a continuación se realizan mapas de calor que muestren aquellas zonas de la cara con mayor influencia para las redes para el reconocimiento. Finalmente, se realizan varios experimentos que muestran estos mapas de calor según diferentes enfoques. Y a través de la divergencia Kullback–Leibler, se obtienen los dendrogramas relativos para sacar conclusiones de todo el conjunto de datos obtenidos.

Las tareas a realizar son las siguientes: 

- [X] Creación máscaras LIME.
- [X] Normalización caras y máscaras
- [X] Creación mapas de calor
- [X] Creación gráficos dendrogramas.

En la siguiente imagen queda reflejado todo el proceso a seguir:

<img width="994" alt="Captura de pantalla 2023-11-15 a las 9 46 17" src="https://github.com/maribel95/TFG-LIME/assets/61268027/8ed23970-337b-4fbf-ba90-52ab00ee52ed">


