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



==========================                  E N G L I S H                    ===========================

Final Degree Project. LIME explanation in neural networks for facial recognition.

Neural networks specialized in facial recognition are spreading in all technologies and areas of daily life. However, models are perceived as black boxes whose internal functioning maintains a certain mystery. Therefore, it has become of great interest to understand the reason for their results and what reasoning they apply.
In this project we work with an innovative explainability technique: Local Interpretable Model-agnostic Explanations (LIME). It offers individual explanations for each sample that help understand why a model gives a prediction. This work uses a different approach for the area of facial biometrics. Based on the calculation of distances between images, explanations of the relevant facial regions are obtained without the need to confine ourselves to classification tasks.
This is achieved using an approach based on the cosine distance between the feature vectors obtained from each image. Distance metrics are now used, resulting in a score that indicates the similarity between the images. A process of manipulating the results is followed to generate heat maps that summarize the most identifying facial features.
The results obtained show certain divergences between the networks, especially those with the greatest difference in the number of layers. In general, all of them indicated great fixation in the nose area. Some models highlighted some features more than others and the recognition area also fluctuated depending on depth. The models also showed certain differences by ethnicity and sex.
This new cosine distance-based approach could be leveraged to drive new studies related to facial biometrics. One idea could be to force networks to look at specific traits and see their performance. Another, the importance of image quality for success in classification. Networks can also be combined for different types of detection. These are some of the many ideas that can be explored in the future.

Within the main project folder there should be these three folders, which are not uploaded to the repository due to having excessive memory weight.
- Models
- resources
- Datos

In principle all this additional material can be easily found and downloaded at this drive link:
https://drive.google.com/drive/folders/1FrBMYl6eoBmATRG4wk3BWuTouRE761b8?usp=sharing

The following shows how each resource has been obtained individually:

# Models

The models were downloaded from this github repository: https://github.com/deepinsight/insightface/tree/master/model_zoo
The ResNets trained with ArcFace were chosen. In the image you can see the chosen models. All except R100 Casia and R100 MS1MV2.
<img width="849" alt="Captura de pantalla 2023-08-29 a las 16 15 39" src="https://github.com/maribel95/TFG-LIME/assets/61268027/cd569c30-6cad-4af6-92e4-0a7b1a5a55c5">

# resources

The first resource that is needed is the recognition of haarcascades faces. It can be downloaded at the following link and is simply placed inside the resources folder.
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

The second one needed is the 68 face landmark detection file:
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat



# Data

The data was provided by the researcher Fernando Alonso Fernández. It's basically a subset of the VGG2Face database. They belong in this drive folder:https://drive.google.com/drive/folders/1cu7eGTg2zqQPEyreqvt4BGLWGca5MNB9

<img width="751" alt="Captura de pantalla 2023-08-29 a las 16 33 34" src="https://github.com/maribel95/TFG-LIME/assets/61268027/bff05eee-ef16-4d04-9834-3816a2661020">

Although it should be noted that a series of adaptations were made for this project. All individual folders were uploaded one directory and the test folder was removed. So the structure to access the folders of each of the users (nxxxxxx format) looks like this.


<img width="130" alt="Captura de pantalla 2023-08-29 a las 16 35 43" src="https://github.com/maribel95/TFG-LIME/assets/61268027/6fc52846-c282-4fec-9f0b-740fec8d580b">


# Workflow












