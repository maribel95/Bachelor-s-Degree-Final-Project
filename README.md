# LIME explanation in neural networks for facial recognition.

<img width="978" alt="Captura de pantalla 2023-11-15 a las 17 01 46" src="https://github.com/maribel95/TFG-LIME/assets/61268027/5b8758ab-9d33-49b1-831f-f70c81a9440c">

Final degree project. LIME explanation in neural networks for facial recognition.

Neural networks specialized in facial recognition are spreading in all technologies and areas of daily life. However, models are perceived as black boxes whose internal functioning maintains a certain mystery. Therefore, it has become of great interest to understand the reason for their results and what reasoning they apply.
In this project we work with an innovative explainability technique: Local Interpretable Model-agnostic Explanations (LIME). It offers individual explanations for each sample that help understand why a model gives a prediction. This work uses a different approach for the area of facial biometrics. Based on the calculation of distances between images, explanations of the relevant facial regions are obtained without the need to confine ourselves to classification tasks.
This is achieved using an approach based on the cosine distance between the feature vectors obtained from each image. Distance metrics are now used, resulting in a score that indicates the similarity between the images. A process of manipulating the results is followed to generate heat maps that summarize the most identifying facial features.
The results obtained show certain divergences between the networks, especially those with the greatest difference in the number of layers. In general, all of them indicated great fixation in the nose area. Some models highlighted some features more than others and the recognition area also fluctuated depending on depth. The models also showed certain differences by ethnicity and sex.
This new cosine distance-based approach could be leveraged to drive new studies related to facial biometrics. One idea could be to force networks to look at specific traits and see their performance. Another, the importance of image quality for success in classification. Networks can also be combined for different types of detection. These are some of the many ideas that can be explored in the future.


# Workflow


This work consists of the analysis of neural networks to understand what they look at to recognize faces. For this, the LIME method is applied. The idea is to obtain explanations in several examples of individual subjects. With a large enough sample, a better understanding of the overall performance of the model can be achieved.
The main objective is to carry out a statistical study and analyze how neural networks specialized in facial recognition work. It is suggestive to explore whether the conclusions reached by these networks have certain criteria and do not arise from chance; If each one focuses on the same traits, which are the most important and the possible pre-training biases that may underlie them.
In order to carry out the study, an orderly and structured work flow has been followed. The first thing is to decide which models to use and which database to choose. Then we proceed to apply the individual LIME explanations of each and every one of the subjects. The masks are obtained by this method, using the new approach based on cosine similarity and the regions of the face with the greatest importance for each person are obtained. The masks obtained are normalized. To do this, a series of landmarks are applied that detect the same points on each face; They are then connected and aligned. In particular, the following anatomical features are detected: eyebrows, eyes, nose, lips and contour. The transformations place the features in the same pixel regions to later obtain heat maps. In summary, the most important masks are obtained, normalized and then heat maps are created that show those areas of the face with the greatest influence on the recognition networks. Finally, several experiments are performed showing these heat maps according to different approaches. And through the Kullback–Leibler divergence, relative dendrograms are obtained to draw conclusions from the entire set of data obtained.

The tasks to be performed are as follows:

- [X] LIME masks creation.
- [X] Faces and masks normalization.
- [X] Heatmaps creation.
- [X] Dendogram graphics creation.

The following image shows the entire process to follow:



<img width="985" alt="Captura de pantalla 2023-11-15 a las 11 10 21" src="https://github.com/maribel95/TFG-LIME/assets/61268027/a68f3d76-b5ac-46e7-88ac-fc387f7884f9">

### LIME explanations
In general, the feature spaces that deep learning models deal with have non-linear boundaries and high complexity. To explain these models, LIME is used, which is a novel explainability technique that first appeared in 2016 and whose intention is to clarify the predictions of any classifier. The idea is to build a simple and easy-to-interpret local model based on particular predictions. Based on various representative samples, the results are extrapolated to a more general level, thus giving a global explanation of the model.

- The first thing to do is generate several disturbed samples around it. These samples present slight variations with respect to the original, so they are all located in a very close dimensional space.
- Next, a kernel function is used to assign weights to the subset of perturbed samples, according to the distance from the original. In this way, you can know which samples are most similar to the initial instance.
- The next step is to train a simple model. This generates a result for the particular subset of data, giving an interpretable, albeit very specific, explanation. Among the local model options are linear regression, decision trees or those based on rules.
- Finally, the coefficients produced by the model are analyzed, showing those factors that contribute most to the prediction. Therefore, the most important characteristics that define the original instance are obtained.

In the context of image classification, the neighborhood portion is generated through multiple samples with different combinations of occluded regions. Then the importance they have for the original image is analyzed. So if the patch covered an important part, this directly affects the class prediction. A binary vector is used that indicates the presence or absence of those patches in the new distorted images.

The data processing process is as follows:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 17 46 58" src="https://github.com/maribel95/TFG-LIME/assets/61268027/0dc611e3-aef8-4cd8-b47f-473ca66ff0a4">


And the image processing process is this:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 17 46 50" src="https://github.com/maribel95/TFG-LIME/assets/61268027/a1256d66-b7d4-45d7-9481-686af3054e67">

### Images normalization

Once the LIME explanations are obtained, the next step is normalization to align and standardize the faces. The aim is to reduce variability in poses and facial expressions, so that all images maintain the same orientation and alignment.

In this phase the following tasks must be carried out:
- [X] Face detection. The first step for normalization is to apply a face detector that focuses on the person and ignores objects that may be a distraction when applying landmarks. In this case, the Haarcascade face detector from the OpenCV Python library is used.
- [X] Landmark Detection. Accurately identifying landmarks within the face is the second step in normalizing. Facial landmark detectors are applied that place features into a uniform and identical template for all other faces.
An approach is used that involves a descriptor that detects 68 facial landmarks. It adapts to each different face, focusing on the most representative points: mouth, right and left eyebrow, right and left eye, nose and jaw.
- [X] Mesh creation using Delaunay triangularization. The last step is the creation of a mesh formed by triangles that satisfy the Delaunay property and that connects the reference points of the face. Landmarks act as control points, they are the connectors of the vertices and it is around these where the triangles are built.

The results would be the following:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 19 35 43" src="https://github.com/maribel95/TFG-LIME/assets/61268027/6bbd3b60-f767-415c-8a59-37985eaef9aa">


### Heatmaps creation

Heat maps are very useful as they facilitate the understanding of large amounts of data, so that they are summarized in patterns that show relationships, variations or disturbances. Only numerical data is used in the traces of both axes on the grid.
In the scope of this work, the distribution of the masks obtained from the entire database is processed. There will be one axis that will represent the height and another the width. The matrix nature of the heat map is equivalent to that of an image.

The advantages of a heat map are the following:
<img width="775" alt="Captura de pantalla 2023-11-15 a las 19 51 33" src="https://github.com/maribel95/TFG-LIME/assets/61268027/a3995f78-510c-4d14-8e23-a6a4007b0ebb">




Data processing follows the following flow of actions:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 19 50 08" src="https://github.com/maribel95/TFG-LIME/assets/61268027/9762c543-cf04-45e8-95f8-d012ed6533cb">

### Dendrogram creation

To compare the heat maps, representation using dendrograms has been chosen. To create these, it is done using the Kullback-Leibler divergence. It is a distance metric that measures the loss of information between two probabilistic distributions. It is used in the field of information theory, statistics, machine learning and inference.
KL is used to see how similar two distribution functions are, where large distances penalize more. The measure of Euclidean distance or the absolute value of the difference does not have a probabilistic meaning, so they are discarded in the scope of this project.

### Heatmap comparison example



<img width="775" alt="Captura de pantalla 2023-11-17 a las 6 55 32" src="https://github.com/maribel95/Bachelor-s-Degree-Final-Project/assets/61268027/cff6be28-ecbb-48cb-9368-70ccb3385b80">

# Project resources:

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











#
<img width="978" alt="Captura de pantalla 2023-11-15 a las 16 49 54" src="https://github.com/maribel95/TFG-LIME/assets/61268027/ccbaa0c0-5bab-481f-abb9-1f6b33eb29e6">


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
<img width="775" alt="Captura de pantalla 2023-11-15 a las 16 21 45" src="https://github.com/maribel95/TFG-LIME/assets/61268027/82e092e8-48ab-4525-8315-0136e9d3080f">


Y el proceso de tratamiento de imágenes es este:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 10 36 52" src="https://github.com/maribel95/TFG-LIME/assets/61268027/0440394e-6ed2-449f-b3cc-eb18f9525da2">


### Normalización imágenes

Una vez obtenidas las explicaciones LIME, el siguiente paso es la normalización para alinear y estandarizar las caras. Se busca reducir la variabilidad en las poses y en las expresiones faciales, de manera que todas las imágenes mantengan una misma orientación y alineación.

En esta fase se deben realizar las siguientes tareas:
- [X] Detección facial. El primer paso para la normalización es aplicar un detector de caras que focalice la persona e ignore objetos que puedan suponer una distracción a la ahora de aplicar landmarks. En este caso se utiliza el detector de caras Haarcascade de la librería de OpenCV de Python.
- [X] Detección Landmarks. La identificación exacta de puntos de referencia dentro de la cara es el segundo paso para normalizar. Se aplican detectores de referencias faciales que colocan los rasgos en una plantilla uniforme e idéntica para todos los demás rostros.
Se utiliza un enfoque que involucra un descriptor que detecta 68 landmarks faciales. Se adapta a cada cara distinta, centrándose en los puntos más representativos: boca, ceja derecha e izquierda, ojo derecho e izquierdo, nariz y mandíbula.
- [X] Creación malla mediante triangularización Delaunay. El último paso es la creación de una malla formada por triángulos que cumplan la propiedad de Delaunay y que conecte los puntos de referencia de la cara. Los landmarks actúan como puntos de control, son los conectores de los vértices y es alrededor de estos donde se construyen los triángulos.

Los resultados serían los siguientes:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 10 48 04" src="https://github.com/maribel95/TFG-LIME/assets/61268027/11e75800-0183-487b-ab27-5e0b0be0d276">

### Creación mapas de calor

Los mapas de calor son muy útiles ya que facilitan la comprensión de gran cantidad de datos, de manera que quedan resumidos en patrones que muestran relaciones, variaciones o perturbaciones. Solo se utilizan datos numéricos en las trazas de ambos ejes en la cuadrícula.
En el ámbito de este trabajo, se procesa la distribución de las máscaras obtenidas del conjunto de la base de datos. Habrá un eje que representará la altura y otro la anchura. La naturaleza matricial del mapa de calor es equivalente a la de una imagen.

Las ventajas de un mapa de calor son las siguientes:
<img width="775" alt="Captura de pantalla 2023-11-15 a las 16 40 53" src="https://github.com/maribel95/TFG-LIME/assets/61268027/7303b5e4-72f9-4c63-82ee-8408491b8c2a">


El procesamiento de los datos sigue el siguiente flujo de acciones:

<img width="775" alt="Captura de pantalla 2023-11-15 a las 16 40 34" src="https://github.com/maribel95/TFG-LIME/assets/61268027/80ab8c8c-bedd-4ff1-b8be-57d1ea2b619b">

### Creación dendrogramas

Para la comparación de los mapas de calor se ha optado por la representación mediante dendrogramas. Para la creación de estos, se hace mediante la divergencia de Kullback-Leibler. Es una métrica de distancia que mide la pérdida de información entre dos distribuciones probabilísticas. Se utiliza en el ámbito de la teoría de la información, la estadística, aprendizaje automático e inferencia.
KL sirve para ver cuánto se parecen dos funciones de distribución, donde penalizan más las grandes distancias. La medida de la distancia euclídea o el valor absoluto de la diferencia no tienen un significado probabilístico, así que quedan descartadas en el ámbito de este proyecto.

### Ejemplo comparación mapas de calor


<img width="775" alt="Captura de pantalla 2023-11-15 a las 16 47 39" src="https://github.com/maribel95/TFG-LIME/assets/61268027/67b574d4-5251-4dd6-9603-51261002336c">



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
<img width="749" alt="Captura de pantalla 2023-08-29 a las 16 15 39" src="https://github.com/maribel95/TFG-LIME/assets/61268027/cd569c30-6cad-4af6-92e4-0a7b1a5a55c5">


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






