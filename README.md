# TFG-LIME

=======================                  C A S T E L L A N O                    =========================

Trabajo final de grado. Explicación LIME en redes neuronales para reconocimiento facial.

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



==========================                  E N G L I S H                    ===========================

Final Degree Project. LIME explanation in neural networks for facial recognition.


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















