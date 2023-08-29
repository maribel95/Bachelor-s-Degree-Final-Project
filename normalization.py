from PIL import Image

import numpy as np
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import Delaunay # triangulariza

from matplotlib import pyplot as plt
def transform_mask(predictor, maxx, maxy, vertical_space, image_cropped, image_lime):
    """
    transforma una imagen
    :param predictor: predictor para encontrar los puntos en image_cropped
    :param maxx: anchura maxima
    :param maxy: altura maxima
    :param vertical_space: desplazamiento vertical
    :param image_cropped: imagen con la cara, usada para detectar los puntos
    :param image_lime: imagen a transformar
    :return: imagen LIME normalizada
    """

    # Init transformed images
    image_transformed = np.zeros([maxy, maxx, 3], dtype='uint8')
    lime_transformed = np.zeros([maxy, maxx, 3], dtype='uint8')
    # Get points from the RGB image and from the transformed RGB image
    points = get_points(image_cropped, predictor, image_cropped.shape[1], image_cropped.shape[0], 0)
    points_transformed = get_points(image_transformed, predictor, maxx, maxy, vertical_space)
    
    # Get triangles
    triangles = Delaunay(points)
    triangle_list = triangles.simplices.copy()
    np.insert(triangle_list, 1, triangles.simplices[0].copy())

    # Transform RGB and LIME mask images
    copy_images(maxx, maxy, image_cropped, image_transformed, image_lime, lime_transformed, points_transformed, points, triangle_list)
                 
    # Dibujar puntos y triangulos sobre cada imagen
    draw_points_and_triangles(image_cropped, points, triangles)
    draw_points_and_triangles(image_lime, points, triangles)
    # draw_points_and_triangles(image_transformed, points_transformed, triangles)
    # draw_points_and_triangles(lime_transformed, points_transformed, triangles)

    return image_cropped, image_lime, image_transformed, lime_transformed


def get_points(img, predictor, width_image, height_image, vertical_space):
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Encuadramos donde está la cara (en este caso como ya és una imagen de la cara cogemos el tamaño de la imagen)
    # Si fuese una imagen que incluyera un paisaje con una persona, en este punto se debería aplicar un detector de caras
    # y sacar las coordenadas de donde está la cara

    dlib_rect = dlib.rectangle(0, 0, width_image, height_image - vertical_space)
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(x, y, x+w, y+h)


    # Detecta los landmarks de la imagen
    detected_landmarks = predictor(img, dlib_rect)
    detected_landmarks = face_utils.shape_to_np(detected_landmarks)

    # añade en la partes superior de la imagen los 17 puntos de la barbilla
    for i in range(0, 17):
        newdl = detected_landmarks[i].copy()
        newdl[1] = 0
        detected_landmarks = np.append(detected_landmarks, [newdl], axis=0)

    # añade las esquinas
    detected_landmarks = np.append(detected_landmarks, [[0, 0]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[0, height_image]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[width_image, 0]], axis=0)
    detected_landmarks = np.append(detected_landmarks, [[width_image, height_image]], axis=0)

    points = detected_landmarks

    return points


def draw_points_and_triangles(img, points, triangles):


    # dibujamos los triangulos en la imagen
    for triangulo in triangles.simplices:
        cv2.line(img, (points[triangulo[0], 0], points[triangulo[0], 1]),
                 (points[triangulo[1], 0], points[triangulo[1], 1]), (255, 255, 0), 1)
        cv2.line(img, (points[triangulo[1], 0], points[triangulo[1], 1]),
                 (points[triangulo[2], 0], points[triangulo[2], 1]), (255, 255, 0), 1)
        cv2.line(img, (points[triangulo[2], 0], points[triangulo[2], 1]),
                 (points[triangulo[0], 0], points[triangulo[0], 1]), (255, 255, 0), 1)
    
    # dibujamos los puntos en la imagen
    for (x, y) in points:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)


# deterimna en que lado de la recta p2-p3, está el punto p1
# aunque devuelve un número viene determinado por el signo
def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


# determina si el punto pt esta dentro del triángulo formado por los vértices v1, v2 y v3
def point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    # esta dentro si los tres son negativos o los tres son positivos
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def copy_images(maxx, maxy, image, image_result, mask, mask_result, points_mask, points, triangle_list):
    """
    Funcion que transforma una imagen para normalizar
    :param maxx: anchura maxima a procesar en la imagen resultante
    :param maxy: altura maxima a procesar en la imagen resultante
    :param image: imagen origen
    :param image_result: imagen resultado
    :param points_mask: puntos de la mascara normalizaada
    :param points: puntos de imagen a normalizar
    :param triangle_list: lista de triangulos que se usan para transfomar, estan definidos segun points y points_mask
    :return: no devuelve nada, en image_result está la imagen normalizada
    """
    width_image = image.shape[0]
    height_image = image.shape[1]
    for x in range(0, maxx):
        for y in range(0, maxy):
            pt = np.array([x, y])
            for tri in triangle_list:
                v1 = points_mask[tri[0], :]
                v2 = points_mask[tri[1], :]
                v3 = points_mask[tri[2], :]
                if point_in_triangle(pt, v1, v2, v3):
                    v1v2 = v2 - v1
                    v1v3 = v3 - v1

                    N = np.cross(v1v2, v1v3)
                    area = np.linalg.norm(N) / 2
                    if area == 0:
                        continue

                    edge1 = v3 - v2
                    vp1 = pt - v2
                    C = np.cross(edge1, vp1)
                    u = (np.linalg.norm(C) / 2) / area

                    edge2 = v1 - v3
                    vp3 = pt - v3
                    C = np.cross(edge2, vp3)
                    v = (np.linalg.norm(C) / 2) / area

                    w = 1 - u - v

                    v1o = points[tri[0], :]
                    v2o = points[tri[1], :]
                    v3o = points[tri[2], :]
                    pto = u * v1o + v * v2o + w * v3o

                    try:
                        ptox = int(pto[0])
                        ptoy = int(pto[1])
                        if ptox >= 0 and ptox < width_image and ptoy >= 0 and ptoy < height_image:
                            image_result[y, x] = image[ptoy, ptox]
                            mask_result[y, x] = mask[ptoy, ptox]
                    except:
                        print(pt)
                        print(v1)
                        print(v2)
                        print(v3)
                        print(u)
                        print(v)
                        print(w)
                        print(pto)
                    break
