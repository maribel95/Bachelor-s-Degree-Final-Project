from PIL import Image
import os
import cv2
import numpy as np
import sys
from functools import partial
import torch
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.color import label2rgb

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from lime2.lime_image import LimeImageExplainer
from utils import *


# Function to normalize tensor. The norm has to be 1.
def normalize_tensor(tensor):
    norm = np.linalg.norm(tensor)       # Calculate euclidean norm
    normalized_vector = tensor / norm   # Divide each element by the norm
    return normalized_vector

# Returns feature vector from an image
def vector_from_img(img, session, input_name, output_name):
    img = np.expand_dims(img, axis=0)                       # Need 1 dimension more for the batch
    img = np.transpose(img, (0, 3, 1, 2))                   # Added a dimension to match the expected model input
    model_output = session.run([output_name], {input_name: img})           # Get model result
    features_tensor_output = torch.from_numpy(model_output[0])             # Get vector features
    features_normalized_tensor = normalize_tensor(features_tensor_output)  # Normalize vector

    return features_normalized_tensor
# Distance between two feature vectors
def vector_distance(v_x, v_y):
    scalar_product = np.dot(v_x.flatten(), v_y.flatten())   # Cosine similitude
    cosine_distance = (scalar_product + 1) / 2              # Cosine distance
    return cosine_distance

def vectors_from_imgs(imgs, session, input_name, output_name):
    vectors = []
    for img in imgs:
        vectors.append(vector_from_img(img, session, input_name, output_name))
    return np.array(vectors)

def vector_distances(vectors_x, vector_y):
    distances = []
    for v in vectors_x:
        distances.append(vector_distance(v, vector_y))
    return np.reshape(np.array(distances), (10, 1))     # Need a list of single elements lists

def classifier_fn(imgs, vector_y, session, input_name, output_name):
    vectors = vectors_from_imgs(imgs, session, input_name, output_name)
    distances = vector_distances(vectors, vector_y)
    return distances

def classifier_fn_2(imgs, vector_y):
    distances = []
    for img in imgs:
        vector_x = vector_from_img(img)
        distance = vector_distance(vector_x, vector_y)
        distances.append(distance)
    return np.array(distances)

def apply_lime(session, input_name, output_name, img_path, exp_img_path, exp_mask_path, exp_seg_path,
             segmentation_fn=None, hide_color=0, num_samples=1000, batch_size=10,
             th=None, top_k=None, min_accum=None, improve_background=False, pos_only=False, neg_only=False,
             hist_stretch=True, invert=True):



    # Load, resize, convert image...
    img = load_img(img_path)
    img = np.transpose(img, (1, 2, 0))

    # Segment
    if segmentation_fn is None:
        segments = slic(img, n_segments=30, compactness=20.0, start_label=0)
    elif segmentation_fn == 'quickshift':
        segments = quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
    elif segmentation_fn == 'slic':
        segments = slic(img, n_segments=30, compactness=20.0, start_label=0)
    else:
        segments = segmentation_fn(img)

    # Compute LIME explanation
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        img,
        classifier_fn=partial(classifier_fn, vector_y=vector_from_img(img, session, input_name, output_name), session=session, input_name=input_name, output_name=output_name),
        batch_size=batch_size,
        segments=segments,
        hide_color=hide_color,
        num_samples=num_samples,
        progress_bar=False)
    score_map = explanation.get_score_map(
        th,
        top_k,
        min_accum,
        improve_background,
        pos_only,
        neg_only)
    score_map_rgb = explanation.get_score_map_rgb(
        score_map,
        hist_stretch,
        invert)

    img = (img + 1) / 2
    # Save LIME explanation image and mask
    cv2.imwrite(os.path.join(exp_seg_path, os.path.basename(img_path)[:-4] + '_seg.png'), (label2rgb(segments) * 255).astype('uint8'))
    cv2.imwrite(os.path.join(exp_mask_path, os.path.basename(img_path)[:-4] + '_mask.png'), (score_map*255).astype('uint8'))
    cv2.imwrite(os.path.join(exp_img_path, os.path.basename(img_path)[:-4] + '_exp.png'), painter((img*255).astype('uint8'), score_map_rgb))
