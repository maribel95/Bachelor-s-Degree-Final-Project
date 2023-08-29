import os
import sys
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *

def save_heatmaps(img_accum, path_gray=None, path_colormap=None, colormap=cv2.COLORMAP_JET, min_th=None):
    heatmap = img_accum
    heatmap = histogram_stretching(heatmap, 0, 1)
    if min_th is not None:
        heatmap[heatmap < min_th] = 0
    heatmap = (heatmap * 255).astype('uint8')

    # Save in gray scale
    if path_gray is not None:
        cv2.imwrite(path_gray, heatmap)

    # Save in a COLORMAP
    if path_colormap is not None:
        heatmap = cv2.applyColorMap(heatmap, colormap=colormap)
        cv2.imwrite(path_colormap, heatmap)

def apply_threshold(heatmap, th='otsu'):
    if th == 'otsu':
        _, th_heatmap = cv2.threshold(heatmap, 0, 255, cv2.THRESH_OTSU)
    elif type(th) == int:
        _, th_heatmap = cv2.threshold(heatmap, th, 255, cv2.THRESH_BINARY)
    elif type(th) == float:
        _, th_heatmap = cv2.threshold(heatmap, int(th*255), 255, cv2.THRESH_BINARY)
    else:
        raise Exception('Bad value for th parameter.')
    return th_heatmap

def distance_gray(image1, image2, method=cv2.TM_CCOEFF_NORMED):
    tm_res = cv2.matchTemplate(image1, image2, method)[0][0]
    return tm_res if method == cv2.TM_SQDIFF_NORMED else 1 - tm_res

def iou(image1, image2):
    return np.sum(np.logical_and(image1, image2)) / np.sum(np.logical_or(image1, image2))

def get_best_threshold(gt_image, heatmap, distance_metric='f1_score', range_white=[0.5, 1.5], path_save=None):

    # Lists of thresholded images and their f1_score
    ths = []
    scores = []

    # Ground truth and heatmap flattened
    gt = (gt_image/255).flatten()
    hm = heatmap

    total_white = np.sum(gt)
    min_white = total_white * range_white[0]
    max_white = total_white * range_white[1]

    # For each gray value
    for i in range(256):

        # Apply current threshold
        th = np.zeros_like(hm)
        th[hm > i] = 1

        sum_th = np.sum(th)

        if sum_th <= max_white:
            ths.append(th)

            # Compute distance with ground truth
            if distance_metric == 'f1_score':
                dist = f1_score(gt, th.flatten())
            elif distance_metric == 'iou':
                dist = iou(gt, th.flatten())
            elif distance_metric == 'white':
                dist = np.abs(sum_th - total_white)
            scores.append(dist)

            if path_save is not None:
                cv2.imwrite(os.path.join(path_save, str(i) + '_' + str(dist) + '.png'), th * 255)

            if sum_th < min_white:
                break

    # Return thresholded image with maximum f1_score
    if distance_metric == 'white':
        return ths[np.argmin(scores)]*255
    else:
        return ths[np.argmax(scores)]*255

def plot_dendogram(labels, heatmaps, title, save_path, linkage_method='average', width=10, height=15, title_size=20, font_size=16):

    # Compute distance between one heatmap and the remaining, for all heatmaps
    # Distance matrix will have 0s on the diagonal and wil be symetric
    distances = [] 
    for i in range(len(heatmaps)-1):
        for j in range(i+1, len(heatmaps)):
            distances.append(distance_method(heatmaps[i], heatmaps[j]))
    
    # Perform hierarchical/agglomerative clustering
    links = linkage(distances, linkage_method)

    # Plot dendogram
    plt.rc('font', size=font_size)
    plt.figure(figsize=(width, height))
    plt.title(title, fontdict={'fontsize':title_size})
    dendrogram(links, labels=labels, orientation='right', leaf_font_size=font_size)
    plt.savefig(save_path)
    plt.clf()

    return

def plot_bar(labels, distances, title, save_path, sort=True, width=10, height=15, title_size=20, font_size=16, reverse=True):
    
    # Sort if specified
    if sort:
        sorted_list = sorted(list(zip(labels, distances)), key=lambda t: t[1], reverse=reverse)
        labs, dists = [t[0] for t in sorted_list], [round(t[1], 2) for t in sorted_list]
    else:
        labs, dists = labels, distances

    # Plot
    plt.rc('font', size=font_size)
    plt.figure(figsize=(width, height))
    plt.title(title, fontdict={'fontsize':title_size})
    bars = plt.barh(width=dists, y=labs, color=(0.2, 0.4, 0.6, 1.0))
    plt.bar_label(bars)
    plt.savefig(save_path)
    plt.clf()

def heatmap_score(gt_bw, heatmap):
    """
    heatmap and gt_bw in range [0, 1]
    """
    return np.sum(heatmap*gt_bw)/np.sum(heatmap)

def get_masks_distance(gt, th_heatmap, method='iou'):
    if method == 'iou':
        return iou(gt, th_heatmap)
    elif method == 'f1_score':
        return f1_score(gt.flatten(), th_heatmap.flatten())