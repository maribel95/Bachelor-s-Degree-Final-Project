import os
import sys
from tqdm import tqdm
import dlib
from PIL import Image
from sklearn.metrics import f1_score

from nets import *
from normalization import *

from heatmaps import *
from utils import *



# Folders where the heatmaps are saved
heatmaps_dir_gray = '/Users/maribelcrespivalero/Desktop/TFG/explain_framework/Experimentos'

# Folders where to store dendograms and barplots
dendograms_dir = '/Users/maribelcrespivalero/Desktop/TFG/explain_framework/Dendograms'

# databases
db_names = ['GLINT', 'MS1MV3', 'TOTAL']

# ------------ Models to evaluate
model_names = ['ResNet18GLINT' ,'ResNet18MS1MV3', 'ResNet34GLINT', 'ResNet34MS1MV3',
             'ResNet50GLINT', 'ResNet50MS1MV3', 'ResNet100GLINT',
             'ResNet100MS1MV3','TOTAL'
            ]

# Subdirectories of heatmaps
heatmaps_bd_path = os.path.join(heatmaps_dir_gray, 'heatmaps_by_db')
heatmaps_net_path = os.path.join(heatmaps_dir_gray, 'heatmaps_by_net')



# ==================================     HEATMAPS POR DATABASE     ====================================
db_names_list = []
db_heats_list = []
for class_i, model_name in enumerate(db_names):
    # Total
    # Load heatmap and compute distance
    h_name = model_name + '_heatmap.png'
    print(h_name)
    heatmap = cv2.imread(os.path.join(heatmaps_bd_path, h_name), 0)

    # Store name and distance
    db_names_list.append(model_name)
    db_heats_list.append(heatmap)

# By expression
plot_dendogram(
    labels=db_names_list,
    heatmaps=db_heats_list,
    title='DISTANCIAS ENTRE BASES DE DATOS',
    save_path=os.path.join(dendograms_dir, 'bd_dendograma'),
    linkage_method='ward',
    width=16,
    height=5,
    title_size=16,
    font_size=9
)

# ==================================     HEATMAPS POR REDES     ====================================
net_names_list = []
net_heats_list = []
for class_i, model_name in enumerate(model_names):
    # Total
    # Load heatmap and compute distance
    h_name = model_name + '_heatmap.png'
    print(h_name)
    heatmap = cv2.imread(os.path.join(heatmaps_net_path, h_name), 0)

    # Store name and distance
    net_names_list.append(model_name)
    net_heats_list.append(heatmap)

# By expression
plot_dendogram(
    labels=net_names_list,
    heatmaps=net_heats_list,
    title='DISTANCIAS ENTRE TODOS LOS MODELOS',
    save_path=os.path.join(dendograms_dir, 'modelos_dendograma'),
    linkage_method='ward',
    width=16,
    height=5,
    title_size=16,
    font_size=9
)