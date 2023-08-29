import os
import sys
from tqdm import tqdm
import dlib

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from normalization import transform_mask


# ------------ Models to evaluate
model_names = ['ResNet18GLINT.onnx', 'ResNet18MS1MV3.onnx', 'ResNet34GLINT.onnx', 'ResNet34MS1MV3.onnx',
                'ResNet50GLINT.onnx', 'ResNet50MS1MV3.onnx', 'ResNet100GLINT.onnx', 'ResNet100MS1MV3.onnx'
              ]
dataset_path = r'Datos'

results_folder = 'Results'

# Path to the predictor file
predictor_path = 'resources/shape_predictor_68_face_landmarks.dat'


# ------------- ABSOLUTE PATH
absolute_path = os.path.dirname(os.path.abspath(__file__))
# ------------- FINAL PATHS
results_dir = os.path.join(absolute_path, results_folder)
test_dataset_path = os.path.join(absolute_path, dataset_path)


# Dimensions of normalized images
norm_width = 224
norm_height = 275
vertical_space = 25

# Init progress bar
progress = tqdm(total=len(model_names) * 3690)

# Load predictor
predictor = dlib.shape_predictor(os.path.join(absolute_path, predictor_path))

# Iterate over each net
for model_name in model_names:

    print(model_name)

    for view_folder in os.listdir(test_dataset_path):
        test_dataset_path_view = os.path.join(test_dataset_path, view_folder)
        # Results foder
        results_net = os.path.join(results_dir, model_name + '_results', view_folder)
        # Iterate over each class
        for class_dir in os.listdir(test_dataset_path_view):
            # Path of imgs folder
            img_path = os.path.join(test_dataset_path_view, class_dir)

            # Path of LIME masks
            exp_mask_path = os.path.join(results_net, class_dir, 'lime_masks')

            # Path of imgs with landmarks
            positives_landmarks_path = os.path.join(results_net, class_dir, 'positives_landmarks')
            if not os.path.exists(positives_landmarks_path):
                os.mkdir(positives_landmarks_path)

            # Path of LIME masks with landmarks
            lime_masks_landmarks_path = os.path.join(results_net, class_dir, 'lime_masks_landmarks')
            if not os.path.exists(lime_masks_landmarks_path):
                os.mkdir(lime_masks_landmarks_path)

            # Path of transformed imgs
            positives_transformed_path = os.path.join(results_net, class_dir, 'positives_transformed')
            if not os.path.exists(positives_transformed_path):
                os.mkdir(positives_transformed_path)

            # Path of transformed LIME masks
            lime_masks_transformed_path = os.path.join(results_net, class_dir, 'lime_masks_transformed')
            if not os.path.exists(lime_masks_transformed_path):
                os.mkdir(lime_masks_transformed_path)

            # Transform each img and mask
            for img_name, mask_name in zip(os.listdir(img_path), os.listdir(exp_mask_path)):
                # Load image and LIME mask
                img = load_img_explanations(os.path.join(img_path, img_name))

                mask = cv2.imread(os.path.join(exp_mask_path, mask_name))

                # Draw landmarks on img and LIME mask, and transform both
                img_land, lime_land, img_trans, lime_trans = transform_mask(
                    predictor=predictor,
                    maxx=norm_width,
                    maxy=norm_height,
                    vertical_space=vertical_space,
                    image_cropped=img,
                    image_lime=mask)


                # Save images
                cv2.imwrite(os.path.join(positives_landmarks_path, img_name[:-4] + '_landmarks.png'), img_land)
                cv2.imwrite(os.path.join(lime_masks_landmarks_path, img_name[:-4] + '_mask_landmarks.png'), lime_land)
                cv2.imwrite(os.path.join(positives_transformed_path, img_name[:-4] + '_transformed.png'), img_trans)
                cv2.imwrite(os.path.join(lime_masks_transformed_path, img_name[:-4] + '_mask_transformed.png'), lime_trans)

                # Update progress
                progress.update(1)

# Close progress
progress.close()