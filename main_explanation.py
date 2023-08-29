from tqdm import tqdm
import onnxruntime
from explanations import *


# ------------ Models to evaluate
model_names = ['ResNet18GLINT.onnx', 'ResNet18MS1MV3.onnx', 'ResNet34GLINT.onnx', 'ResNet34MS1MV3.onnx',
                 'ResNet50GLINT.onnx', 'ResNet50MS1MV3.onnx', 'ResNet100GLINT.onnx', 'ResNet100MS1MV3.onnx'
              ]

# ------------- FOLDERS
dataset_path = r'Datos'
results_folder = 'Results'
# ------------- ABSOLUTE PATH
absolute_path = os.path.dirname(os.path.abspath(__file__))
# ------------- FINAL PATHS
results_dir = os.path.join(absolute_path, results_folder)
test_dataset_path = os.path.join(absolute_path, dataset_path)

# Results folder
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Init progress bar
progress = tqdm(total=len(model_names) * 368 * 10)

for model_name in model_names:
    # We get the folder of the model we want to use
    model_to_explain_path = os.path.join(absolute_path, "Models", model_name)
    # Results folder(we will save all the results here)
    model_results_path = os.path.join(results_dir, model_name + '_results')

    session = onnxruntime.InferenceSession(model_to_explain_path)   # Create a model session
    input_name = session.get_inputs()[0].name                       # Model input
    output_name = session.get_outputs()[0].name                     # Model output

    if not os.path.exists(model_results_path):
        os.mkdir(model_results_path)

    for view_folder in os.listdir(test_dataset_path):
        test_dataset_path_view = os.path.join(test_dataset_path, view_folder)
        if not os.path.exists(os.path.join(model_results_path, view_folder)):
            os.mkdir(os.path.join(model_results_path, view_folder))
        # Explain each positive with LIME
        for  class_dir in os.listdir(test_dataset_path_view):

            class_dir_view = os.path.join(model_results_path, view_folder, class_dir)
            # Path of folder to explain
            path_dir = os.path.join(test_dataset_path_view, class_dir)
            # Create folder for each user
            if not os.path.exists(os.path.join(model_results_path, class_dir_view)):
                os.mkdir(os.path.join(model_results_path, class_dir_view))

            # Path where to store masks
            exp_img_path = os.path.join(model_results_path, class_dir_view, 'lime_explanations')
            if not os.path.exists(exp_img_path):
                os.mkdir(exp_img_path)

            # Path where to store LIME explanations
            exp_mask_path = os.path.join(model_results_path, class_dir_view, 'lime_masks')
            if not os.path.exists(exp_mask_path):
                os.mkdir(exp_mask_path)

            # Path where to store segmentations
            exp_seg_path = os.path.join(model_results_path, class_dir_view, 'lime_seg')
            if not os.path.exists(exp_seg_path):
                os.mkdir(exp_seg_path)

            # Iterate over each image
            for img_name in os.listdir(path_dir):
                # Run LIME explanation
                apply_lime(session, input_name, output_name, os.path.join(path_dir, img_name), exp_img_path,
                           exp_mask_path, exp_seg_path,
                           hide_color=0, num_samples=1000, batch_size=10,
                           th=None, top_k=None, min_accum=None, improve_background=False, pos_only=True, neg_only=False,
                           hist_stretch=True, invert=True)
                # Update progress
                print(img_name, class_dir, view_folder, model_name)
                progress.update(1)

# Close progress
progress.close()