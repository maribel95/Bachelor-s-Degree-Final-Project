
from tqdm import tqdm

from heatmaps import *
from utils import *
from individuals_classification import AFRICANOS, CAUCASICOS, ASIATICOS, HOMBRES, MUJERES

# ------------ Models to evaluate
model_names = ['ResNet18GLINT.onnx', 'ResNet18MS1MV3.onnx', 'ResNet34GLINT.onnx', 'ResNet34MS1MV3.onnx',
                 'ResNet50GLINT.onnx', 'ResNet50MS1MV3.onnx', 'ResNet100GLINT.onnx', 'ResNet100MS1MV3.onnx'
              ]

generos = ['HOMBRES', 'MUJERES']
lista_generos = [HOMBRES, MUJERES]
# Posibles bases de datos con los que se han entrenado las redes
db = ['GLINT', 'MS1MV3']
# Posibles bases de datos con los que se han entrenado las redes
razas = ['AFRICANOS', 'CAUCASICOS','ASIATICOS']
lista_razas = [AFRICANOS, CAUCASICOS, ASIATICOS]
# ------------- FOLDERS
results_folder = 'Results'
heatmaps_folder = 'Heatmaps'
heatmaps_colormap = os.path.join(heatmaps_folder, 'COLORMAP_JET')
heatmaps_gray = os.path.join(heatmaps_folder, 'GRAY')

# ------------- ABSOLUTE PATH
absolute_path = os.path.dirname(os.path.abspath(__file__))
# ------------- FINAL PATHS
results_dir = os.path.join(absolute_path, results_folder)

# Folders where the heatmaps are saved
heatmaps_dir_colormap =  os.path.join(absolute_path, heatmaps_colormap)
heatmaps_dir_gray =  os.path.join(absolute_path, heatmaps_gray)

if not os.path.exists(os.path.join(absolute_path, "Heatmaps")):
    os.mkdir(os.path.join(absolute_path, "Heatmaps"))

if not os.path.exists(os.path.join(absolute_path, heatmaps_colormap)):
    os.mkdir(os.path.join(absolute_path, heatmaps_colormap))

if not os.path.exists(os.path.join(absolute_path, heatmaps_gray)):
    os.mkdir(os.path.join(absolute_path, heatmaps_gray))

# Dimensions of normalized images
norm_width = 224
norm_height = 275
vertical_space = 25

explain_method = 'lime'

# Init progress bar
progress = tqdm(total=50)
heatmaps_gray_person_and_net = '/Users/maribelcrespivalero/Desktop/TFG/explain_framework/Heatmaps/GRAY/heatmaps_by_person_and_net'

# Create heatmaps folders if they don't exist
for f1 in [heatmaps_dir_colormap, heatmaps_dir_gray]:
    for f2 in ['heatmaps_by_person_and_net', 'heatmaps_by_person', 'heatmaps_by_net', 'heatmaps_by_db', 'heatmaps_by_type_of_ResNet']:
        if not os.path.exists(os.path.join(f1, f2)):
            os.mkdir(os.path.join(f1, f2))


num_users = os.path.join(results_dir, 'ResNet18GLINT.onnx' + '_results', 'Frontal')
# Init total accumulator
label_names = os.listdir(num_users)
accum_tot_pers_img = np.zeros((norm_height, norm_width, len(os.listdir(num_users))), dtype=float)
accum_tot_pers_counter = np.zeros((len(os.listdir(num_users))), dtype=int)
n_models = int(len(model_names) / 2)
# Init ResNet accumulator
accum_resnet_img = np.zeros((norm_height, norm_width, n_models), dtype=float)
accum_resnet_counter = np.zeros(n_models, dtype=int)

# Init DB accumulator
accum_db_img = np.zeros((norm_height, norm_width, len(db)), dtype=float)
accum_db_counter = np.zeros((len(db)), dtype=int)
# Iterador que va según resnet. Como hay de dos bases de datos, se actualizará cada dos ciclos
resnet_i = -1
# Iterate over each net
for model_i, model_name in enumerate(model_names):
    results_net = os.path.join(results_dir, model_name + '_results')
    test_dataset_path_view = os.path.join(results_net, "Frontal")
    # Init model accumulator
    accum_model_img = np.zeros((norm_height, norm_width, len(model_names)), dtype=float)
    accum_model_counter = np.zeros((len(model_names)), dtype=int)
    iterator = model_i % 2
    # Init ResNet accumulator
    if iterator == 0:
        resnet_i += 1
        accum_resnet_img = np.zeros((norm_height, norm_width, n_models), dtype=float)
        accum_resnet_counter = np.zeros((n_models), dtype=int)

    # ======================== Carpetas de cada modelo para heatmaps_by_person_and_net
    heatmaps_pers_net_gray_path = os.path.join(heatmaps_dir_gray, 'heatmaps_by_person_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_gray_path):
        os.mkdir(heatmaps_pers_net_gray_path)

    heatmaps_pers_net_color_path = os.path.join(heatmaps_dir_colormap, 'heatmaps_by_person_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_color_path):
        os.mkdir(heatmaps_pers_net_color_path)
    # Iterate over each class
    for class_i, class_dir in enumerate(entry for entry in os.listdir(test_dataset_path_view)):
        # Init persons accumulator
        accum_pers_img = np.zeros((norm_height, norm_width, len(os.listdir(test_dataset_path_view))), dtype=float)
        accum_pers_counter = np.zeros((len(os.listdir(test_dataset_path_view))), dtype=int)
        # Path of imgs folder
        img_path = os.path.join(test_dataset_path_view, class_dir)

        # Path of transformed LIME masks
        lime_masks_transformed_path = os.path.join(test_dataset_path_view, class_dir, explain_method + '_masks_transformed')
        print("lime_masks_transformed_path: ",lime_masks_transformed_path)
        # Accumulate each mask
        for mask_name in os.listdir(lime_masks_transformed_path):
            # Load, to gray and to [0, 1] range
            mask = cv2.imread(os.path.join(lime_masks_transformed_path, mask_name))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask / 255
            # Accumulate mask
            accum_pers_img[:, :, class_i] += mask
            accum_pers_counter[class_i] += 1

            # Update progress
            progress.update(1)
        # ----- HEATMAP POR PERSONA Y POR RED
        heatmap_name = model_name + '_' + label_names[class_i] + '_heatmap.png'
        save_heatmaps(
            img_accum=accum_pers_img[:, :, class_i] / accum_pers_counter[class_i],
            path_gray=os.path.join(heatmaps_pers_net_gray_path, heatmap_name),
            path_colormap=os.path.join(heatmaps_pers_net_color_path, heatmap_name),
            colormap=cv2.COLORMAP_JET,
            min_th=0
        )
        # Accumulate imgs and counters from a model set
        accum_model_img[:, :, model_i] += accum_pers_img[:, :, class_i]
        accum_model_counter[model_i] += np.sum(accum_pers_counter)
        # Accumulate imgs and counters from a ResNet set
        accum_resnet_img[:, :, resnet_i] += accum_pers_img[:, :, class_i]
        accum_resnet_counter[resnet_i] += np.sum(accum_pers_counter)
        # Accumulate imgs and counters from the total set
        accum_tot_pers_img += accum_pers_img
        accum_tot_pers_counter[class_i] += np.sum(accum_pers_counter)
        # Accumulate imgs and counters from a BD set
        accum_db_img[:, :, iterator] += accum_pers_img[:, :, class_i]
        accum_db_counter[iterator] += np.sum(accum_pers_counter)


    # ----- HEATMAP POR RED
    heatmap_name = model_name + '_heatmap.png'
    save_heatmaps(
        img_accum=accum_model_img[:, :, model_i] / accum_model_counter[model_i],
        path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_net', heatmap_name),
        path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_net', heatmap_name),
        colormap=cv2.COLORMAP_JET,
        min_th=0
    )

    if iterator == 0:
        # ----- HEATMAP POR RESNET
        heatmap_name = model_name + '_heatmap.png'
        save_heatmaps(
            img_accum=accum_resnet_img[:, :, resnet_i] / accum_resnet_counter[resnet_i],
            path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_type_of_ResNet', heatmap_name),
            path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_type_of_ResNet', heatmap_name),
            colormap=cv2.COLORMAP_JET,
            min_th=0
        )




# ----- HEATMAP TOTAL DE PERSONAS
# Save heatmap_model: with hist_stretch and color_map COLORMAP_JET
for class_i in range(len(label_names)):
    heatmap_name = label_names[class_i] + '_heatmap.png'
    save_heatmaps(
        img_accum=accum_tot_pers_img[:, :, class_i] / accum_tot_pers_counter[class_i],
        path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_person', heatmap_name),
        path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_person', heatmap_name),
        colormap=cv2.COLORMAP_JET,
        min_th=0
    )

# ----- HEATMAP TOTAL SEGUN LA BASE DE DATOS
# Save heatmap_model: with hist_stretch and color_map COLORMAP_JET
for class_i in range(len(db)):
    heatmap_name = db[class_i] + '_heatmap.png'
    save_heatmaps(
        img_accum=accum_db_img[:, :, class_i] / accum_db_counter[class_i],
        path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_db', heatmap_name),
        path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_db', heatmap_name),
        colormap=cv2.COLORMAP_JET,
        min_th=0
    )


# =====================================      HEATMAPS POR AGRUPAMIENTO       ===========================================
# Init DB accumulator
accum_total_supremo_raza_img = np.zeros((norm_height, norm_width, len(razas)), dtype=float)
accum_total_supremo_raza_counter = np.zeros((len(razas)), dtype=int)

# Init DB accumulator
accum_total_supremo_sexo_img = np.zeros((norm_height, norm_width, len(generos)), dtype=float)
accum_total_supremo_sexo_counter = np.zeros((len(generos)), dtype=int)
for class_i, model_name in enumerate(os.listdir(heatmaps_gray_person_and_net)):
    # Init race accumulator
    accum_raza_img = np.zeros((norm_height, norm_width, len(razas)), dtype=float)
    accum_raza_counter = np.zeros((len(razas)), dtype=int)
    # Init sex accumulator
    accum_genero_img = np.zeros((norm_height, norm_width, len(generos)), dtype=float)
    accum_genero_counter = np.zeros((len(generos)), dtype=int)
    # ============    AHORA HAREMOS LOS HEATMAPS POR CLASIFICACIÓN DE RAZAS     ================================

    # ======================== Carpetas de cada modelo para heatmaps_by_person_and_net
    heatmaps_pers_net_gray_path = os.path.join(heatmaps_dir_gray, 'heatmaps_by_race_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_gray_path):
        os.mkdir(heatmaps_pers_net_gray_path)

    heatmaps_pers_net_color_path = os.path.join(heatmaps_dir_colormap, 'heatmaps_by_race_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_color_path):
        os.mkdir(heatmaps_pers_net_color_path)
    for class_rasa, rasa_name in enumerate(razas):
        model_path = os.path.join(heatmaps_gray_person_and_net, model_name)
        for person_name in lista_razas[class_rasa]:
            # Load heatmap and compute distance
            h_name = model_name + '_' + person_name + '_heatmap.png'
            # Load, to gray and to [0, 1] range
            print(h_name)
            mask = cv2.imread(os.path.join(model_path, h_name), cv2.IMREAD_GRAYSCALE)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask / 255
            # Accumulate mask
            accum_raza_img[:, :, class_rasa] += mask
            accum_raza_counter[class_rasa] += 1
        h_name = rasa_name + '_heatmap.png'
        save_heatmaps(
            img_accum=accum_raza_img[:, :, class_rasa] / accum_raza_counter[class_rasa],
            path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_race_and_net', model_name, h_name),
            path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_race_and_net', model_name, h_name),
            colormap=cv2.COLORMAP_JET,
            min_th=0
        )
    accum_total_supremo_raza_img[:, :, class_rasa] += accum_raza_img[:, :, class_rasa]
    accum_total_supremo_raza_counter[class_rasa] += np.sum(accum_raza_counter[class_rasa])
    # ============    AHORA HAREMOS LOS HEATMAPS POR CLASIFICACIÓN DE SEXOS     ================================
    # ======================== Carpetas de cada modelo para heatmaps_by_person_and_net
    heatmaps_pers_net_gray_path = os.path.join(heatmaps_dir_gray, 'heatmaps_by_sex_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_gray_path):
        os.mkdir(heatmaps_pers_net_gray_path)

    heatmaps_pers_net_color_path = os.path.join(heatmaps_dir_colormap, 'heatmaps_by_sex_and_net', model_name)
    if not os.path.exists(heatmaps_pers_net_color_path):
        os.mkdir(heatmaps_pers_net_color_path)
    for class_setso, setso_name in enumerate(generos):
        model_path = os.path.join(heatmaps_gray_person_and_net, model_name)
        for person_name in lista_generos[class_setso]:
            # Load heatmap and compute distance
            h_name = model_name + '_' + person_name + '_heatmap.png'
            # Load, to gray and to [0, 1] range
            print(h_name)
            mask = cv2.imread(os.path.join(model_path, h_name), cv2.IMREAD_GRAYSCALE)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask / 255
            # Accumulate mask
            accum_genero_img[:, :, class_setso] += mask
            accum_genero_counter[class_setso] += 1
        h_name = setso_name + '_heatmap.png'
        save_heatmaps(
            img_accum=accum_genero_img[:, :, class_setso] / accum_genero_counter[class_setso],
            path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_sex_and_net', model_name, h_name),
            path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_sex_and_net', model_name, h_name),
            colormap=cv2.COLORMAP_JET,
            min_th=0
        )
        accum_total_supremo_sexo_img[:, :, class_setso] += accum_genero_img[:, :, class_setso]
        accum_total_supremo_sexo_counter[class_setso] += np.sum(accum_genero_counter[class_setso])


# ----- HEATMAP TOTAL DE GENEROS
# Save heatmap_model: with hist_stretch and color_map COLORMAP_JET
for class_i in range(len(generos)):
    heatmap_name = label_names[class_i] + '_heatmap.png'
    save_heatmaps(
        img_accum=accum_total_supremo_sexo_img[:, :, class_i] / accum_total_supremo_sexo_counter[class_i],
        path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_sex', heatmap_name),
        path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_sex', heatmap_name),
        colormap=cv2.COLORMAP_JET,
        min_th=0
    )

# ----- HEATMAP TOTAL DE RAZAS
# Save heatmap_model: with hist_stretch and color_map COLORMAP_JET
for class_i in range(len(razas)):
    heatmap_name = label_names[class_i] + '_heatmap.png'
    save_heatmaps(
        img_accum=accum_total_supremo_raza_img[:, :, class_i] / accum_total_supremo_raza_counter[class_i],
        path_gray=os.path.join(heatmaps_dir_gray, 'heatmaps_by_race', heatmap_name),
        path_colormap=os.path.join(heatmaps_dir_colormap, 'heatmaps_by_race', heatmap_name),
        colormap=cv2.COLORMAP_JET,
        min_th=0
    )


# Close progress
progress.close()
