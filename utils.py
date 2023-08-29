import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from scipy.special import kl_div

# Load image from memory
def load_img(img_path):
    image = Image.open(img_path).convert('RGB')
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
    ])
    img = preprocess(image)
    img = (2 * img) - 1 # Normalization between [-1, 1] because that's the format for the neural network
    return img.numpy()

def load_img_explanations(img_path):

    image = Image.open(img_path).convert('RGB')

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
    ])
    img = preprocess(image).numpy()

    imagen_normalizada = img / np.max(img)

    # Convertir la imagen normalizada a uint8
    img = (imagen_normalizada * 255)
    img = np.array(img)
    img = np.transpose(img, (1, 2, 0))
    img = np.ascontiguousarray(img, dtype=np.uint8)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # En caso de querer ponerlo en un único canal
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def distance_method(P, Q):

    P = P.flatten()
    P = (P - P.min()) / (P.max() - P.min())

    Q = Q.flatten()
    Q = (Q - Q.min()) / (Q.max() - Q.min())
    # Calcular la imagen media R como la media de P y Q
    R = (P + Q) / 2.0
    parametros = R.shape[0]
    # Calcular la divergencia KL entre P y R
    kl_divergence_PR = sum(kl_div(P, R)) / parametros
    # Calcular la divergencia KL entre Q y R
    kl_divergence_QR = sum(kl_div(Q, R)) / parametros
    # Sumar las divergencias KL
    total_kl_divergence = kl_divergence_PR + kl_divergence_QR
    return total_kl_divergence


def histogram_stretching(img, h_min=0, h_max=1):
    max_value = np.max(img)
    min_value = np.min(img)
    if max_value > 0 and min_value != max_value:
        return h_min+(h_max-h_min)*(img-min_value)/(max_value-min_value)
    else:
        return img


def painter(img1, img2, alpha2=0.5):
    """ Merges two videos into one, according to alpha factor.

    Args:
        vid1: 1st video
        vid2: 2nd video
        alpha2: Importance of 2nd video. 1 maximum, 0 minimum.

    Returns:
        Merged images.
    """
    return (img1.astype('float') * (1 - alpha2)
            + img2.astype('float') * alpha2).astype('uint8')
    
