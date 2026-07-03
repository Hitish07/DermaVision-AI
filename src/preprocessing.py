import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from .config import DATASET, PATHS

def dullrazor(img):
    """
    Removes hair from the image using the DullRazor algorithm.
    Uses morphological blackhat operations and inpainting.
    """
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    return dst

def apply_clahe(img):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def preprocess_image_cv2(image):
    """Full preprocessing pipeline using OpenCV."""
    if image.dtype != np.uint8:
        # Scale back to 0-255 if it's float
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
    image = dullrazor(image)
    image = apply_clahe(image)
    image = cv2.resize(image, DATASET.image_size)
    return image

def generate_preprocessing_samples(image_paths, num_samples=3):
    """Generates and saves before/after images of preprocessing."""
    plt.figure(figsize=(10, num_samples * 3))
    for i in range(num_samples):
        img_path = image_paths[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        processed = preprocess_image_cv2(img)
        
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(processed)
        plt.title("Preprocessed (DullRazor + CLAHE)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(PATHS.figures / 'preprocessing_samples.png', dpi=300)
    plt.close()

def tf_preprocess_image(image_path, label):
    """Wrapper to use cv2 preprocessing within tf.data pipeline."""
    def _preprocess(path):
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = img.numpy()
        processed = preprocess_image_cv2(img)
        # Convert to float32 [0, 1] for model input
        processed = processed.astype(np.float32) / 255.0
        return processed

    [image] = tf.py_function(_preprocess, [image_path], [tf.float32])
    image.set_shape([DATASET.image_size[0], DATASET.image_size[1], 3])
    return image, label
