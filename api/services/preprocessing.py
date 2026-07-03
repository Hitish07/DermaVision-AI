import cv2
import numpy as np
import tensorflow as tf

def dullrazor(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Black hat filter to find hairs
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Gaussian blur to enhance hair contours
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    
    # Thresholding
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint
    result = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)
    return result

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Convert to LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    cl = clahe.apply(l)
    
    # Merge back
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def preprocess_for_inference(image_path: str, target_size=(224, 224)):
    """
    Exact preprocessing pipeline from training.
    Hair Removal -> CLAHE -> Resize -> MobileNet preprocess_input
    """
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # 1. Hair Removal
    img_no_hair = dullrazor(img)
    
    # 2. CLAHE
    img_clahe = apply_clahe(img_no_hair)
    
    # Convert to RGB for TensorFlow
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    
    # 3. Resize
    img_resized = cv2.resize(img_rgb, target_size)
    
    # 4. TensorFlow Preprocess
    img_tensor = np.expand_dims(img_resized, axis=0)
    img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(img_tensor.astype(np.float32))
    
    return img_tensor, img_rgb
