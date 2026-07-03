import numpy as np
import tensorflow as tf
import logging
import cv2
from .config import DATASET, PATHS
from .preprocessing import preprocess_image_cv2
from .augmentation import get_tta_transforms
from .explainability import explain_prediction

logger = logging.getLogger(__name__)

def predict_single_image(model, model_name, image_path, use_tta=False):
    """Predicts the class of a single image, with optional Test Time Augmentation."""
    logger.info(f"Predicting image {image_path} with {model_name}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Preprocess
    processed_img = preprocess_image_cv2(img)
    
    # Note: Our preprocess_image_cv2 returns uint8 image if we didn't normalize, 
    # but let's ensure it's float32 [0,1] for prediction as expected by the model.
    if processed_img.dtype == np.uint8:
        processed_img = processed_img.astype(np.float32) / 255.0
        
    input_tensor = np.expand_dims(processed_img, axis=0) # [1, H, W, 3]
    
    # 2. Predict (with or without TTA)
    if use_tta:
        logger.info("Applying Test Time Augmentation...")
        predictions = []
        # Original prediction
        predictions.append(model.predict(input_tensor, verbose=0)[0])
        
        # Augmented predictions
        tta_transforms = get_tta_transforms()
        for _ in range(4): # 4 TTA passes
            aug = tta_transforms(image=processed_img)['image']
            aug_tensor = np.expand_dims(aug, axis=0)
            predictions.append(model.predict(aug_tensor, verbose=0)[0])
            
        final_prediction = np.mean(predictions, axis=0)
    else:
        final_prediction = model.predict(input_tensor, verbose=0)[0]
        
    # 3. Format results
    top_3_idx = np.argsort(final_prediction)[-3:][::-1]
    results = [
        {"class": DATASET.class_names[idx], "confidence": float(final_prediction[idx])}
        for idx in top_3_idx
    ]
    
    # 4. Explainability (Grad-CAM)
    explain_prediction(model, model_name, str(image_path), tf.convert_to_tensor(input_tensor))
    
    return results

def predict_ensemble(models, image_path):
    """Predicts using an ensemble of models (probability averaging)."""
    logger.info(f"Predicting image {image_path} with Ensemble of {len(models)} models")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    processed_img = preprocess_image_cv2(img)
    if processed_img.dtype == np.uint8:
        processed_img = processed_img.astype(np.float32) / 255.0
        
    input_tensor = np.expand_dims(processed_img, axis=0)
    
    all_preds = []
    for model in models:
        all_preds.append(model.predict(input_tensor, verbose=0)[0])
        
    final_prediction = np.mean(all_preds, axis=0)
    
    top_3_idx = np.argsort(final_prediction)[-3:][::-1]
    results = [
        {"class": DATASET.class_names[idx], "confidence": float(final_prediction[idx])}
        for idx in top_3_idx
    ]
    return results
