import os
import time
import logging
import tensorflow as tf
from src.predict import predict_single_image
from src.explainability import get_integrated_gradients, generate_shap, save_and_display_overlay
from src.preprocessing import preprocess_image_cv2
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cache model globally in FastAPI
_model = None
MODEL_NAME = "MobileNetV2"

def get_model():
    global _model, MODEL_NAME
    if _model is None:
        from src.config import PATHS
        # Auto-detect available models
        supported_architectures = ["Ensemble", "VisionTransformer", "ConvNeXt", "EfficientNet", "DenseNet", "ResNet", "MobileNetV2"]
        best_model_path = None
        
        for arch in supported_architectures:
            p = PATHS.models / f"{arch}_final.keras"
            if p.exists():
                best_model_path = p
                MODEL_NAME = arch
                break
            # Fallback for .h5
            p_h5 = PATHS.models / f"{arch}_final.h5"
            if p_h5.exists():
                best_model_path = p_h5
                MODEL_NAME = arch
                break
                
        if not best_model_path:
            raise FileNotFoundError(f"No supported model found in {PATHS.models}. Did you run training?")
            
        logger.info(f"Loading primary inference model [{MODEL_NAME}] from {best_model_path}...")
        _model = tf.keras.models.load_model(str(best_model_path))
    return _model

def run_prediction_sync(image_path: str):
    """
    Runs the core prediction and synchronous XAI (Grad-CAM, LIME).
    Returns (results, processing_time).
    """
    start_time = time.time()
    model = get_model()
    
    # 1. Exact MobileNetV2 Preprocessing to match training exactly
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # DullRazor (Hair Removal) -> CLAHE (Mocked inside preprocess_image_cv2)
    processed_img = preprocess_image_cv2(img)
    
    # Resize exactly
    processed_img = cv2.resize(processed_img, (224, 224))
    
    # Tensor Conversion & Model-Specific Preprocessing (MobileNetV2)
    input_tensor = np.expand_dims(processed_img, axis=0)
    input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor.astype(np.float32))
    
    # Prediction
    final_prediction = model.predict(input_tensor, verbose=0)[0]
    
    from src.config import DATASET
    
    top_3_idx = np.argsort(final_prediction)[-3:][::-1]
    
    # Medical Risk Mapping Logic (Disease-specific)
    top_class_name = DATASET.class_names[top_3_idx[0]]
    if top_class_name in ["MEL", "BCC", "AKIEC"]:
        risk_level = "High"
        recommendation = "URGENT: Consult an oncologist or dermatologist immediately for biopsy."
    elif top_class_name in ["DF", "BKL"]:
        risk_level = "Medium"
        recommendation = "Schedule a dermatological review. Lesion appears benign but requires professional confirmation."
    else:
        risk_level = "Low"
        recommendation = "Routine monitoring. If the lesion changes in size, color, or shape, consult a dermatologist."

    # STRICT PIPELINE DIAGNOSTIC LOGGING
    logger.info("=== PREDICTION PIPELINE TRACE ===")
    logger.info(f"Image Path     : {image_path}")
    logger.info(f"Preprocessing  : DullRazor -> CLAHE -> Resize(224x224) -> mobilenet_v2.preprocess_input")
    logger.info(f"Tensor Shape   : {input_tensor.shape}")
    logger.info(f"Predicted Class: {top_class_name}")
    logger.info(f"Confidence     : {final_prediction[top_3_idx[0]]:.4f}")
    logger.info(f"Mapped Disease : {top_class_name}")
    logger.info(f"Mapped Risk    : {risk_level}")
    logger.info(f"Recommendation : {recommendation}")
    logger.info("=================================")

    results = [
        {"class_name": DATASET.class_names[idx], "confidence": float(final_prediction[idx])}
        for idx in top_3_idx
    ]
    
    # Generate Grad-CAM immediately
    from src.explainability import find_last_conv_layer, make_gradcam_heatmap
    last_conv = find_last_conv_layer(model)
    gradcam_path = None
    if last_conv:
        heatmap = make_gradcam_heatmap(tf.convert_to_tensor(input_tensor), model, last_conv)
        save_and_display_overlay(image_path, heatmap, MODEL_NAME, f"gradcam_{os.path.basename(image_path)}")
        from src.config import PATHS
        gradcam_path = str(PATHS.figures / f"{MODEL_NAME}_gradcam_{os.path.basename(image_path)}")
        
    processing_time = time.time() - start_time
    return results, processing_time, gradcam_path

def run_async_xai(scan_id: str, image_path: str):
    """
    Background task to generate SHAP and Integrated Gradients.
    Updates the SQLite database when complete.
    """
    logger.info(f"Starting async XAI for {scan_id}")
    try:
        from api.database import SessionLocal
        from api.models import ScanHistory
        from api.services.explainability import run_async_xai_pipeline

        model = get_model()
        
        # We need the tensor again
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = preprocess_image_cv2(img)
        processed_img = cv2.resize(processed_img, (224, 224))
        input_tensor = np.expand_dims(processed_img, axis=0)
        input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor.astype(np.float32))
        input_tensor_tf = tf.convert_to_tensor(input_tensor)
        
        shap_path, ig_path, lime_path, gradcam_pp_path = run_async_xai_pipeline(model, input_tensor_tf, image_path, scan_id)

        # Update Database
        db = SessionLocal()
        item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
        if item:
            if shap_path: item.shap_path = shap_path
            if ig_path: item.ig_path = ig_path
            if lime_path: item.lime_path = lime_path
            if gradcam_pp_path: item.gradcam_plus_plus_path = gradcam_pp_path
            item.status = "xai_completed"
            db.commit()
        db.close()
        
        logger.info(f"Async XAI completed for {scan_id}")

    except Exception as e:
        logger.error(f"Error in async XAI task for {scan_id}: {str(e)}")
        
        # Mark as failed in DB
        try:
            from api.database import SessionLocal
            from api.models import ScanHistory
            db = SessionLocal()
            item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
            if item:
                item.status = "xai_failed"
                db.commit()
            db.close()
        except:
            pass
        from api.database import SessionLocal
        from api.models import ScanHistory
        db = SessionLocal()
        record = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
        if record:
            record.async_xai_status = "failed"
            db.commit()
        db.close()
