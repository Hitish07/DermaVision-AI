import os
import cv2
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt

from src.explainability import get_integrated_gradients, generate_shap
from src.config import PATHS

logger = logging.getLogger(__name__)

def run_async_xai_pipeline(model, img_tensor, image_path: str, scan_id: str):
    """
    Stage 5: Explainability Pipeline (Heavy computations).
    Runs SHAP, LIME, and Integrated Gradients and saves artifacts.
    """
    logger.info(f"Starting XAI Pipeline for {scan_id}")
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot run XAI: image {image_path} not found.")
        return None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # Integrated Gradients
        logger.info("Generating Integrated Gradients...")
        ig_map = get_integrated_gradients(img_tensor, model)
        
        # Save IG heatmap overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.imshow(ig_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        
        ig_path = os.path.join(figures_dir, f"{scan_id}_integrated_gradients.jpg")
        plt.savefig(ig_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    except Exception as e:
        logger.error(f"IG Failed: {e}")
        ig_path = None
        
    try:
        # SHAP
        logger.info("Generating SHAP...")
        shap_values = generate_shap(img_tensor, model)
        
        # We just grab the first class's shap values for a basic heatmap
        shap_map = np.abs(shap_values[0][0].sum(axis=-1))
        shap_map = cv2.resize(shap_map, (img.shape[1], img.shape[0]))
        shap_map = shap_map / (shap_map.max() + 1e-9)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.imshow(shap_map, cmap='hot', alpha=0.5)
        plt.axis('off')
        
        shap_path = os.path.join(figures_dir, f"{scan_id}_shap.jpg")
        plt.savefig(shap_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    except Exception as e:
        logger.error(f"SHAP Failed: {e}")
        shap_path = None
        
    try:
        from src.explainability import generate_lime, mark_boundaries
        logger.info("Generating LIME...")
        explanation = generate_lime(img_tensor, model)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        lime_bound = mark_boundaries(temp, mask)
        
        lime_path = os.path.join(figures_dir, f"{scan_id}_lime.jpg")
        cv2.imwrite(lime_path, cv2.cvtColor(np.uint8(lime_bound * 255), cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.error(f"LIME Failed: {e}")
        lime_path = None
        
    try:
        from src.explainability import make_gradcam_heatmap, find_last_conv_layer
        logger.info("Generating Grad-CAM++...")
        last_conv = find_last_conv_layer(model)
        heatmap_pp = make_gradcam_heatmap(img_tensor, model, last_conv, plus_plus=True)
        
        heatmap_pp = cv2.resize(heatmap_pp, (img.shape[1], img.shape[0]))
        heatmap_pp = np.uint8(255 * heatmap_pp)
        heatmap_pp = cv2.applyColorMap(heatmap_pp, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_pp, 0.4, 0)
        
        gradcam_plus_plus_path = os.path.join(figures_dir, f"{scan_id}_gradcam++.jpg")
        cv2.imwrite(gradcam_plus_plus_path, superimposed_img)
    except Exception as e:
        logger.error(f"Grad-CAM++ Failed: {e}")
        gradcam_plus_plus_path = None
        
    logger.info(f"XAI Pipeline Complete for {scan_id}")
    return shap_path, ig_path, lime_path, gradcam_plus_plus_path
