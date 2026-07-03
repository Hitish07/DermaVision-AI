import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import logging
import os
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
from .config import PATHS, DATASET

logger = logging.getLogger(__name__)

def find_last_conv_layer(model):
    import tensorflow as tf
    # Flatten all layers if it's a nested model
    def get_all_layers(m):
        layers = []
        for layer in m.layers:
            if isinstance(layer, tf.keras.Model):
                layers.extend(get_all_layers(layer))
            else:
                layers.append(layer)
        return layers

    all_layers = get_all_layers(model)
    for layer in reversed(all_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, plus_plus=False):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            if plus_plus:
                grads = tape3.gradient(class_channel, last_conv_layer_output)
            else:
                grads = tape1.gradient(class_channel, last_conv_layer_output)

        if plus_plus:
            first_derivative = grads
            second_derivative = tape2.gradient(first_derivative, last_conv_layer_output)
            third_derivative = tape1.gradient(second_derivative, last_conv_layer_output)
            
            global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))
            alpha_num = second_derivative
            alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
            alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
            alphas = alpha_num / alpha_denom
            weights = tf.maximum(first_derivative, 0.0)
            alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0,1,2))
            alphas /= alpha_normalization_constant
            pooled_grads = tf.reduce_sum(weights * alphas, axis=(0,1,2))
        else:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_integrated_gradients(img_array, model, pred_index=None, steps=50):
    baseline = tf.zeros(shape=img_array.shape)
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    
    gradient_batches = []
    for alpha in alphas:
        interpolated_image = baseline + alpha * (img_array - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_image)
            preds = model(interpolated_image)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, interpolated_image)
        gradient_batches.append(grads)
        
    avg_gradients = tf.reduce_mean(gradient_batches, axis=0)
    integrated_gradients = (img_array - baseline) * avg_gradients
    
    # Process for visualization
    ig_img = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
    ig_img = tf.squeeze(ig_img)
    ig_img = ig_img / tf.math.reduce_max(ig_img)
    return ig_img.numpy()

def generate_shap(img_array, model):
    # Use DeepExplainer or GradientExplainer
    background = tf.zeros_like(img_array) # Simple background
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_array.numpy() if hasattr(img_array, 'numpy') else img_array)
    return shap_values

def generate_lime(img_array, model):
    explainer = lime_image.LimeImageExplainer()
    # img_array should be numpy
    img_np = img_array[0].numpy() if hasattr(img_array, 'numpy') else img_array[0]
    
    def predict_fn(images):
        return model.predict(images, verbose=0)
        
    explanation = explainer.explain_instance(
        img_np.astype('double'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=50 # Reduced for speed
    )
    return explanation

def save_and_display_overlay(img_path, map_array, model_name, suffix, alpha=0.4, is_heatmap=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, DATASET.image_size)

    if is_heatmap:
        map_array = np.uint8(255 * map_array)
        import matplotlib as mpl
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        colored_map = jet_colors[map_array]
        colored_map = tf.keras.preprocessing.image.array_to_img(colored_map)
        colored_map = colored_map.resize((img.shape[1], img.shape[0]))
        colored_map = tf.keras.preprocessing.image.img_to_array(colored_map)
        
        superimposed_img = colored_map * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    else:
        # Just map directly (like for LIME boundaries)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(map_array)

    output_path = PATHS.figures / f"{model_name}_{suffix}"
    superimposed_img.save(str(output_path))
    logger.info(f"Saved explanation to {output_path}")

def explain_prediction(model, model_name, image_path, preprocessed_image_tensor, full_suite=False):
    """Generates explainability visualizations."""
    try:
        last_conv = find_last_conv_layer(model)
        if last_conv:
            # Grad-CAM
            heatmap = make_gradcam_heatmap(preprocessed_image_tensor, model, last_conv)
            save_and_display_overlay(image_path, heatmap, model_name, f"gradcam_{os.path.basename(image_path)}")
            
            # Grad-CAM++
            heatmap_pp = make_gradcam_heatmap(preprocessed_image_tensor, model, last_conv, plus_plus=True)
            save_and_display_overlay(image_path, heatmap_pp, model_name, f"gradcam++_{os.path.basename(image_path)}")
        else:
            logger.warning(f"No conv layer found in {model_name} for Grad-CAM.")

        if full_suite:
            # Integrated Gradients
            logger.info("Running Integrated Gradients...")
            ig_map = get_integrated_gradients(preprocessed_image_tensor, model)
            save_and_display_overlay(image_path, ig_map, model_name, f"ig_{os.path.basename(image_path)}")
            
            # LIME
            logger.info("Running LIME...")
            explanation = generate_lime(preprocessed_image_tensor, model)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            lime_bound = mark_boundaries(temp, mask)
            save_and_display_overlay(image_path, lime_bound * 255.0, model_name, f"lime_{os.path.basename(image_path)}", is_heatmap=False)
            
            # SHAP
            logger.info("Running SHAP...")
            try:
                shap_vals = generate_shap(preprocessed_image_tensor, model)
                # Just saving a basic visualization is hard without plt.show(), so we skip strict image saving for SHAP
                # or save it via shap.image_plot
                plt.figure()
                shap.image_plot(shap_vals, preprocessed_image_tensor.numpy(), show=False)
                plt.savefig(PATHS.figures / f"{model_name}_shap_{os.path.basename(image_path)}")
                plt.close()
            except Exception as e:
                logger.error(f"SHAP failed: {e}")

    except Exception as e:
        logger.error(f"Explainability failed for {model_name}: {e}")
