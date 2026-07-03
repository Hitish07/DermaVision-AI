import os
import tensorflow as tf
from src.config import MODELS, DATASET
from src.models import build_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_models")

def verify_all():
    input_shape = (1, *DATASET.image_size, 3)
    dummy_input = tf.random.uniform(input_shape, minval=0, maxval=1)
    dummy_labels = tf.convert_to_tensor([[0, 1, 0, 0, 0, 0, 0]], dtype=tf.float32)

    results = []
    
    for model_name in MODELS.architectures:
        try:
            logger.info(f"Testing {model_name}...")
            # 1. Instantiate
            model, base_model = build_model(model_name)
            
            # 2. Forward Pass
            preds = model(dummy_input)
            assert preds.shape == (1, DATASET.num_classes), f"Output shape mismatch: {preds.shape}"
            
            # 3. Loss & Gradients
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            with tf.GradientTape() as tape:
                preds = model(dummy_input, training=True)
                loss = loss_fn(dummy_labels, preds)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            assert gradients is not None and len(gradients) > 0, "Gradients could not be computed!"
            
            # 4. Backpropagation
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            logger.info(f"[SUCCESS] {model_name} passes forward/backward verification.")
            results.append((model_name, "PASS"))
            
        except Exception as e:
            logger.error(f"[FAILED] {model_name}: {e}")
            results.append((model_name, "FAIL"))

    # Print Summary
    print("\n===============================")
    print("      VERIFICATION SUMMARY       ")
    print("===============================")
    for model_name, status in results:
        print(f"{model_name:20s} : {status}")

if __name__ == "__main__":
    verify_all()
