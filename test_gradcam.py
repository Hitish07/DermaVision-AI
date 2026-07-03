import tensorflow as tf
from src.explainability import find_last_conv_layer
from src.config import PATHS

model = tf.keras.models.load_model(str(PATHS.models / "MobileNetV2_final.keras"))
print(model.summary())
print("Last conv layer:", find_last_conv_layer(model))
