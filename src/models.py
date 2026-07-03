import tensorflow as tf
from tensorflow.keras import layers, models, applications
import logging
from .config import DATASET

logger = logging.getLogger(__name__)

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    return inputs, x

def get_backbone(model_name, input_shape):
    """Factory to get the requested backbone architecture."""
    if model_name == "CustomCNN":
        return build_custom_cnn(input_shape)
    
    inputs = layers.Input(shape=input_shape)
    
    if model_name == "ResNet50":
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "EfficientNetB0":
        base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "EfficientNetB3":
        base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "DenseNet121":
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "MobileNetV2":
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "InceptionV3":
        base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "Xception":
        base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "ConvNeXtTiny":
        base_model = applications.ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=inputs)
    elif model_name == "ViT":
        from vit_keras import vit
        base_model = vit.vit_b16(
            image_size=DATASET.image_size[0],
            activation='linear',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=DATASET.num_classes
        )
        x = base_model(inputs)
        return inputs, x, base_model
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    # For standard TF applications
    base_model.trainable = False # Freeze backbone initially
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    return inputs, x, base_model

def build_model(model_name="EfficientNetB0"):
    """Builds an image-only classification model."""
    logger.info(f"Building {model_name} model...")
    input_shape = (*DATASET.image_size, DATASET.image_channels)
    
    if model_name == "CustomCNN":
        inputs, x = build_custom_cnn(input_shape)
        base_model = None
    else:
        inputs, x, base_model = get_backbone(model_name, input_shape)

    # Classification Head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(DATASET.num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, base_model

def build_multimodal_fusion_model(base_image_model, num_metadata_features=10):
    """
    Builds a multi-modal fusion model taking the pre-trained image model 
    and fusing it with metadata features.
    """
    logger.info("Building Multi-modal Fusion model...")
    
    # 1. Image Branch (Reuse the trained base image model up to the GlobalAveragePooling layer)
    # We find the GlobalAveragePooling layer to extract features before the dense head
    gap_layer = None
    for layer in base_image_model.layers:
        if isinstance(layer, layers.GlobalAveragePooling2D) or 'global_average_pooling2d' in layer.name:
            gap_layer = layer
            break
            
    if gap_layer is None: # Fallback if using ViT which extracts CLS token
        gap_layer = base_image_model.layers[-7] # approximate before Dense heads
        
    image_features = gap_layer.output
    
    # 2. Metadata Branch
    metadata_inputs = layers.Input(shape=(num_metadata_features,), name='metadata_input')
    m = layers.Dense(64, activation='relu')(metadata_inputs)
    m = layers.BatchNormalization()(m)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(128, activation='relu')(m)
    metadata_features = layers.BatchNormalization()(m)
    
    # 3. Late Fusion
    fused = layers.Concatenate()([image_features, metadata_features])
    
    f = layers.Dense(512, activation='relu')(fused)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.5)(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.3)(f)
    
    outputs = layers.Dense(DATASET.num_classes, activation='softmax', dtype='float32', name='fusion_output')(f)
    
    model = models.Model(inputs=[base_image_model.input, metadata_inputs], outputs=outputs, name="MultiModal_Fusion")
    return model
