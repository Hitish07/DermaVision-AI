import os
import time
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD
from .config import PATHS, TRAIN, DATASET
from .models import build_model, build_multimodal_fusion_model

logger = logging.getLogger(__name__)

def setup_mixed_precision():
    if TRAIN.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled ('mixed_float16').")

def get_optimizer(name, lr):
    """Returns the requested optimizer with gradient clipping enabled."""
    clipnorm = 1.0 # Gradient clipping
    if name.lower() == 'adam':
        return Adam(learning_rate=lr, clipnorm=clipnorm)
    elif name.lower() == 'adamw':
        return AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=clipnorm)
    elif name.lower() == 'rmsprop':
        return RMSprop(learning_rate=lr, clipnorm=clipnorm)
    elif name.lower() == 'sgd':
        return SGD(learning_rate=lr, momentum=0.9, clipnorm=clipnorm)
    else:
        return Adam(learning_rate=lr, clipnorm=clipnorm)

def get_callbacks(model_name, phase):
    checkpoint_path = PATHS.checkpoints / f"{model_name}_{phase}_best.keras"
    return [
        EarlyStopping(monitor='val_loss', patience=TRAIN.early_stopping_patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=TRAIN.reduce_lr_patience, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor='val_loss'),
        TensorBoard(log_dir=str(PATHS.logs / f"{model_name}_{phase}_{int(time.time())}")),
        CSVLogger(str(PATHS.logs / f"{model_name}_{phase}_training.csv"), append=True)
    ]

def train_model(model_name, train_dataset, val_dataset, class_weights=None):
    setup_mixed_precision()
    logger.info(f"--- Starting training for {model_name} ---")
    model, base_model = build_model(model_name)
    
    # Check if a checkpoint exists to resume
    phase2_ckpt = PATHS.checkpoints / f"{model_name}_phase2_best.keras"
    phase1_ckpt = PATHS.checkpoints / f"{model_name}_phase1_best.keras"
    
    if phase2_ckpt.exists():
        logger.info(f"Resuming from Phase 2 checkpoint: {phase2_ckpt}")
        model = tf.keras.models.load_model(str(phase2_ckpt))
        return model
    elif phase1_ckpt.exists():
        logger.info(f"Resuming from Phase 1 checkpoint: {phase1_ckpt}")
        model = tf.keras.models.load_model(str(phase1_ckpt))
    else:
        # Phase 1: Train Custom Head
        logger.info("Phase 1: Training top layers only...")
        optimizer = get_optimizer(TRAIN.optimizer, TRAIN.learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        
        callbacks = get_callbacks(model_name, "phase1")
        
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRAIN.epochs_head,
            class_weight=class_weights if TRAIN.class_imbalance_strategy == 'class_weights' else None,
            callbacks=callbacks
        )
    
    # Phase 2: Fine-Tuning
    if base_model is not None:
        logger.info("Phase 2: Fine-tuning entire network...")
        base_model.trainable = True
        
        ft_optimizer = get_optimizer(TRAIN.optimizer, TRAIN.finetune_lr)
        model.compile(optimizer=ft_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        
        ft_callbacks = get_callbacks(model_name, "phase2")
        
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRAIN.epochs_finetune,
            class_weight=class_weights if TRAIN.class_imbalance_strategy == 'class_weights' else None,
            callbacks=ft_callbacks
        )
    
    # Save final models in various formats
    final_path = PATHS.models / f"{model_name}_final.keras"
    model.save(str(final_path))
    model.save(str(PATHS.models / f"{model_name}_final.h5"))
    try:
        model.export(str(PATHS.models / f"{model_name}_savedmodel"))
    except Exception as e:
        logger.warning(f"SavedModel export failed (may require TF2.15+): {e}")
        
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(PATHS.models / f"{model_name}_final.tflite", 'wb') as f:
            f.write(tflite_model)
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        
    try:
        import tf2onnx
        spec = (tf.TensorSpec((None, *DATASET.image_size, 3), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(PATHS.models / f"{model_name}_final.onnx"))
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        
    logger.info(f"Model {model_name} saved and exported successfully.")
    
    return model
