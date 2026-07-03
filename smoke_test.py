import argparse
import logging
import tensorflow as tf
from src.utils import setup_logging, set_random_seed, setup_hardware
from src.dataset import SkinDataset
from src.config import TRAIN
from src.train import train_model
from src.evaluate import evaluate_model
from main import build_tf_dataset
import pandas as pd

def run_smoke_test():
    logger = setup_logging()
    set_random_seed()
    setup_hardware()
    
    logger.info("Starting Smoke Test...")
    dataset = SkinDataset()
    
    # 1. Subset Data (ensure we have enough classes by taking head, but it might not have all classes. 
    # Let's take a stratified sample or just a larger head)
    dataset.metadata = dataset.metadata.sample(n=400, random_state=42).reset_index(drop=True)
    
    train_df, val_df, test_df = dataset.create_splits()
    
    # 2. Test Oversampling
    logger.info("Testing Oversampling...")
    train_df = dataset.apply_oversampling(train_df)
    
    # 3. Modify Config for Smoke Test
    TRAIN.epochs_head = 2
    TRAIN.epochs_finetune = 1 # Test phase 2 logic
    TRAIN.batch_size = 16
    
    train_ds = build_tf_dataset(train_df, is_training=True)
    val_ds = build_tf_dataset(val_df, is_training=False)
    test_ds = build_tf_dataset(test_df, is_training=False)
    
    # 4. Train
    logger.info("Testing Training (MobileNetV2)...")
    model = train_model("MobileNetV2", train_ds, val_ds)
    
    # 5. Evaluate
    logger.info("Testing Evaluation...")
    evaluate_model(model, "MobileNetV2", test_ds, test_df['label'].values)
    
    # 6. Test Explainability
    logger.info("Testing Explainability...")
    from src.explainability import explain_prediction
    from src.preprocessing import tf_preprocess_image
    
    # Take first test image
    test_img_path = test_df.iloc[0]['image_path']
    # Use the tf.data pipeline to get the preprocessed tensor
    img_tensor, _ = tf_preprocess_image(test_img_path, 0)
    import numpy as np
    img_tensor = np.expand_dims(img_tensor.numpy(), axis=0)
    
    explain_prediction(model, "MobileNetV2", test_img_path, tf.convert_to_tensor(img_tensor), full_suite=True)
    
    logger.info("Smoke Test Completed Successfully.")

if __name__ == "__main__":
    run_smoke_test()
