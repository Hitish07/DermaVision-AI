import argparse
import logging
import tensorflow as tf
from src.utils import setup_logging, set_random_seed, setup_hardware
from src.dataset import SkinDataset
from src.config import MODELS, TRAIN
from src.train import train_model
from src.evaluate import evaluate_model
from src.preprocessing import tf_preprocess_image
from src.augmentation import tf_augment_image

def build_tf_dataset(df, is_training=False):
    """Builds optimized tf.data.Dataset with caching and prefetching."""
    path_ds = tf.data.Dataset.from_tensor_slices(df['image_path'].values)
    label_ds = tf.data.Dataset.from_tensor_slices(df['label'].values)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    
    # 1. Map OpenCV preprocessing (DullRazor + CLAHE + Resize)
    ds = ds.map(tf_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Cache to RAM/Disk after expensive CPU preprocessing to accelerate subsequent epochs
    ds = ds.cache()
    
    # 3. Augment (Only for training)
    if is_training:
        ds = ds.map(tf_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)
        
    # 4. Batch and Prefetch
    ds = ds.batch(TRAIN.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Detection Research Pipeline")
    parser.add_argument('--run_eda', action='store_true', help='Run Exploratory Data Analysis')
    parser.add_argument('--train', action='store_true', help='Train all baseline models')
    args = parser.parse_args()

    logger = setup_logging()
    set_random_seed()
    setup_hardware()
    
    logger.info("Initializing Dataset...")
    dataset = SkinDataset()
    
    if args.run_eda:
        dataset.perform_eda()
        
    train_df, val_df, test_df = dataset.create_splits()
    
    class_weights = dataset.get_class_weights(train_df) if TRAIN.class_imbalance_strategy == 'class_weights' else None
    
    train_ds = build_tf_dataset(train_df, is_training=True)
    val_ds = build_tf_dataset(val_df, is_training=False)
    test_ds = build_tf_dataset(test_df, is_training=False)
    
    if args.train:
        trained_models = {}
        for model_name in MODELS.architectures:
            try:
                model = train_model(model_name, train_ds, val_ds, class_weights)
                trained_models[model_name] = model
                
                # Evaluate on Test Set
                evaluate_model(model, model_name, test_ds, test_df['label'].values)
                
            except Exception as e:
                logger.error(f"Failed to train/evaluate {model_name}: {e}")

    logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    main()
