import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from .config import PATHS, DATASET
from .utils import timeit

logger = logging.getLogger(__name__)

class SkinDataset:
    def __init__(self):
        self.metadata = pd.read_csv(PATHS.metadata_csv)
        self.image_paths = self._get_image_paths()
        self._map_paths_to_metadata()
        self.class_mapping = {c: i for i, c in enumerate(DATASET.class_names)}
        self.metadata['label'] = self.metadata['dx'].map(self.class_mapping)

    def _get_image_paths(self):
        """Scans both part 1 and part 2 directories to map image_id to file paths."""
        paths = {}
        for folder in [PATHS.images_part_1, PATHS.images_part_2]:
            if not folder.exists():
                logger.warning(f"Image folder not found: {folder}")
                continue
            for img_name in os.listdir(folder):
                if img_name.endswith('.jpg'):
                    img_id = img_name.replace('.jpg', '')
                    paths[img_id] = str(folder / img_name)
        return paths

    def _map_paths_to_metadata(self):
        self.metadata['image_path'] = self.metadata['image_id'].map(self.image_paths)
        missing = self.metadata['image_path'].isna().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing images in dataset! Removing them from metadata.")
            self.metadata = self.metadata.dropna(subset=['image_path']).reset_index(drop=True)

    @timeit
    def perform_eda(self):
        """Generates dataset statistics and handles missing metadata values."""
        logger.info("Performing Exploratory Data Analysis...")
        
        # 1. Dataset stats
        stats = {
            'Total Images': len(self.metadata),
            'Unique Lesions': self.metadata['lesion_id'].nunique(),
            'Classes': DATASET.num_classes
        }
        with open(PATHS.reports / 'dataset_stats.txt', 'w') as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        
        # 2. Class Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.metadata, x='dx', order=DATASET.class_names)
        plt.title("Class Distribution")
        plt.savefig(PATHS.figures / 'class_distribution.png', dpi=300)
        plt.close()
        
        # 3. Handle missing metadata
        self.metadata['age'] = self.metadata['age'].fillna(self.metadata['age'].median())
        self.metadata['sex'] = self.metadata['sex'].fillna('unknown')
        self.metadata['localization'] = self.metadata['localization'].fillna('unknown')

        logger.info("EDA completed and plots saved.")

    def create_splits(self):
        """Splits data grouping by lesion_id to prevent data leakage."""
        logger.info("Splitting dataset grouped by lesion_id...")
        
        # Test split
        gss_test = GroupShuffleSplit(n_splits=1, test_size=DATASET.test_size, random_state=DATASET.random_seed)
        train_val_idx, test_idx = next(gss_test.split(self.metadata, groups=self.metadata['lesion_id']))
        
        train_val_df = self.metadata.iloc[train_val_idx].reset_index(drop=True)
        test_df = self.metadata.iloc[test_idx].reset_index(drop=True)
        
        # Validation split
        val_size_adjusted = DATASET.val_size / (1.0 - DATASET.test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=DATASET.random_seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['lesion_id']))
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        
        logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Save splits to csv for reproducibility
        train_df.to_csv(PATHS.experiments / 'train_split.csv', index=False)
        val_df.to_csv(PATHS.experiments / 'val_split.csv', index=False)
        test_df.to_csv(PATHS.experiments / 'test_split.csv', index=False)
        
        return train_df, val_df, test_df
        
    def get_class_weights(self, train_df):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(train_df['label'])
        weights = compute_class_weight('balanced', classes=classes, y=train_df['label'].values)
        return dict(zip(classes, weights))

    def apply_oversampling(self, train_df):
        """Applies static dataframe-based oversampling."""
        logger.info("Applying static oversampling to balance classes...")
        from sklearn.utils import resample
        
        max_size = train_df['label'].value_counts().max()
        oversampled_dfs = []
        
        for class_idx in range(DATASET.num_classes):
            class_df = train_df[train_df['label'] == class_idx]
            if len(class_df) == 0:
                continue
            if len(class_df) < max_size:
                oversampled = resample(class_df, replace=True, n_samples=max_size, random_state=DATASET.random_seed)
                oversampled_dfs.append(oversampled)
            else:
                oversampled_dfs.append(class_df)
                
        balanced_train_df = pd.concat(oversampled_dfs).sample(frac=1, random_state=DATASET.random_seed).reset_index(drop=True)
        logger.info(f"Oversampling complete. Original size: {len(train_df)}, New size: {len(balanced_train_df)}")
        return balanced_train_df
