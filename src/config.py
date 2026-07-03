import os
from pathlib import Path
from dataclasses import dataclass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"C:\Users\admin\Downloads\Skin_Scan_AI")

@dataclass
class Paths:
    metadata_csv: Path = DATA_DIR / "HAM10000_metadata.csv"
    images_part_1: Path = DATA_DIR / "HAM10000_images_part_1"
    images_part_2: Path = DATA_DIR / "HAM10000_images_part_2"
    
    outputs: Path = BASE_DIR / "outputs"
    models: Path = BASE_DIR / "models"
    reports: Path = BASE_DIR / "reports"
    logs: Path = BASE_DIR / "logs"
    experiments: Path = BASE_DIR / "experiments"
    figures: Path = BASE_DIR / "figures"
    checkpoints: Path = BASE_DIR / "checkpoints"

@dataclass
class DatasetConfig:
    class_names: tuple = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')
    num_classes: int = len(class_names)
    image_size: tuple = (224, 224)
    image_channels: int = 3
    test_size: float = 0.1
    val_size: float = 0.1 # This will be 0.1 of the total, so val is 10%, test 10%, train 80%
    random_seed: int = 42

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs_head: int = 15
    epochs_finetune: int = 40
    learning_rate: float = 1e-3
    finetune_lr: float = 1e-4
    mixed_precision: bool = True
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    optimizer: str = 'adam' # 'adam', 'adamw', 'rmsprop', 'sgd'
    loss_function: str = 'categorical_crossentropy' # 'categorical_crossentropy', 'focal_loss', 'weighted_crossentropy'
    class_imbalance_strategy: str = 'class_weights' # 'class_weights', 'oversampling', 'none'
    
@dataclass
class ModelsConfig:
    architectures = [
        "CustomCNN",
        "ResNet50",
        "EfficientNetB0",
        "EfficientNetB3",
        "DenseNet121",
        "MobileNetV2",
        "InceptionV3",
        "Xception",
        "ConvNeXtTiny",
        "ViT"
    ]

# Initialize configurations
PATHS = Paths()
DATASET = DatasetConfig()
TRAIN = TrainingConfig()
MODELS = ModelsConfig()

# Ensure all output directories exist
for path in [PATHS.outputs, PATHS.models, PATHS.reports, PATHS.logs, PATHS.experiments, PATHS.figures, PATHS.checkpoints]:
    path.mkdir(parents=True, exist_ok=True)
