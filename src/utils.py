import os
import random
import numpy as np
import tensorflow as tf
import logging
from time import time
from functools import wraps
from .config import PATHS, DATASET

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(PATHS.logs / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_random_seed(seed=DATASET.random_seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    # Optionally, for full reproducibility in TF
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def setup_hardware():
    logger = logging.getLogger(__name__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            logger.error(f"Error configuring GPUs: {e}")
    else:
        logger.warning("No GPUs found. Falling back to CPU. Training will be slow.")

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logger = logging.getLogger(__name__)
        logger.info(f"Execution time for '{func.__name__}': {end - start:.2f} seconds.")
        return result
    return wrapper
