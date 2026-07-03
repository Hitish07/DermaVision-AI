import albumentations as A
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_train_transforms():
    """Returns the Albumentations composition for training."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ColorJitter(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2),
        A.GaussianBlur(p=0.1),
        A.MotionBlur(p=0.1)
    ])

def get_tta_transforms():
    """Returns minimal transformations for Test Time Augmentation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5)
    ])

def tf_augment_image(image, label):
    """Wraps Albumentations into a tf.data pipeline."""
    def _augment(img):
        # img is a float32 tensor [0, 1]
        img_np = img.numpy()
        aug = get_train_transforms()(image=img_np)
        return aug['image'].astype(np.float32)

    [augmented_image] = tf.py_function(_augment, [image], [tf.float32])
    augmented_image.set_shape(image.shape)
    return augmented_image, label
