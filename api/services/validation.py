import tensorflow as tf
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ImageValidator:
    """
    Stage 1: Intelligent Image Validation
    Combines MobileNetV2 blocklist, YCrCb Skin Segmentation, and Vignette Detection.
    """
    def __init__(self):
        logger.info("Loading ImageNet Validator (MobileNetV2)...")
        # Load lightweight MobileNetV2
        self.model = tf.keras.applications.MobileNetV2(weights='imagenet')
        self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # ImageNet classes that are definitively NOT skin lesions.
        self.block_keywords = [
            'dog', 'cat', 'car', 'vehicle', 'truck', 'bird', 'fish', 'flower', 
            'landscape', 'building', 'food', 'pizza', 'burger', 'laptop', 'phone', 
            'furniture', 'chair', 'desk', 'person', 'face', 'hand', 'document',
            'envelope', 'book', 'screenshot', 'text', 'menu', 'screen', 'monitor'
        ]

    def _check_human_skin(self, img) -> float:
        """
        Calculates the percentage of human skin pixels using YCrCb color space.
        """
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Standard YCrCb bounds for human skin
        lower_skin = np.array([0, 135, 85])
        upper_skin = np.array([255, 180, 135])
        skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
        
        # Apply morphological opening to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = img.shape[0] * img.shape[1]
        
        return (skin_pixels / total_pixels) * 100.0

    def _detect_image_type(self, img) -> str:
        """
        Determines if the image is 'Dermoscopic' or 'Smartphone'.
        Dermoscopic images typically have a dark circular vignette masking the corners.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold dark pixels (the vignette mask)
        _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        
        # Check corners. If all four corners are extremely dark, it's likely a dermoscopic vignette.
        h, w = gray.shape
        corner_size = min(h, w) // 10
        
        tl = np.mean(dark_mask[0:corner_size, 0:corner_size])
        tr = np.mean(dark_mask[0:corner_size, w-corner_size:w])
        bl = np.mean(dark_mask[h-corner_size:h, 0:corner_size])
        br = np.mean(dark_mask[h-corner_size:h, w-corner_size:w])
        
        # If the average intensity of the dark mask in the corners is high, the vignette exists.
        corners = [tl, tr, bl, br]
        vignette_score = sum(1 for c in corners if c > 200) # At least >80% dark in the corner
        
        if vignette_score >= 3: # 3 or 4 corners are dark
            return "Dermoscopic"
        
        # Additional check: Circular contour detection
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.1): # Large dark object
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.6:
                    return "Dermoscopic"
                    
        return "Smartphone"

    def validate(self, image_path: str) -> dict:
        """
        Validates if the image is suitable for skin cancer detection.
        Returns a dictionary with validation results.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"is_valid": False, "reason": "Could not read image file.", "image_type": "Unknown"}
            
            # 1. Skin Detection
            skin_percentage = self._check_human_skin(img)
            logger.info(f"Skin Detection: {skin_percentage:.1f}%")
            if skin_percentage < 10.0:
                return {
                    "is_valid": False,
                    "reason": f"This image does not appear to contain human skin (Detected: {skin_percentage:.1f}%). Please upload a skin lesion image.",
                    "image_type": "Non-Skin Object"
                }

            # 2. Dermoscopic vs Smartphone Detection
            image_type = self._detect_image_type(img)
            logger.info(f"Image Type Detected: {image_type}")

            # 3. MobileNetV2 Blocklist (Objects, Animals)
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            x = np.expand_dims(img_rgb, axis=0)
            x = self.preprocess_input(x.astype(np.float32))

            preds = self.model.predict(x, verbose=0)
            decoded = self.decode_predictions(preds, top=3)[0]
            
            logger.info(f"MobileNetV2 Top Predictions: {decoded}")
            
            for _, class_name, prob in decoded:
                class_name_lower = class_name.lower()
                if any(kw in class_name_lower for kw in self.block_keywords) and prob > 0.4:
                    # It's an object/animal with high confidence
                    return {
                        "is_valid": False,
                        "reason": f"This image appears to be a {class_name.replace('_', ' ')} (confidence: {prob:.0%}). The AI requires a human skin lesion image.",
                        "image_type": "Non-Skin Object"
                    }

            return {
                "is_valid": True,
                "reason": "Validation Passed",
                "image_type": image_type
            }

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return {"is_valid": False, "reason": "Internal server error during validation.", "image_type": "Unknown"}
