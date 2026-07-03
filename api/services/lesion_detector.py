import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class LesionDetector:
    """
    Stage 3: Advanced Morphological OpenCV Lesion Detection
    Proves that a lesion actually exists on the skin before sending to TensorFlow.
    """
    
    def __init__(self, figures_dir="figures"):
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def _remove_hair(self, img):
        """DullRazor heuristic hair removal."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)
        return inpainted

    def _apply_clahe(self, img):
        """CLAHE enhancement on L channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def detect(self, image_path: str, scan_id: str) -> tuple[bool, str]:
        """
        Runs the full detection pipeline. Saves intermediate artifacts.
        Returns (lesion_detected, message).
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Failed to load image for lesion detection."
                
            h, w = img.shape[:2]
            
            # 1. Hair Removal
            hair_removed = self._remove_hair(img)
            cv2.imwrite(os.path.join(self.figures_dir, f"{scan_id}_hair_removed.jpg"), hair_removed)
            
            # 2. CLAHE
            clahe_img = self._apply_clahe(hair_removed)
            cv2.imwrite(os.path.join(self.figures_dir, f"{scan_id}_clahe.jpg"), clahe_img)
            
            # 3. Grayscale & Blur
            gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # 4. Adaptive Thresholding (Detecting dark anomalies)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            
            # 5. Morphological Closing & Opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(os.path.join(self.figures_dir, f"{scan_id}_binary_mask.jpg"), opened)
            
            # 6. Connected Components & Contour Analysis
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, "No suspicious skin lesion detected. The uploaded image appears to contain healthy skin."
                
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            total_area = h * w
            
            # Draw contour mask
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [largest_contour], -1, (0, 255, 0), cv2.FILLED)
            cv2.imwrite(os.path.join(self.figures_dir, f"{scan_id}_lesion_mask.jpg"), mask)
            
            # Bounding box for visual proof
            box_img = clahe_img.copy()
            x, y, w_b, h_b = cv2.boundingRect(largest_contour)
            cv2.rectangle(box_img, (x, y), (x+w_b, y+h_b), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self.figures_dir, f"{scan_id}_largest_contour.jpg"), box_img)
            
            # If the largest distinct anomaly is less than 2% of the image area, it's likely just noise/freckles on normal skin.
            if (area / total_area) < 0.02:
                return False, "No suspicious skin lesion detected. The uploaded image appears to contain healthy skin."
                
            return True, "Lesion detected successfully."
            
        except Exception as e:
            logger.error(f"Error in Lesion Detector: {e}")
            return False, "Internal error during lesion detection."
