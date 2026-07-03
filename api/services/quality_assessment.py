import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def assess_image_quality(image_path: str) -> dict:
    """
    Stage 2: Comprehensive Image Quality Assessment.
    Returns scores and grades for blur, brightness, contrast, noise, and resolution.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"quality_score": 0, "quality_grade": "Poor", "reason": "Unreadable image"}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Resolution
        resolution = f"{w}x{h}"
        res_score = min(100, (w * h) / (224 * 224) * 100) # At least 224x224 is required

        # 2. Blur (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = "Excellent" if laplacian_var > 300 else "Good" if laplacian_var > 100 else "Acceptable" if laplacian_var > 50 else "Poor"
        blur_pts = 100 if blur_score == "Excellent" else 80 if blur_score == "Good" else 50 if blur_score == "Acceptable" else 0

        # 3. Brightness
        mean_brightness = np.mean(gray)
        if 80 < mean_brightness < 180:
            bright_score = "Excellent"
            bright_pts = 100
        elif 40 <= mean_brightness <= 220:
            bright_score = "Good"
            bright_pts = 80
        else:
            bright_score = "Poor"
            bright_pts = 20

        # 4. Contrast (RMS Contrast)
        rms_contrast = np.std(gray)
        if 40 < rms_contrast < 80:
            contrast_score = "Excellent"
            contrast_pts = 100
        elif 20 <= rms_contrast <= 100:
            contrast_score = "Good"
            contrast_pts = 80
        else:
            contrast_score = "Poor"
            contrast_pts = 30

        # 5. Noise (High-pass filter standard deviation)
        # Using a simple 3x3 blur and subtracting from original to isolate high frequency noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        high_pass = cv2.subtract(gray, blurred)
        noise_std = np.std(high_pass)
        
        if noise_std < 5:
            noise_score = "Low"
            noise_pts = 100
        elif noise_std < 15:
            noise_score = "Medium"
            noise_pts = 70
        else:
            noise_score = "High"
            noise_pts = 30

        # Overall Quality Score
        total_score = (res_score * 0.1) + (blur_pts * 0.4) + (bright_pts * 0.2) + (contrast_pts * 0.2) + (noise_pts * 0.1)
        total_score = min(100, max(0, total_score))

        if total_score >= 90:
            quality_grade = "Excellent"
        elif total_score >= 70:
            quality_grade = "Good"
        elif total_score >= 40:
            quality_grade = "Acceptable"
        else:
            quality_grade = "Poor"

        return {
            "quality_score": round(total_score),
            "quality_grade": quality_grade,
            "blur_score": blur_score,
            "brightness": bright_score,
            "contrast": contrast_score,
            "noise": noise_score,
            "resolution": "Excellent" if res_score > 90 else "Good"
        }

    except Exception as e:
        logger.error(f"Error in Quality Assessment: {e}")
        return {"quality_score": 0, "quality_grade": "Error", "reason": str(e)}
