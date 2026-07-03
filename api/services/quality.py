import cv2
import numpy as np

def assess_image_quality(image_path: str) -> dict:
    """
    Evaluates image quality using OpenCV heuristics.
    Returns scores and an overall string rating (Excellent, Good, Acceptable, Poor).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image for quality assessment.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Detection (Variance of Laplacian)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Brightness
    brightness = np.mean(gray)
    
    # 3. Contrast (Standard deviation of pixel intensities)
    contrast = np.std(gray)
    
    # Determine overall quality
    # Thresholds are heuristic-based
    issues = []
    if blur_score < 100:
        issues.append("Image is too blurry")
    if brightness < 40:
        issues.append("Image is underexposed (too dark)")
    elif brightness > 220:
        issues.append("Image is overexposed (too bright)")
    if contrast < 30:
        issues.append("Low contrast")
        
    if len(issues) == 0:
        if blur_score > 300 and 80 < brightness < 180 and contrast > 50:
            overall = "Excellent"
        else:
            overall = "Good"
    elif len(issues) == 1:
        overall = "Acceptable"
    else:
        overall = "Poor"
        
    return {
        "overall": overall,
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "issues": issues
    }
