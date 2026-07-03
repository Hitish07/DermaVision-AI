from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionOutput(BaseModel):
    class_name: str
    confidence: float

class ValidationRequest(BaseModel):
    image_path: str

class ValidationResponse(BaseModel):
    is_valid: bool
    reason: str
    image_type: str
    skin_percentage: Optional[float] = None

class QualityResponse(BaseModel):
    quality_score: float
    quality_grade: str
    blur_score: str
    brightness: str
    contrast: str
    noise: str
    resolution: str

class DetectionResponse(BaseModel):
    lesion_detected: bool
    message: str
    artifacts: Dict[str, str]

class PredictResponse(BaseModel):
    disease: str
    confidence: float
    risk_level: str
    recommendation: str
    top_3: List[PredictionOutput]

class ScanResultResponse(BaseModel):
    status: str
    scan_id: str
    validation: ValidationResponse
    quality: QualityResponse
    prediction: Optional[PredictResponse] = None
    artifacts: Dict[str, str]

class AsyncXAIResponse(BaseModel):
    status: str
    shap_path: Optional[str]
    ig_path: Optional[str]
    lime_path: Optional[str]
    gradcam_plus_plus_path: Optional[str]

class HistoryItem(BaseModel):
    id: str
    disease: str
    confidence: float
    risk_level: str
    date: str
    image_type: str
    quality_score: float
    
    class Config:
        from_attributes = True

class FullScanDetails(HistoryItem):
    top_predictions: Any
    recommendation: str
    blur_score: str
    brightness: str
    noise: str
    resolution: str
    skin_ratio: float
    is_dermoscopic: bool
    is_smartphone: bool
    processing_time: float
    image_path: str
    hair_removed_path: Optional[str]
    clahe_path: Optional[str]
    lesion_mask_path: Optional[str]
    gradcam_path: Optional[str]
    gradcam_plus_plus_path: Optional[str]
    shap_path: Optional[str]
    ig_path: Optional[str]
    lime_path: Optional[str]
    pdf_path: Optional[str]

