from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from .database import Base

class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(String, primary_key=True, index=True)
    patient_id = Column(String, default="UNKNOWN")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Prediction metrics
    predicted_class = Column(String)
    confidence = Column(Float)
    risk_level = Column(String)
    recommendation = Column(String)
    top_predictions = Column(String) # JSON string
    
    # Quality metrics
    quality_score = Column(Float)
    blur_score = Column(String)
    brightness = Column(String)
    noise = Column(String)
    resolution = Column(String)
    skin_ratio = Column(Float)
    
    # Image Type
    is_dermoscopic = Column(Boolean, default=False)
    is_smartphone = Column(Boolean, default=False)
    
    # System metrics
    processing_time = Column(Float)
    status = Column(String, default="completed") # pending, completed, xai_completed, xai_failed
    
    # Artifact Paths
    image_path = Column(String)
    hair_removed_path = Column(String, nullable=True)
    clahe_path = Column(String, nullable=True)
    lesion_mask_path = Column(String, nullable=True)
    
    # XAI Paths
    gradcam_path = Column(String, nullable=True)
    gradcam_plus_plus_path = Column(String, nullable=True)
    shap_path = Column(String, nullable=True)
    ig_path = Column(String, nullable=True)
    lime_path = Column(String, nullable=True)
    
    # Reporting
    pdf_path = Column(String, nullable=True)

