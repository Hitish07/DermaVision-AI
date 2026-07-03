from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)

    predicted_class = Column(String)
    confidence = Column(Float)
    risk_level = Column(String)
    recommendation = Column(String)
    top_predictions = Column(String)  # JSON string

    quality_score = Column(Float)
    blur_score = Column(String)
    brightness = Column(String)
    noise = Column(String)
    resolution = Column(String)
    skin_ratio = Column(Float)

    is_dermoscopic = Column(Boolean, default=False)
    is_smartphone = Column(Boolean, default=False)

    processing_time = Column(Float)
    status = Column(String, default="completed")

    image_path = Column(String)
    hair_removed_path = Column(String, nullable=True)
    clahe_path = Column(String, nullable=True)
    lesion_mask_path = Column(String, nullable=True)

    gradcam_path = Column(String, nullable=True)
    gradcam_plus_plus_path = Column(String, nullable=True)
    shap_path = Column(String, nullable=True)
    ig_path = Column(String, nullable=True)
    lime_path = Column(String, nullable=True)

    pdf_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
