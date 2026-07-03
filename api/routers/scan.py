from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import uuid
import os
import shutil
from pathlib import Path

from ..database import get_db
from ..models import ScanHistory
from ..schemas import ValidationResponse, QualityResponse, DetectionResponse, PredictResponse, AsyncXAIResponse, FullScanDetails
from ..auth import get_current_user

from ..services.validation import ImageValidator
from ..services.quality_assessment import assess_image_quality
from ..services.lesion_detector import LesionDetector
from ..services.inference import run_prediction_sync, run_async_xai
from ..services.recommendation import get_clinical_recommendation
from ..services.report_generator import generate_pdf_report

router = APIRouter()

UPLOAD_DIR = Path("api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

validator = ImageValidator()
detector = LesionDetector()

@router.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    scan_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1]
    image_path = UPLOAD_DIR / f"{scan_id}.{file_extension}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_item = ScanHistory(
        id=scan_id,
        user_id=current_user,
        image_path=str(image_path),
        status="uploaded"
    )
    db.add(db_item)
    db.commit()

    return {"scan_id": scan_id, "image_path": str(image_path)}

@router.post("/api/validate/{scan_id}", response_model=ValidationResponse)
def validate_scan(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only validate your own scans")

    val_res = validator.validate(item.image_path)

    item.is_dermoscopic = (val_res["image_type"] == "Dermoscopic")
    item.is_smartphone = (val_res["image_type"] == "Smartphone")
    db.commit()

    return ValidationResponse(
        is_valid=val_res["is_valid"],
        reason=val_res["reason"],
        image_type=val_res["image_type"],
        skin_percentage=val_res.get("skin_percentage", 100.0)
    )

@router.post("/api/quality/{scan_id}", response_model=QualityResponse)
def quality_scan(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only assess quality of your own scans")

    q_res = assess_image_quality(item.image_path)

    item.quality_score = q_res.get("quality_score", 0)
    item.blur_score = q_res.get("blur_score", "Unknown")
    item.brightness = q_res.get("brightness", "Unknown")
    item.contrast = q_res.get("contrast", "Unknown")
    item.noise = q_res.get("noise", "Unknown")
    item.resolution = q_res.get("resolution", "Unknown")
    db.commit()

    return QualityResponse(**q_res)

@router.post("/api/detect/{scan_id}", response_model=DetectionResponse)
def detect_lesion(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only detect lesions in your own scans")

    has_lesion, msg = detector.detect(item.image_path, scan_id)

    item.hair_removed_path = f"figures/{scan_id}_hair_removed.jpg"
    item.clahe_path = f"figures/{scan_id}_clahe.jpg"
    item.lesion_mask_path = f"figures/{scan_id}_lesion_mask.jpg"
    db.commit()

    return DetectionResponse(
        lesion_detected=has_lesion,
        message=msg,
        artifacts={
            "hair_removed": f"/figures/{scan_id}_hair_removed.jpg",
            "clahe": f"/figures/{scan_id}_clahe.jpg",
            "lesion_mask": f"/figures/{scan_id}_lesion_mask.jpg"
        }
    )

@router.post("/api/predict/{scan_id}", response_model=PredictResponse)
def predict_scan(
    scan_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only predict on your own scans")

    results, proc_time, gradcam_path = run_prediction_sync(item.image_path)
    top_prediction = results[0]

    disease = top_prediction["class_name"]
    risk, rec = get_clinical_recommendation(disease)

    item.predicted_class = disease
    item.confidence = top_prediction["confidence"]
    item.risk_level = risk
    item.recommendation = rec
    item.processing_time = proc_time
    import json
    item.top_predictions = json.dumps(results)
    item.gradcam_path = f"figures/MobileNetV2_gradcam_{os.path.basename(item.image_path)}"
    item.status = "completed"

    db.commit()

    background_tasks.add_task(run_async_xai, scan_id, item.image_path)

    from ..schemas import PredictionOutput
    return PredictResponse(
        disease=disease,
        confidence=top_prediction["confidence"],
        risk_level=risk,
        recommendation=rec,
        top_3=[PredictionOutput(**r) for r in results]
    )

@router.get("/api/scan/{scan_id}/status", response_model=AsyncXAIResponse)
def get_scan_status(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only view status of your own scans")

    return AsyncXAIResponse(
        status=item.status,
        shap_path=f"/{item.shap_path}" if item.shap_path else None,
        ig_path=f"/{item.ig_path}" if item.ig_path else None,
        lime_path=f"/{item.lime_path}" if item.lime_path else None,
        gradcam_plus_plus_path=f"/{item.gradcam_plus_plus_path}" if item.gradcam_plus_plus_path else None
    )

@router.get("/api/scan/{scan_id}", response_model=FullScanDetails)
def get_scan_details(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only view details of your own scans")

    import json
    return FullScanDetails(
        id=item.id,
        disease=item.predicted_class or "Unknown",
        confidence=item.confidence or 0.0,
        risk_level=item.risk_level or "Unknown",
        date=item.created_at.isoformat(),
        image_type="Dermoscopic" if item.is_dermoscopic else "Smartphone" if item.is_smartphone else "Unknown",
        quality_score=item.quality_score or 0.0,
        top_predictions=json.loads(item.top_predictions) if item.top_predictions else [],
        recommendation=item.recommendation or "",
        blur_score=item.blur_score or "",
        brightness=item.brightness or "",
        noise=item.noise or "",
        resolution=item.resolution or "",
        skin_ratio=item.skin_ratio or 100.0,
        is_dermoscopic=item.is_dermoscopic,
        is_smartphone=item.is_smartphone,
        processing_time=item.processing_time or 0.0,
        image_path=f"/uploads/{os.path.basename(item.image_path)}",
        hair_removed_path=f"/{item.hair_removed_path}" if item.hair_removed_path else None,
        clahe_path=f"/{item.clahe_path}" if item.clahe_path else None,
        lesion_mask_path=f"/{item.lesion_mask_path}" if item.lesion_mask_path else None,
        gradcam_path=f"/{item.gradcam_path}" if item.gradcam_path else None,
        gradcam_plus_plus_path=f"/{item.gradcam_plus_plus_path}" if item.gradcam_plus_plus_path else None,
        shap_path=f"/{item.shap_path}" if item.shap_path else None,
        ig_path=f"/{item.ig_path}" if item.ig_path else None,
        lime_path=f"/{item.lime_path}" if item.lime_path else None,
        pdf_path=f"/api/scan/{item.id}/report"
    )

@router.get("/api/scan/{scan_id}/report")
def get_pdf_report_endpoint(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only download reports for your own scans")

    import json
    scan_data = {
        "scan_id": item.id,
        "disease": item.predicted_class or "Unknown",
        "confidence": item.confidence or 0.0,
        "risk": item.risk_level or "Unknown",
        "recommendation": item.recommendation or "",
        "quality_score": item.quality_score or 0.0,
        "blur_score": item.blur_score or "Unknown",
        "brightness": item.brightness or "Unknown",
        "noise": item.noise or "Unknown",
        "top_3": json.loads(item.top_predictions) if item.top_predictions else [],
        "original_image": item.image_path,
        "gradcam_image": item.gradcam_path if item.gradcam_path else ""
    }

    try:
        pdf_path = generate_pdf_report(scan_data)
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"DermaVision_Report_{scan_id}.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
