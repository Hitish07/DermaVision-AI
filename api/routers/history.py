from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os
from ..database import get_db
from ..models import ScanHistory
from ..schemas import HistoryItem
from ..auth import get_current_user

router = APIRouter()

@router.get("/api/history", response_model=List[HistoryItem])
def get_history(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    items = db.query(ScanHistory).filter(ScanHistory.user_id == current_user).order_by(ScanHistory.created_at.desc()).all()
    out = []
    for i in items:
        out.append(HistoryItem(
            id=i.id,
            disease=i.predicted_class or "Unknown",
            confidence=i.confidence or 0.0,
            risk_level=i.risk_level or "Unknown",
            date=i.created_at.isoformat(),
            image_type="Dermoscopic" if i.is_dermoscopic else "Smartphone",
            quality_score=i.quality_score or 0.0
        ))
    return out

@router.delete("/api/history/{scan_id}")
def delete_history_item(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    item = db.query(ScanHistory).filter(ScanHistory.id == scan_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Scan not found")
    if item.user_id != current_user:
        raise HTTPException(status_code=403, detail="You can only delete your own scans")
    try:
        if os.path.exists(item.image_path):
            os.remove(item.image_path)
    except:
        pass
    db.delete(item)
    db.commit()
    return {"message": "Deleted successfully"}
