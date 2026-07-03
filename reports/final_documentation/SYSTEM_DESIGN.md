# System Design Document

## Data Flow Diagram
1. User Uploads Image -> Nginx Reverse Proxy -> FastAPI `/api/scan`
2. FastAPI validates image (`validation.py`). If fail -> 400 Bad Request.
3. FastAPI evaluates quality (`quality.py`).
4. FastAPI invokes `inference.py` -> `predict_single_image()`.
5. Sync XAI generates Grad-CAM.
6. DB creates `ScanHistory` record. Status: `pending`.
7. `BackgroundTasks` executes `get_integrated_gradients()` and `generate_shap()`.
8. React polls `/api/explainability/{id}` -> Updates DOM.

## Database Schema (SQLite)
`ScanHistory`
- id (PK)
- image_path
- predicted_class
- confidence
- risk_level
- processing_time
- timestamp
- async_xai_status
