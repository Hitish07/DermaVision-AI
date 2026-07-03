import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_upload():
    # Mocking upload endpoint
    response = client.post("/api/upload", files={"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")})
    assert response.status_code == 200
    assert "scan_id" in response.json()

def test_api_history():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_validate_invalid_id():
    response = client.post("/api/validate/invalid-id")
    assert response.status_code == 404

def test_quality_invalid_id():
    response = client.post("/api/quality/invalid-id")
    assert response.status_code == 404

def test_detect_invalid_id():
    response = client.post("/api/detect/invalid-id")
    assert response.status_code == 404

def test_predict_invalid_id():
    response = client.post("/api/predict/invalid-id")
    assert response.status_code == 404
