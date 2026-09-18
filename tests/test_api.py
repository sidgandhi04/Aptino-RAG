from fastapi.testclient import TestClient
from api.main import app
import json
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "workflow_initialized": True}

def test_analyze_validation_error():
    # Sending empty request
    response = client.post("/analyze", json={})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "errors" in data
