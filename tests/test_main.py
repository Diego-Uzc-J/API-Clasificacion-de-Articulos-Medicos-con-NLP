from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_structure():
    payload = {
        "titulo": "Estudio cardiovascular avanzado",
        "resumen": "Evaluación de la insuficiencia cardíaca y tratamientos."
    }
    response = client.post("/predict", json=payload)
    # 200 si el modelo está presente, 503 si el contenedor no tiene los binarios del modelo ONNX en CI
    assert response.status_code in [200, 503]
