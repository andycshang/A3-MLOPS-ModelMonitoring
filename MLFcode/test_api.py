from fastapi.testclient import TestClient
from A3.MLFcode.app.api import app

client = TestClient(app)

# correct input: 200 + prediction
def test_predict_happy_path():
    payload = {
        "CRIM": 0.1,
        "ZN": 12.5,
        "INDUS": 7.5,
        "CHAS": 0,
        "NOX": 0.45,
        "RM": 6.2,
        "AGE": 65.2,
        "DIS": 4.1,
        "RAD": 5,
        "TAX": 300.0,
        "PTRATIO": 18.5,
        "B": 390.0,
        "LSTAT": 5.2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


# missing field: 422（Unprocessable Entity）
def test_predict_missing_field():
    payload = {
        "CRIM": 0.1,
        "ZN": 12.5

    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # FastAPI 自动触发 Pydantic 验证错误


# invalid type
def test_predict_invalid_type():
    payload = {
        "CRIM": "wrong_type",
        "ZN": 12.5,
        "INDUS": 7.5,
        "CHAS": 0,
        "NOX": 0.45,
        "RM": 6.2,
        "AGE": 65.2,
        "DIS": 4.1,
        "RAD": 5,
        "TAX": 300.0,
        "PTRATIO": 18.5,
        "B": 390.0,
        "LSTAT": 5.2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
