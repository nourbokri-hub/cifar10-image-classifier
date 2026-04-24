from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_docs_route():
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_route():
    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_predict_route_exists():
    # On envoie volontairement une requête incomplète juste pour vérifier
    # que la route existe bien.
    response = client.post("/predict")
    assert response.status_code in [400, 422]