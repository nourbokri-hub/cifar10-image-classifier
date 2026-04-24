import io
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def test_predict_valid_image():
    img_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)
    assert "prediction" in data or "predicted_class" in data