import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app, MAX_UPLOAD_BYTES

client = TestClient(app)


def create_test_image(image_format="PNG"):
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image_format)
    img_bytes.seek(0)
    return img_bytes


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_predict_valid_image(image_format):
    img_bytes = create_test_image(image_format)

    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data) == {"prediction", "confidence", "top3"}
    assert data["prediction"] == data["top3"][0]["class"]
    assert data["confidence"] == data["top3"][0]["confidence"]
    assert len(data["top3"]) == 3
    confidences = [item["confidence"] for item in data["top3"]]
    assert all(0.0 <= confidence <= 1.0 for confidence in confidences)
    assert confidences == sorted(confidences, reverse=True)


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict",
        files={"file": ("invalid.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or corrupted image file"


def test_predict_rejects_unsupported_image_format():
    image_bytes = create_test_image("BMP")
    response = client.post(
        "/predict",
        files={"file": ("test.bmp", image_bytes, "image/bmp")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == (
        "Unsupported image format. Upload a PNG or JPEG image."
    )


def test_predict_rejects_oversized_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "oversized.png",
                b"x" * (MAX_UPLOAD_BYTES + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Image exceeds the maximum size of 10 MiB"
    )
