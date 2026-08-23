from fastapi.testclient import TestClient
from PIL import Image
import io

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))

    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="JPEG")
    image_bytes_io.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": ("test.jpg", image_bytes_io, "image/jpeg")
        }
    )

    assert response.status_code == 200

    result = response.json()
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["predicted_class"] in ["cat", "dog"]