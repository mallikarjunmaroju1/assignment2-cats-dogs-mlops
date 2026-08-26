import io
import sys
import requests
from PIL import Image


BASE_URL = "http://localhost:8080"


def test_health():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"

    print("Health check passed")


def test_prediction():
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    files = {
        "file": ("smoke_test.jpg", image_bytes, "image/jpeg")
    }

    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    assert response.status_code == 200

    data = response.json()

    assert "predicted_class" in data
    assert data["predicted_class"] in ["cat", "dog"]
    assert "confidence" in data
    assert "probabilities" in data

    print("Prediction smoke test passed")
    print("Prediction response:")
    print(data)


if __name__ == "__main__":
    try:
        test_health()
        test_prediction()
        print("All smoke tests passed")
    except Exception as exc:
        print("Smoke test failed:", exc)
        sys.exit(1)