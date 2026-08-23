import csv
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8080"
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "processed" / "test"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "post_deployment_predictions.csv"


def predict_image(image_path, true_label):
    start_time = time.time()

    with open(image_path, "rb") as image_file:
        files = {
            "file": (image_path.name, image_file, "image/jpeg")
        }
        response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)

    latency = round(time.time() - start_time, 4)

    if response.status_code != 200:
        return {
            "filename": image_path.name,
            "true_label": true_label,
            "predicted_class": "ERROR",
            "confidence": 0,
            "latency_seconds": latency,
            "status_code": response.status_code,
            "correct": False
        }

    result = response.json()
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    return {
        "filename": image_path.name,
        "true_label": true_label,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "latency_seconds": latency,
        "status_code": response.status_code,
        "correct": predicted_class == true_label
    }


def main():
    records = []

    cat_images = list((TEST_DIR / "cat").glob("*.jpg"))[:5]
    dog_images = list((TEST_DIR / "dog").glob("*.jpg"))[:5]

    for image_path in cat_images:
        records.append(predict_image(image_path, "cat"))

    for image_path in dog_images:
        records.append(predict_image(image_path, "dog"))

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        fieldnames = [
            "filename",
            "true_label",
            "predicted_class",
            "confidence",
            "latency_seconds",
            "status_code",
            "correct"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(r["latency_seconds"] for r in records) / total if total > 0 else 0

    print("Post-deployment tracking completed")
    print(f"Total requests: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Simulated accuracy: {accuracy:.4f}")
    print(f"Average latency seconds: {avg_latency:.4f}")
    print(f"Results saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()