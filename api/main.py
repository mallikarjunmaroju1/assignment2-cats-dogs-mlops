import io
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cats_dogs_model.keras"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Cats vs Dogs Image Classification API",
    description="MLOps Assignment 2 API for binary image classification",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {str(exc)}")


@app.get("/")
def home():
    return {
        "message": "Cats vs Dogs Prediction API is running",
        "model": "baseline_cnn",
        "classes": CLASS_NAMES
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    probability_dog = float(model.predict(processed_image)[0][0])
    probability_cat = 1.0 - probability_dog

    predicted_class = "dog" if probability_dog >= 0.5 else "cat"
    confidence = probability_dog if predicted_class == "dog" else probability_cat

    response_time = round(time.time() - start_time, 4)

    result = {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(probability_cat, 4),
            "dog": round(probability_dog, 4)
        },
        "response_time_seconds": response_time
    }

    logging.info(
        f"Prediction request | file={file.filename} | "
        f"predicted_class={predicted_class} | confidence={round(confidence, 4)} | "
        f"response_time={response_time}s"
    )

    return result