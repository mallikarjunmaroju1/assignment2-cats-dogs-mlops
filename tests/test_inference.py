from pathlib import Path

import numpy as np
from PIL import Image

from api.main import preprocess_image, IMAGE_SIZE, MODEL_PATH


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_preprocess_image_for_api():
    # Create dummy image bytes
    image = Image.new("RGB", (300, 300), color=(0, 255, 0))

    import io
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="JPEG")
    image_bytes = image_bytes_io.getvalue()

    processed = preprocess_image(image_bytes)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)