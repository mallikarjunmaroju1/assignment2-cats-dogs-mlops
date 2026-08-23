import numpy as np
from PIL import Image
from pathlib import Path

from src.prepare_data import preprocess_and_save_image, IMAGE_SIZE


def test_preprocess_and_save_image(tmp_path):
    # Create a dummy RGB image
    input_image_path = tmp_path / "sample.jpg"
    output_image_path = tmp_path / "processed.jpg"

    image = Image.new("RGB", (300, 300), color=(255, 0, 0))
    image.save(input_image_path)

    result = preprocess_and_save_image(input_image_path, output_image_path)

    assert result is True
    assert output_image_path.exists()

    processed_image = Image.open(output_image_path)
    assert processed_image.size == IMAGE_SIZE
    assert processed_image.mode == "RGB"