import os
import random
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

IMAGE_SIZE = (224, 224)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# For faster assignment execution, you can keep 1000 or 2000.
# Set to None if you want to use all images.
MAX_IMAGES_PER_CLASS = 1000

CLASSES = {
    "cat": ["Cat", "cat", "cats", "PetImages/Cat"],
    "dog": ["Dog", "dog", "dogs", "PetImages/Dog"]
}


def find_class_folder(class_variants):
    for folder in class_variants:
        path = RAW_DIR / folder
        if path.exists() and path.is_dir():
            return path
    return None


def clean_processed_dir():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES.keys():
            folder = PROCESSED_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def is_image_file(file_path):
    return file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]


def preprocess_and_save_image(src_path, dest_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(dest_path, format="JPEG", quality=95)
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def split_files(files):
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def process_class(class_name, source_folder):
    image_files = [f for f in source_folder.iterdir() if f.is_file() and is_image_file(f)]

    if MAX_IMAGES_PER_CLASS is not None:
        image_files = image_files[:MAX_IMAGES_PER_CLASS]

    train_files, val_files, test_files = split_files(image_files)

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    success_count = 0
    failed_count = 0

    for split_name, files in split_map.items():
        for idx, file_path in enumerate(files):
            output_name = f"{class_name}_{idx}.jpg"
            output_path = PROCESSED_DIR / split_name / class_name / output_name

            if preprocess_and_save_image(file_path, output_path):
                success_count += 1
            else:
                failed_count += 1

    print(f"{class_name.upper()} completed")
    print(f"Successful images: {success_count}")
    print(f"Failed/corrupt images skipped: {failed_count}")


def print_summary():
    print("\nProcessed Dataset Summary")
    print("-" * 40)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES.keys():
            folder = PROCESSED_DIR / split / class_name
            count = len(list(folder.glob("*.jpg")))
            print(f"{split}/{class_name}: {count}")


def main():
    random.seed(42)

    print("Raw data folder:", RAW_DIR)
    print("Processed data folder:", PROCESSED_DIR)

    clean_processed_dir()

    for class_name, variants in CLASSES.items():
        source_folder = find_class_folder(variants)

        if source_folder is None:
            raise FileNotFoundError(
                f"Could not find folder for class '{class_name}'. "
                f"Expected one of these under data/raw: {variants}"
            )

        print(f"\nProcessing {class_name} images from: {source_folder}")
        process_class(class_name, source_folder)

    print_summary()


if __name__ == "__main__":
    main()