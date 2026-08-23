import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "processed" / "train"
VAL_DIR = BASE_DIR / "data" / "processed" / "val"
TEST_DIR = BASE_DIR / "data" / "processed" / "test"

MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        data_augmentation,

        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_curves(history):
    loss_curve_path = ARTIFACT_DIR / "training_loss_curve.png"
    accuracy_curve_path = ARTIFACT_DIR / "training_accuracy_curve.png"

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_curve_path)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(accuracy_curve_path)
    plt.close()

    return loss_curve_path, accuracy_curve_path


def evaluate_model(model, test_ds):
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images)
        y_prob.extend(probs.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1_score": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_prob)
    }

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat", "dog"])

    confusion_matrix_path = ARTIFACT_DIR / "confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    disp.plot()
    plt.title("Confusion Matrix - Cats vs Dogs CNN")
    plt.savefig(confusion_matrix_path)
    plt.close()

    return metrics, confusion_matrix_path


def main():
    mlflow.set_experiment("Cats vs Dogs MLOps Assignment 2")

    train_ds, val_ds, test_ds, class_names = load_datasets()

    model = build_model()
    model.summary()

    with mlflow.start_run(run_name="baseline_cnn"):
        mlflow.log_param("model_type", "Simple CNN")
        mlflow.log_param("image_size", IMAGE_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("classes", class_names)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )

        loss_curve_path, accuracy_curve_path = plot_training_curves(history)

        metrics, confusion_matrix_path = evaluate_model(model, test_ds)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(loss_curve_path))
        mlflow.log_artifact(str(accuracy_curve_path))
        mlflow.log_artifact(str(confusion_matrix_path))

        model_path = MODEL_DIR / "cats_dogs_model.keras"
        model.save(model_path)

        mlflow.keras.log_model(model, "cats_dogs_model")

        print("\nTest Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print(f"\nModel saved at: {model_path}")


if __name__ == "__main__":
    main()