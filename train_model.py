"""Train a compact CNN for MNIST handwritten digit recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_PATH = Path("models/mnist_cnn.keras")
METRICS_PATH = Path("models/metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional number of training examples for quick experiments.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional number of test examples for quick experiments.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_mnist(
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if limit_train:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test:
        x_test = x_test[:limit_test]
        y_test = y_test[:limit_test]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Keras Conv2D layers expect a channel dimension: 28 x 28 x 1.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_digit_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist(
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = build_model()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    metrics = {
        "model": str(MODEL_PATH),
        "dataset": "MNIST handwritten digits",
        "training_examples": int(x_train.shape[0]),
        "test_examples": int(x_test.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
