"""Streamlit app for handwritten digit recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


MODEL_PATH = Path("models/mnist_cnn.keras")
METRICS_PATH = Path("models/metrics.json")
DIGIT_LABELS = list(range(10))


st.set_page_config(
    page_title="DigitVision - MNIST CNN",
    page_icon="123",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_model():
    if not MODEL_PATH.exists():
        return None

    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading MNIST demo images...")
def load_mnist_examples() -> dict[int, Image.Image]:
    from tensorflow import keras

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    examples: dict[int, Image.Image] = {}
    for image, label in zip(x_test, y_test, strict=False):
        label = int(label)
        if label not in examples:
            examples[label] = Image.fromarray(image)
        if len(examples) == 10:
            break
    return examples


def read_metrics() -> dict[str, float | int | str]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convert a user image into the same 28x28 format used by MNIST."""
    grayscale = image.convert("L")
    grayscale = ImageOps.autocontrast(grayscale)

    # Most uploaded handwritten digits are dark ink on white paper. MNIST is the
    # opposite, so invert bright-background images before resizing.
    if np.asarray(grayscale).mean() > 127:
        grayscale = ImageOps.invert(grayscale)

    resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(resized).astype("float32") / 255.0
    batch = array.reshape(1, 28, 28, 1)
    return batch, resized


def predict_digit(model, image: Image.Image) -> tuple[int, float, pd.DataFrame, Image.Image]:
    batch, processed_image = prepare_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    chart_data = pd.DataFrame(
        {
            "Digit": DIGIT_LABELS,
            "Confidence": probabilities,
        }
    )
    return prediction, confidence, chart_data, processed_image


def show_project_intro() -> None:
    st.title("DigitVision: Handwritten Digit Recognition")
    st.caption("Deep Learning Mini-Project | CNN + MNIST + Streamlit")

    st.markdown(
        """
        This app predicts handwritten digits from **0 to 9** using a compact
        Convolutional Neural Network trained on the MNIST dataset. It is a good
        course project because the dataset is built into Keras, the model trains
        quickly, and the Streamlit interface makes the demo visual and simple.
        """
    )


def show_sidebar(metrics: dict[str, float | int | str]) -> None:
    with st.sidebar:
        st.header("Project Snapshot")
        st.write("**Selected idea:** Handwritten Digit Recognition")
        st.write("**Dataset:** MNIST, 70,000 labeled digit images")
        st.write("**Model:** 2-layer CNN classifier")
        st.write("**Classes:** Digits 0-9")

        if metrics:
            st.divider()
            st.subheader("Saved Model Metrics")
            accuracy = float(metrics.get("test_accuracy", 0.0))
            validation_accuracy = float(metrics.get("final_validation_accuracy", 0.0))
            st.metric("Test accuracy", f"{accuracy:.2%}")
            st.metric("Validation accuracy", f"{validation_accuracy:.2%}")
            st.write(f"Training examples: {metrics.get('training_examples', 'N/A')}")
            st.write(f"Epochs: {metrics.get('epochs', 'N/A')}")

        st.divider()
        st.subheader("How to retrain")
        st.code("python train_model.py --epochs 3", language="bash")


def main() -> None:
    show_project_intro()
    metrics = read_metrics()
    show_sidebar(metrics)

    model = load_model()
    if model is None:
        st.error(
            "No trained model was found. Run `python train_model.py --epochs 3` "
            "to create `models/mnist_cnn.keras`, then restart Streamlit."
        )
        st.stop()

    upload_tab, sample_tab, explain_tab = st.tabs(
        ["Upload a digit", "Try MNIST samples", "How it works"]
    )

    with upload_tab:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Upload a handwritten digit")
            uploaded_file = st.file_uploader(
                "Choose a PNG, JPG, or JPEG image",
                type=["png", "jpg", "jpeg"],
            )
            st.info(
                "Tip: write one large digit on a plain background. The app "
                "automatically converts it to MNIST-style grayscale."
            )

        if uploaded_file:
            image = Image.open(uploaded_file)
            prediction, confidence, chart_data, processed_image = predict_digit(model, image)
            with left:
                st.image(image, caption="Original upload", use_container_width=True)
            with right:
                st.subheader("Prediction")
                st.metric("Predicted digit", prediction)
                st.metric("Confidence", f"{confidence:.2%}")
                st.image(
                    processed_image.resize((180, 180), Image.Resampling.NEAREST),
                    caption="Model input after preprocessing",
                    width=180,
                )
                st.bar_chart(chart_data, x="Digit", y="Confidence", height=260)
        else:
            with right:
                st.subheader("Prediction")
                st.write("Upload an image to see the CNN prediction.")

    with sample_tab:
        st.subheader("Demo with built-in MNIST test images")
        examples = load_mnist_examples()
        selected_digit = st.selectbox("Pick a sample digit", DIGIT_LABELS, index=5)
        sample_image = examples[selected_digit]
        prediction, confidence, chart_data, processed_image = predict_digit(model, sample_image)

        left, right = st.columns([1, 2])
        with left:
            st.image(
                processed_image.resize((220, 220), Image.Resampling.NEAREST),
                caption=f"MNIST sample label: {selected_digit}",
                width=220,
            )
        with right:
            st.metric("Predicted digit", prediction)
            st.metric("Confidence", f"{confidence:.2%}")
            st.bar_chart(chart_data, x="Digit", y="Confidence", height=300)

    with explain_tab:
        st.subheader("Deep learning pipeline")
        st.markdown(
            """
            1. **Preprocess images:** convert to grayscale, invert if needed,
               resize to 28x28 pixels, and normalize pixel values to 0-1.
            2. **Extract visual features:** convolution layers learn strokes,
               curves, corners, and digit shapes.
            3. **Classify:** dense layers map learned features to one of ten
               digit classes.
            4. **Deploy:** Streamlit provides a simple web interface for upload,
               prediction, and confidence visualization.
            """
        )
        st.code(
            """
Conv2D(32) -> MaxPooling2D
Conv2D(64) -> MaxPooling2D
Dropout -> Flatten -> Dense(128) -> Dense(10, softmax)
            """.strip(),
            language="text",
        )


if __name__ == "__main__":
    main()
