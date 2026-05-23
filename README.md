# DigitVision: Handwritten Digit Recognition

DigitVision is a simple but presentation-ready deep learning mini-project for a
course demo. It uses a Convolutional Neural Network (CNN) to recognize
handwritten digits from 0 to 9 and serves the model through a Streamlit web app.

## Why this project was selected

From the provided idea list, **Handwritten Digit Recognition** is one of the
best choices for a quick, impressive Streamlit deployment:

- **Easy dataset:** MNIST is built into TensorFlow/Keras, so no Kaggle setup is
  required.
- **Clear deep learning concept:** CNNs are easy to explain visually with image
  preprocessing, convolution, pooling, and classification.
- **Fast to train:** A compact CNN can reach high accuracy in a few epochs.
- **Strong demo value:** Users can upload a digit image or try built-in test
  samples and instantly see prediction confidence.

## Project structure

```text
.
+-- app.py              # Streamlit web app
+-- train_model.py      # CNN training script
+-- requirements.txt    # Python dependencies
+-- requirements-train.txt # Optional TensorFlow training dependency
+-- PROJECT_FILE.md     # Course submission summary
+-- data/               # Exported sample digit images for the app
+-- models/             # Generated trained model and metrics
```

## How to run locally

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

The app uses the committed NumPy model weights in
`models/mnist_cnn_weights.npz`, so Streamlit deployment does not need
TensorFlow.

## Optional: retrain the CNN

Training requires TensorFlow, which is intentionally kept out of
`requirements.txt` so Streamlit Cloud can deploy on newer Python versions.

```bash
pip install -r requirements-train.txt
python train_model.py --epochs 3
```

## Streamlit deployment notes

For Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Select `app.py` as the entry point.
4. Make sure `requirements.txt` is included.

Important: `requirements.txt` does not include TensorFlow because Streamlit
Cloud may run Python versions that TensorFlow does not support yet. The app
loads exported CNN weights and runs inference with NumPy, which keeps deployment
lightweight and reliable.

## Model summary

- Dataset: MNIST handwritten digit dataset
- Input shape: 28 x 28 grayscale image
- Output classes: 10 digits, from 0 to 9
- Architecture: Conv2D, MaxPooling2D, Dropout, Dense, Softmax
- Loss function: Sparse categorical cross-entropy
- Optimizer: Adam

## Presentation talking points

- Problem: Manual digit reading is a classic image classification problem.
- Dataset: MNIST contains thousands of labeled handwritten digits.
- Method: A CNN learns visual patterns such as strokes and curves.
- Demo: Upload a digit image and show the predicted digit plus confidence chart.
- Future improvements: Add a drawing canvas, support more handwriting styles,
  and train on custom classroom examples.
