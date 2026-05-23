# Deep Learning Mini-Project File

## Basic details

- **Institute:** Institute of Business Management
- **Course title:** Deep Learning
- **Course code:** BDS 421
- **Faculty:** Dr. Kashif Laeeq
- **Project title:** DigitVision: Handwritten Digit Recognition
- **Selected idea:** Idea 2 - Handwritten Digit Recognition
- **Area:** Computer Vision
- **Tools:** Python, TensorFlow/Keras, Streamlit

## Problem statement

The goal is to build a deep learning system that can recognize handwritten
digits from images. The user provides a digit image, and the model predicts
which number from 0 to 9 is shown.

## Why this project is suitable

This project is simple to deploy and still looks impressive in a short
presentation. The MNIST dataset is included in Keras, so it is easy to access
without manual data collection. The Streamlit app gives an interactive demo
where users can upload digit images and view prediction confidence.

## Dataset

- **Dataset name:** MNIST
- **Data type:** Grayscale images of handwritten digits
- **Image size:** 28 x 28 pixels
- **Classes:** 10 classes, digits 0 to 9
- **Size:** 60,000 training images and 10,000 test images

## Deep learning concepts used

- Image preprocessing
- Convolutional Neural Networks
- Pooling layers
- Dropout regularization
- Softmax multi-class classification
- Training, validation, and testing

## Methodology

1. Load the MNIST dataset using TensorFlow/Keras.
2. Normalize pixel values from 0-255 to 0-1.
3. Train a CNN model on the training images.
4. Evaluate the model on the test set.
5. Save the trained model and metrics.
6. Deploy the model in a Streamlit app.
7. Allow users to upload images and view predictions.

## Model architecture

```text
Input 28x28 grayscale image
Conv2D with ReLU
MaxPooling2D
Conv2D with ReLU
MaxPooling2D
Dropout
Flatten
Dense with ReLU
Dropout
Dense with Softmax for digits 0-9
```

## Expected output

The app displays:

- The uploaded digit image
- The preprocessed 28 x 28 model input
- The predicted digit
- The confidence score
- A confidence chart for all 10 digit classes

## Future improvements

- Add a drawing canvas so users can write digits directly in the app.
- Train with more handwriting styles for better real-world accuracy.
- Show a confusion matrix and class-wise accuracy.
- Add deployment screenshots to the final report.
