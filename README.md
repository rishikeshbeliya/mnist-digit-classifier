# Handwritten Digit Classification using Deep Learning

This project builds and compares multiple deep learning models to classify handwritten digits from the MNIST dataset. 

The project explores three architectures:
- Multilayer Perceptron (MLP)
- MLP with Dropout Regularization
- Convolutional Neural Network (CNN)

A Streamlit web application is included to allow users to upload an image and get a digit prediction.

## Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9).  
Each image is 28×28 pixels.

The dataset is split into:

- Training set
- Development (validation) set
- Test set

## Models Implemented

### 1. Multilayer Perceptron (MLP)
Fully connected neural network using dense layers.

### 2. MLP with Dropout Regularization
Dropout layers added to reduce overfitting.

### 3. Convolutional Neural Network (CNN)
Uses convolution and pooling layers for better spatial feature extraction.

## Model Performance

### Accuracy Plot
![Accuracy Plot](images/accuracy_plot.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

## Project Structure

mnist-digit-classifier
│
├── images
│   ├── accuracy_plot.png
│   └── confusion_matrix.png
│
├── models
│   └── best_cnn.keras
│
├── notebooks
│   ├── Handwritten Digit Classification using Deep Learning.ipynb
│   └── MNIST_MLP_Regularization_Comparison.ipynb
│
├── src
│   ├── app.py
│   └── handwritten.py
│
└── README.md

## Installation

Clone the repository

git clone https://github.com/rishikeshbeliya/mnist-digit-classifier.git

cd mnist-digit-classifier

Install dependencies

pip install -r requirements.txt
streamlit run src/app.py

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
