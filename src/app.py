import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("../models/best_cnn.keras")

st.title("Handwritten Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    # convert to numpy
    img = np.array(img)

    # invert colors
    img = 255 - img

    # remove empty borders
    coords = np.argwhere(img > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0 : y1 + 1, x0 : x1 + 1]

    # resize while keeping aspect ratio
    img = Image.fromarray(img)
    img = img.resize((20, 20))

    img = np.array(img)

    # place digit in center of 28x28 canvas
    canvas = np.zeros((28, 28))
    canvas[4:24, 4:24] = img

    # normalize
    canvas = canvas.astype("float32") / 255.0

    # reshape for model
    canvas = canvas.reshape(1, 28, 28, 1)

    st.image(canvas.reshape(28, 28), caption="Processed Image", width=150)

    prediction = model.predict(canvas, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Digit: {digit}")
    st.write("Confidence:", float(confidence))
