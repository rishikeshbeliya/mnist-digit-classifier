import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("models/best_cnn.keras")
st.title("Handwritten Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))

    st.image(img, caption="Uploaded Image", width=150)

    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
