
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("apple_leaf_disease_model.keras")  # replace with your model path

# Define class names (adjust according to your training data)
class_names = ['Apple Scab', 'Apple Rot', 'Healthy', 'Powdery Mildew']  # example, replace with your actual classes

# Streamlit app
st.title("🍎 Apple Leaf Disease Detection")

st.write("""
Upload an image of an apple leaf, and the model will predict if it is healthy or diseased.
""")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show resultv
    st.success(f"Predicted Disease: {predicted_class}")