import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.header('Klasifikasi Bunga Menggunakan Arsitektur VGG-19')

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Ensure the model file is valid
model_path = 'Flower_Recog_Model.keras'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def classify_images(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    class_name = flower_names[class_index]
    confidence = np.max(score) * 100

    return f'The image belongs to {class_name} with a confidence of {confidence:.2f}%'

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Ensure the 'upload' directory exists
    if not os.path.exists('upload'):
        os.makedirs('upload')
    
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(file_path, caption='Uploaded Image', width=200)
    st.markdown(classify_images(file_path))
