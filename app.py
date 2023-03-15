import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array

# creating the title
st.title("Covid-19 Image Classifier")

# creating a side bar 
st.sidebar.title("Created By:")
st.sidebar.subheader("P.S.S.Keerthana")
st.sidebar.subheader("P.Komal Sai Anurag")
st.sidebar.subheader("Udayagiri Varun")
st.sidebar.subheader("Sejal Singh")
st.sidebar.image("https://post.healthline.com/wp-content/uploads/2020/08/chest-x-ray_thumb.jpg", width=None)

model = tf.keras.models.load_model('covid_classifier.h5')

# creating an uploader to upload the Chest X-ray images
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

# creating a predict button
generate_pred = st.button("Predict")

def predictions(image,model):
    image = cv2.imread(image)
    img = cv2.resize(image, (128,128,3))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    
    return pred


if generate_pred:
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image,caption="Chest X-ray",use_column_width=True)
        pred = predictions(bytes_data, model)
        st.title(pred)
        
        
