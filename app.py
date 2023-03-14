import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# creating the title
st.title("Covid-19 Image Classifier")

# creating a side bar 
st.sidebar.title("Created By:")
st.sidebar.subheader("P.S.S.Keerthana")
st.sidebar.subheader("P.Komal Sai Anurag")
st.sidebar.subheader("Udayagiri Varun")
st.sidebar.subheader("Sejal Singh")
st.sidebar.image("https://post.healthline.com/wp-content/uploads/2020/08/chest-x-ray_thumb.jpg", width=None)

# creating an uploader to upload the Chest X-ray images
upload_file = st.file_uploader("Upload the Chest X-ray", type = 'jpg')

# creating a predict button
generate_pred = st.button("Predict")

model = tf.keras.models.load_model('https://github.com/saianurag234/Covid_komalsai234/blob/main/covid_classifier.h55', compile=False)
