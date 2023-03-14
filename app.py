import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

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
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.title(image)
    
