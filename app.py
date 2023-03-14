import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
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

# creating an uploader to upload the Chest X-ray images
upload_file = st.file_uploader("Upload the Chest X-ray", type = 'jpg')

# creating a predict button
generate_pred = st.button("Predict")


model = tf.keras.models.load_model('covid_classifier.h5', compile=False)
def import_n_pred(image_data,model):
    image = load_img(image_data,target_size=(128,128))  
    img = img_to_array(image)
    img = np.array(image)
    image = img.resize((128,128))
    img = img.reshape(1,128,128,3)
   
    pred = model.predict(img)
    return pred

if generate_pred:
    image = Image.open(upload_file)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image,model)
    labels = ['Covid-19','Healthy']
    st.title(pred)
