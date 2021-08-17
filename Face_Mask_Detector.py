import streamlit as st
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import numpy as np
# import cv2
import pandas as pd
import io
from io import BytesIO
import base64

# Keras utils
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Title & Headers
st.title("COVID-19 Face Mask Detector")
st.subheader("AI based Face Mask Detector\n\n\tMake a prediction using TensorFlow & Keras")
# Expander & Description
with st.expander("Project Description"):
    st.write("This Face Mask Detector developed by Om Mule (All rights reserved).")
    st.write("From the past year there has been the life threatning pandemic of COVID-19 which has led to a lot of deaths. Face Masks are very essential which can prevent the spread of pandemic to a great extent.")
    st.write("This project is specially developed to be detect if a person is wearing Face mask or not.")

# Get Image & Layout items
left, right = st.columns(2)

image = left.file_uploader("Image to Test")
right.image(image, caption="Uploaded Image")

# # Face Mask Model path
# model_path = "D:\Deep Learning and Machine Learning Online Degree\Github\Face Mask Detector\WebApp Deployment\MobileNetV2-facemask.h5"

# model = keras.models.load_model(model_path)

# # Model predict function
# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(160,160))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x, mode='caffe')
    
#     preds = model.predict(x)
#     return preds

# st.write(image.img_to_array())

# Download the image
# result = Image.frombytes(image.size(),image)

data = image.read()

image = Image.open(io.BytesIO(data))
# image.save("Face_Mask.jpeg")
image.save("https://drive.google.com/drive/folders/13z9BZb2rzJuZZkec3oLwqspgMmvnFe6o?usp=sharing/Test.jpeg")

left.success("Face Mask Detected")
left.error("No Face Mask")
st.sidebar.map()

