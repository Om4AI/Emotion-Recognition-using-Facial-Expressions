import streamlit as st
# import cv2
import streamlit.components.v1 as comp
import webbrowser
import warnings
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import numpy as np
import io
from io import BytesIO
import base64
import time
import skimage.transform as skimg

from tensorflow.keras.preprocessing import image

# Deprecate warnings to avoid image upload error
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# Warnings ignore 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
url = 'https://www.streamlit.io/'
# Page configuration
st.set_page_config(
    page_title="EMOTION DETECTOR",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.set_option('deprecation.showfileUploaderEncoding', False)
# Title & Headers
st.title("Emotion Detector")

st.subheader("AI based Emotion Detector\n\n\tMake a prediction using TensorFlow & Keras")
st.markdown("***")

left, mid, right = st.columns(3)

# Other imp functions -------------------------

# Image Uploader
img = left.file_uploader("Image to Test", type=['png', 'jpg','jpeg'])
if img:
    left.image(img, caption="Uploaded Image")
else:
    left.success("Please Upload an Image")

mid.image("ai.gif")

# Prediction ------------------------------

model_path = "Emotion_detect-1.h5"
model = keras.models.load_model(model_path)

class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


if (img):
    data = img.read()
    img = Image.open(io.BytesIO(data))
    x = image.img_to_array(img)
    # img.save("User_image.jpeg")

# path = "User_image.jpeg"

if (img):x = skimg.resize(x, (160,160,3))

# img = image.load_img(path, target_size =(160,160))

res= ""
 # Convert the image into array
if (img) :
    x = np.expand_dims(x, axis=0) # Expand the images as if they were many images
    images = np.vstack([x]) # Important
    classes = model.predict(images, batch_size=10) 
    score = tf.nn.softmax(classes[0]) 

    res = class_names[np.argmax(score)]
# --------------------------------------

# Output here
res = str.upper(res)
mid.markdown('<h3 align = "center"> Emotion Here </h3>',
             unsafe_allow_html=True)          
mid.success(res)



# -----------------------Widgets----------------------
# Side widgets
st.sidebar.title("Side-Bar Widgets")
st.sidebar.write(" ")

st.sidebar.map()
if st.sidebar.button('Feedback'):
    webbrowser.open_new_tab("https://forms.gle/SV1z3q4TWpLbgpMX9")

st.sidebar.write(" ")
# date picker
d = st.sidebar.date_input(
    "When's your birthday"
)
st.sidebar.write('Your birthday is:', d)
st.sidebar.download_button('Download Results', ("Detected Emotion - "+res))
st.sidebar.write(" ")
# Download Results
st.sidebar.write(" ")
# Notes area
user_input = st.sidebar.text_area("Add Notes Here")
st.sidebar.download_button('Download Notes', user_input)

st.sidebar.write(" ")
#Timer
st.sidebar.write("Web-Timer")
def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    st.sidebar.write('Times-up!!⌚')

    
# input time in seconds
t = st.sidebar.number_input("Enter the time in seconds: ",step=5)
  
# function call
if st.sidebar.button('Start'):
    countdown(int(t))

# References
if st.sidebar.button('References'):
    webbrowser.open_new_tab(url) 

#end-of side bar
# ----------------------------Widgets End-----------------------------
# Get Camera

# run = right.button("Open Camera")  # Assign it to right side
# cam = cv2.VideoCapture(0)
turn_on = right.button("TURN ON CAMERA")
# FRAME_WINDOW = right.image([])
# while turn_on:
#     ret, frame = cam.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
    
# Turn Off camera
off = right.button("TURN OFF CAMERA")
if off:
    turn_on = False
