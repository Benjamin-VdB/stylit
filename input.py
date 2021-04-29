import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit import caching
import cv2
import imutils
import matplotlib.pyplot as plt


def style_transfer(content_image, style_image, model):

    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    style_image = tf.image.resize(style_image, (256, 256))

    # Stylize image.
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    return outputs[0]


def image_input(model, style_image):
    
    content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])

    if content_file is not None:
        #content = plt.imread(content_file)
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        #content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR) 
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
     
    #WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200) 
    #content = imutils.resize(content, width=WIDTH)
    generated = style_transfer(content, style_image, model)
    st.sidebar.image(content, width=300, channels='BGR')
    st.image(generated, channels='BGR', clamp=True)


def webcam_input(model, style_image):
    st.header("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([], channels='BGR')
    SIDE_WINDOW = st.sidebar.image([], width=100, channels='BGR')
    camera = cv2.VideoCapture(0)
    #WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50))) 

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # orig = frame.copy()
        orig = imutils.resize(frame, width=300)
        #frame = imutils.resize(frame, width=WIDTH)
        target = style_transfer(frame, style_image, model)
        FRAME_WINDOW.image(target)
        SIDE_WINDOW.image(orig)
    else:        
        st.warning("NOTE: Streamlit currently doesn't support webcam. So to use this, clone this repo and run it on local server.")
        st.warning('Stopped')