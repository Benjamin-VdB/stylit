
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import cv2
import imutils
from streamlit import caching
from input import webcam_input
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Function to load an image from a file, and add a batch dimension.
@st.cache
def load_img(file):
  #img = tf.io.read_file(path_to_img)
  img = plt.imread(file)
  #img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

# Function to pre-process by resizing an central cropping it.
@st.cache
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes target_dim
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  #short_dim = min(shape)
  #scale = target_dim / short_dim
  long_dim = max(shape)
  scale = target_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
  #image = tf.image.resize_with_crop_or_pad(image, new_shape[0], new_shape[1])

  return image

def postprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    #short_dim = min(shape)
    #scale = target_dim / short_dim
    long_dim = max(shape)
    scale = target_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    #image = tf.image.resize_with_crop_or_pad(image, new_shape[0], new_shape[1])

    return image

# Run cartoon transform on preprocessed style image
def cartoon_transform(preprocessed_content_image):

  # Set model input
  interpreter.resize_tensor_input(input_details[0]["index"], preprocessed_content_image.shape, strict=False)
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.invoke()

  # Transform content image.
  cartooned_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return cartooned_image

def image_input(output_size):
    
    content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])

    if content_file is not None:
        #content = plt.imread(content_file)
        content = load_img(content_file)
        prepocessed_image = preprocess_image(content, output_size)
        #content = np.array(content) #pil to cv
        #content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR) 
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
     
    #WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200) 
    #content = imutils.resize(content, width=WIDTH)
    generated = cartoon_transform(prepocessed_image)
    st.sidebar.image(content_file, width=300, channels='BGR')
    st.image(generated, channels='RGB', clamp=True)


def webcam_input():
    st.header("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([], channels='RGB')
    SIDE_WINDOW = st.sidebar.image([], width=100, channels='RGB')
    camera = cv2.VideoCapture(0)
    #WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50))) 

    while run:
        _, frame = camera.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # orig = frame.copy()
        # orig = imutils.resize(frame, width=300)
        #frame = imutils.resize(frame, width=WIDTH)
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = frame[tf.newaxis, :]
        prepocessed_image = preprocess_image(frame, 256)
        target = cartoon_transform(prepocessed_image)
        FRAME_WINDOW.image(target, clamp=True)
        # SIDE_WINDOW.image(frame, clamp=True)
    else:        
        st.warning("NOTE: Streamlit currently doesn't support webcam. So to use this, clone this repo and run it on local server.")
        st.warning('Stopped')

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("StylIt - Test tflite models")
st.sidebar.title('Navigation')
method = st.sidebar.radio('Go To ->', options=['Image', 'Webcam'])
st.sidebar.header('Options')
output_size = st.sidebar.select_slider('Output Size', [256,512,1024])
style_file = st.sidebar.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])
style_image = plt.imread(style_file)
style = Image.open(style_file)
style_image = np.array(style) #pil to cv

cartoon_model = "./models/whitebox_cartoon_gan_fp16.tflite"
style_predict_model = './models/magenta_colab/style_predict_f16_mobilev2.tflite'
style_transfer_model = './models/magenta_colab/style_transfer_f16_mobilev2.tflite'

# Load the model.
interpreter = tf.lite.Interpreter(model_path=cartoon_model)
# Get model input details
input_details = interpreter.get_input_details()


if method == 'Image':
    image_input(output_size)
else:
    webcam_input()