import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools


# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# Run cartoon transform on preprocessed style image
def run_cartoon_transform(preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=cartoon_model)

  # Set model input
  input_details = interpreter.get_input_details()
  interpreter.resize_tensor_input(input_details[0]["index"], preprocessed_content_image.shape, strict=True)
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.invoke()

  # Transform content image.
  cartooned_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return cartooned_image


cartoon_model = "./models/whitebox_cartoon_gan_fp16.tflite"

content_path = './img/20210123_033846000_iOS.jpg'

# Load the input images.
content_image = load_img(content_path)

# Preprocess the input images.
preprocessed_content_image = preprocess_image(content_image, 600)
print('Content Image Shape:', preprocessed_content_image.shape)
plt.subplot(1, 2, 1)
imshow(preprocessed_content_image, 'Content Image')

# Stylize the content image using the style bottleneck.
cartooned_image = run_cartoon_transform(preprocessed_content_image)

# Visualize the output.
imshow(cartooned_image, 'Cartooned Image')

# Super resolution
esrgan_interpreter = tf.lite.Interpreter(model_path='./models/lite-model_esrgan-tf2_1.tflite')
input_details = esrgan_interpreter.get_input_details()
output_details = esrgan_interpreter.get_output_details()