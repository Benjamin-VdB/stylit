import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

esrgan_path = './models/lite-model_esrgan-tf2_1.tflite'

interpreter = tf.lite.Interpreter(model_path=esrgan_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], preprocess_image(style_image, 50))
  

tf_model = tf.saved_model.load('./models/esrgan-tf2_1')
converter = tf.lite.TFLiteConverter.from_saved_model('./models/esrgan-tf2_1')
tflite_model = converter.convert()
with open('./models/esrgan_tflite_converted.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='./models/lite-model_esrgan-tf2_1.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], style_input_data.astype(np.float32))

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
