import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pathlib

content_image_path = './img/20201225_012644000_iOS.jpg'
style_image_path = './style/basquiat/00000084.jpg'

# Load content and style images (see example in the attached colab).
content_image = plt.imread(content_image_path)
style_image = plt.imread(style_image_path)

# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

# Optionally resize the images. It is recommended that the style image is about
# 256 pixels (this size was used when training the style transfer network).
# The content image can be any size.
style_image = tf.image.resize(style_image, (256, 256))

# Load image stylization module.
# hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
tf_model = tf.saved_model.load('./models/magenta_arbitrary-image-stylization-v1-256_2')
infer = tf_model.signatures["serving_default"]

# Stylize image.
# outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
outputs = tf_model(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

tf.keras.preprocessing.image.save_img( './out/stylised3.jpg', stylized_image[0,:,:,:])


# conversion
converter = tf.lite.TFLiteConverter.from_saved_model('./models/magenta_arbitrary-image-stylization-v1-256_2')
tflite_model = converter.convert()
with open('./models/magenta_arbitrary-image-stylization-v1-256_2_converted.tflite', 'wb') as f:
  f.write(tflite_model)

# conversion2
converter = tf.lite.TFLiteConverter.from_saved_model('./models/magenta_arbitrary-image-stylization-v1-256_2')
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("./models/magenta_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"magenta_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model('./models/magenta_arbitrary-image-stylization-v1-256_2')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"magenta_model_quant_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

converter = tf.lite.TFLiteConverter.from_saved_model('./models/magenta_arbitrary-image-stylization-v1-256_2')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"magenta_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)




# check
interpreter = tf.lite.Interpreter(model_path='./models/magenta_tflite_models/magenta_model_quant.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()


# Dynamic range quantization
def convert_tflite_model_dynamic(saved_model_path, tflite_path, type='style_predict'):
    model = tf.saved_model.load(saved_model_path)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    print(concrete_func.inputs)
    if type == 'style_predict':
        concrete_func.inputs[0].set_shape([1, 256, 256, 3])
    
    # if you'd prefer static shape in your models just uncomment this block
    # else:
    #     for input in concrete_func.inputs:
    #         if input.name == 'content_image:0':
    #             input.set_shape([1, 384, 384, 3])
    #         elif input.name == 'Conv/BiasAdd:0':
    #             input.set_shape([1, 1, 1, 100])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print('Quantized model:', tflite_path, 
        'Size:', len(tflite_model) / 1024, "kb")


model_type = 'original' #@param ["original", "mobile"]
#@markdown - Original network (InceptionV3, alpha=1)
#@markdown - Small (MobileNetV2, alpha=0.25)

# Define saved model path
style_predict_network = 'gs://mobile-ml-wg/arbitrary_style_transfer/{}/SavedModel/predict'.format(model_type)
style_transform_network = 'gs://mobile-ml-wg/arbitrary_style_transfer/{}/SavedModel/transfer'.format(model_type)

convert_tflite_model_dynamic(style_predict_network, './models/magenta_tflite_models/', type='style_predict')