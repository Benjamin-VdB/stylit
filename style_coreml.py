import tensorflow as tf
import coremltools as ct
import numpy as np

#tf_model = tf.saved_model.load('./models/magenta_arbitrary-image-stylization-v1-256_2')
tf_model = tf.saved_model.load('./models/magenta_arbitrary-image-stylization-v1-256_2')
tf_variables = tf_model.variables
tf_model.__call__
tf_model.trainable_variables

infer = tf_model.signatures['serving_default']
inputs = infer.structured_input_signature
outputs = infer.structured_outputs



tf.saved_model.save(tf_model, './models/magenta_arbitrary-image-stylization-saved')

ct.convert([infer], source='tensorflow')

tf_model = tf.saved_model.load('./models/magenta_arbitrary-image-stylization-v1-256_1')
tf_model.summary()

model = ct.convert(tf_model, source='tensorflow')

# Load from .h5 file
tf_model = tf.keras.applications.Xception(weights="imagenet", 
                                          input_shape=(299, 299, 3))

# Convert to Core ML
model = ct.convert([tf_model], source='tensorflow')

x = np.random.rand(1, 256, 256, 3)
tf_out = model.predict([x])


# convert functions
predict_model = tf.saved_model.load('./models/magenta_colab/predict_incepv3')
transfer_model = tf.saved_model.load('./models/magenta_colab/transfer_incepv3')
concrete_func = predict_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
print(concrete_func.inputs)
concrete_func.inputs[0].set_shape([1, 256, 256, 3])

# Range for the sequence dimension is "arbitary"
input_shape = ct.Shape(shape=(1, ct.RangeDim(), ct.RangeDim(), 3))
model_input = ct.TensorType(shape=input_shape)

# Convert the model
predict_mlmodel = ct.convert(model=[concrete_func], source='tensorflow', inputs=[model_input])


predict_model = tf.saved_model.load('./models/magenta_colab/predict_incepv3')
predict_model.summary()

model = ct.convert(predict_model, source='tensorflow')

# 
def convert_coreml_model_dynamic(saved_model_path, coreml_path, type='style_predict'):
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

    coreml_path = ct.convert([concrete_func], source='tensorflow')

convert_coreml_model_dynamic('./models/magenta_colab/predict_incepv3', 'style_predict')

convert_coreml_model_dynamic('./models/magenta_colab/transfer_incepv3', 'style_transfer', type='style_transfer')