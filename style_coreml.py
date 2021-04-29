import tensorflow as tf
import coremltools as ct
import numpy as np

tf_model = tf.saved_model.load('./models/magenta_arbitrary-image-stylization-v1-256_2')
#tf_model = tf.keras.models.load_model('./models/magenta_arbitrary-image-stylization-v1-256_2')

tf_model.model.summary()

model = ct.convert(tf_model, source="tensorflow")

# Load from .h5 file
tf_model = tf.keras.applications.Xception(weights="imagenet", 
                                          input_shape=(299, 299, 3))

# Convert to Core ML
model = ct.convert(tf_model)

x = np.random.rand(1, 192, 192, 3)
tf_out = model.predict([x])