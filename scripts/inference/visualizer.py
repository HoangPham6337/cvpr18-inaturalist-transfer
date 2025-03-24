import tensorflow as tf
# import keras
from tensorflow.keras import models
# from keras.layers import TFSMLayer
# from keras import Model, Input
# import tf_keras as keras
# from tensorflow.keras.applications import InceptionV3

# model = models.load_model("saved_model_tf2")
model = models.load_model("model.keras")
model.summary()


# Create a Keras model that wraps the SavedModel
# inputs = Input(shape=(299, 299, 3), name="input")
# outputs = TFSMLayer("saved_model_tf2", call_endpoint="serving_default")(inputs)
# model = Model(inputs, outputs)

# model.summary()
