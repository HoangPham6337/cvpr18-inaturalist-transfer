import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer

model:tf.types.experimental.GenericFunction = tf.saved_model.load("saved_model_tf1")
# class TF2ModelWrapper(Model):
#     def __init__(self, tf2_model):
#         super().__init__()
#         self.tf2_model = tf2_model
#         self._trackable_variables = tf2_model.variables  # Ensure Keras tracks them

#     def call(self, inputs):
#         return self.tf2_model.signatures["serving_default"](input=inputs)["output"]

# input_tensor = Input(shape=(299, 299, 3), name="input")

# wrapped_model = TF2ModelWrapper(model)

# keras_model = Model(inputs=input_tensor, outputs=wrapped_model(input_tensor))

# keras_model.summary()

f = model.signatures["serving_default"]
print(f(x=tf.constant([[1.]])))