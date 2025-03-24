import tensorflow as tf
import keras
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model


model = keras.models.load_model("model.h5")
# for layer in model.layers:
#     if isinstance(layer, tf.keras.layers.Lambda):
#         print(f"Lambda Layer Found: {layer.name}")
# for layer in model.layers:
#     if layer.name == "LAYER_216":
#         print(f"Layer Name: {layer.name}")
#         print(f"Layer Type: {type(layer)}")
#         print(f"Layer Config: {layer.get_config()}")
# print(model.get_layer("LAYER_215").output.shape)
# Get input to LAYER_216
prev_layer_output = model.get_layer("LAYER_215").output

# Replace with Reshape
fixed_layer = Reshape((284,), name="LAYER_216")(prev_layer_output)

fixed_model = Model(inputs=model.input, outputs=fixed_layer, name="converted_model_fixed")

# fixed_model.summary()
fixed_model.save("fixed-model.h5")