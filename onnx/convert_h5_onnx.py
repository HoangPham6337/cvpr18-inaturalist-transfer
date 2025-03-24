import tensorflow as tf
import tf2onnx

keras_model = tf.keras.models.load_model("prunned.h5")

# Convert to ONNX
onnx_model_path = "model-prunned.onnx"
spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)  # Adjust input shape if needed

# Perform conversion
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13)

# Save ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Successfully converted model to ONNX: {onnx_model_path}")