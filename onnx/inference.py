import onnx
import onnxruntime as ort
import cv2
import numpy as np
import scipy.special

# Load image using OpenCV
image_path = "b1af61da4034a59cb7dbad74dc7a6824.jpg"
image = cv2.imread(image_path)

# Resize to match model input size (assuming 299x299x3 for InceptionV3)
image = cv2.resize(image, (299, 299))

# Normalize image (assuming model expects values in [0,1])
image = image.astype(np.float32) / 255.0

# Add batch dimension (ONNX models expect shape: [batch, height, width, channels])
image = np.expand_dims(image, axis=0)
# Load ONNX model
# onnx_model = onnx.load("model.onnx")
session = ort.InferenceSession("model-prunned.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: image})
# print(outputs[0].shape) 
# print("Model Output:", outputs)
probabilities = scipy.special.softmax(outputs[0], axis=1)
top_5_indices = np.argsort(probabilities[0])[-5:][::-1]
for i in top_5_indices:
    print(f"Class {i}: {probabilities[0][i]:.4f}")