import onnx
import tensorflow as tf
from onnx2keras import onnx_to_keras
from tensorflow.keras import models
# Load ONNX model
onnx_model = onnx.load('model.onnx')

# for node in onnx_model.graph.node:
#     if node.op_type == "Squeeze":
#         print(f"Squeeze Node Name: {node.name}")
#         print(f"Inputs: {node.input}")
#         print(f"Outputs: {node.output}")
# print(onnx_model.graph.value_info)

# inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

# Now value_info should be populated
# for vi in inferred_model.graph.value_info:
#     if vi.name == "InceptionV3/Logits/Conv2d_1c_1x1/BiasAdd:0":
#         dims = vi.type.tensor_type.shape.dim
#         shape = [dim.dim_value for dim in dims]
#         print(f"Shape: {shape}")
for node in onnx_model.graph.node:
    if node.op_type == "Squeeze":
        attr_names = [attr.name for attr in node.attribute]
        
        # If "axes" is missing, add it manually
        if "axes" not in attr_names:
            print(f"Fixing missing 'axes' attribute in node: {node.name}")
            
            # Define the axes to squeeze (assuming you want to remove singleton dimensions)
            node.attribute.append(onnx.helper.make_attribute("axes", [2, 3]))  # Adjust as needed
input_all = [node.name for node in onnx_model.graph.input]
input_shapes = {input_all[0]: (299, 299, 3)} 
# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, input_all, name_policy="renumerate")
k_model.save("model.h5")
k_model.summary()