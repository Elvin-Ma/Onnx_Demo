from onnxsim import simplify
import netron
import onnx

input_path = "encoder_new.onnx"
# netron.start(input_path)
output_path = "encoder_simply.onnx"

onnx_model = onnx.load(input_path)  # load onnx model
model_simp, check = simplify(onnx_model)

# # assert(check, "Simplified ONNX model could not be validated")

onnx.save(model_simp, output_path)
netron.start(output_path)
print('finished exporting onnx')
