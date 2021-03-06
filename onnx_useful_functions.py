import onnx

def append_tensor_as_output():
  output = model.graph.value_info[2]
  model.graph.output.append(output)
  
  
def build_weight_data():
  weight_data = np.ones([3,1,3,3],dtype=np.float32)
  bias_data = 2*np.ones([3],dtype=np.float32)
  weight_tensor=onnx.helper.make_tensor("conv1.weight",onnx.TensorProto.FLOAT,[3,1,3,3],weight_data.tobytes(),raw=True)
  bias_tensor=onnx.helper.make_tensor("conv1.bias",onnx.TensorProto.FLOAT,[3],bias_data.tobytes(),raw=True)

def shape_infer():
  model = onnx.load("resnet50_int8.onnx")
  model = onnx.shape_inference.infer_shapes(model)
  onnx.save(model, "resnet50_int8_shaped_temp.onnx")
  netron.start("resnet50_int8_shaped_temp.onnx")
  
if __name__ == "__main__":
  append_tensor_as_output()
