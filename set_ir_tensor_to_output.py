import onnx

def append_tensor_as_output():
  output = model.graph.value_info[2]
  model.graph.output.append(output)
  
if __name__ == "__main__":
  append_tensor_as_output()
