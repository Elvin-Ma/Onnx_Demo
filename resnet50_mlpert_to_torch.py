import onnx
import netron
import numpy as np

def cut_model(model, cut_nodes):
  for i in range(10):
    for node in model.graph.node:
      if node.name in cut_nodes:
        model.graph.node.remove(node)

  return model

def onnx_op_gen(op_type, input_names, output_name, **attr):
  node = onnx.helper.make_node(
         op_type,
         inputs = input_names,
         outputs = output_name,
         **attr,
         )
  return node

def weight_transform(model, weight_name):
  for item in model.graph.initializer:
    if item.name == "resnet_model/dense/kernel/read:0":
      raw_data = np.frombuffer(item.raw_data,dtype="float32").astype(np.float32)
      item.dims[0] = 1001
      item.dims[1] = 2048

      weight_data = raw_data.reshape(2048, 1001).T
      item.raw_data = weight_data.tobytes()

def tensor_value_info_gen(name, type, shape):
  return onnx.helper.make_tensor_value_info(name, type, shape)

def change_output(model, output_value_info):
  for i in range(5):
    for output in model.graph.output:
      model.graph.output.remove(output)
  model.graph.output.append(output_value_info)

if __name__ == "__main__":
  src_path = "resnet50_v1.onnx"
  dst_path = "resnet50_v1_cut.onnx"

  model = onnx.load(src_path)

  cut_nodes = ["softmax_tensor", "ArgMax", "resnet_model/dense/MatMul", "resnet_model/dense/BiasAdd"]

  model = cut_model(model, cut_nodes)

  attr_dict = {"alpha" : 1.0, "beta" : 1.0, "transB" : 1}
  gemm_node = onnx_op_gen("Gemm",
                          ["resnet_model/Squeeze:0",
                           "resnet_model/dense/kernel/read:0",
                           "resnet_model/dense/bias/read:0"],
                          ["output"],
                          **attr_dict)

  model.graph.node.append(gemm_node)

  weight_transform(model, "resnet_model/dense/kernel/read:0")

  output_valut_info = tensor_value_info_gen("output", 1, [1, 1001])

  change_output(model, output_valut_info)

  model = onnx.shape_inference.infer_shapes(model)
  onnx.save(model, dst_path)

  netron.start(dst_path)




