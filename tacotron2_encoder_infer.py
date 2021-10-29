#coding:utf8
# from _typeshed import OpenTextModeWriting
from typing import Sequence
import onnx
import onnxruntime
import netron
import numpy as np
from onnx import shape_inference


def model_change():
  model = onnx.load("encoder.onnx")

  # output_layer_value_info= onnx.helper.make_tensor_value_info("238",10,[167,2,1,256])
  # output_layer_value_info= onnx.helper.make_tensor_value_info("241",10,[167,1,2,256])
  # output_layer_value_info = onnx.helper.make_tensor_value_info("101", 10, [167,1,512]) # Transpose_15
  # output_layer_value_info = onnx.helper.make_tensor_value_info("228", 10, [2,1,256]) #
  # output_layer_value_info = onnx.helper.make_tensor_value_info("96", 10, [1,512,167]) # Cast_10
  # output_layer_value_info = onnx.helper.make_tensor_value_info("99", 10, [1,512,167]) # Relu_13
  # output_layer_value_info = onnx.helper.make_tensor_value_info("86", 10, [1,167,512]) # Gather_0
  # output_layer_value_info = onnx.helper.make_tensor_value_info("92", 10, [1, 512, 167]) # Gather_0
  # output_layer_value_info = onnx.helper.make_tensor_value_info("87", 10, [1, 512, 167]) # Gather_0

  # model.graph.output.append(output_layer_value_info)

  for i in range(10):
    for item in model.graph.input:
      model.graph.input.remove(item)

  sequences= onnx.helper.make_tensor_value_info("sequences", 7, [1, 167])
  model.graph.input.append(sequences)

  sequence_lengths= onnx.helper.make_tensor_value_info("sequence_lengths", 7, [1])
  model.graph.input.append(sequence_lengths)

  onnx.shape_inference.infer_shapes(model)

  onnx.save(model,"encoder_new.onnx")

def inference():
  session = onnxruntime.InferenceSession("encoder_new.onnx")

  input_name_1 = session.get_inputs()[0].name
  input_name_2 = session.get_inputs()[1].name

  data_1 = np.array([50]*167).astype(np.int64).reshape((1,167))
  data_2 = np.array([167]).astype(np.int64)

  outputs = session.run([],{input_name_1:data_1, input_name_2:data_2})
  print("Shape of 86 is ", outputs[1].shape)
  print("\n")
  print(outputs[1])

if __name__ == "__main__":
  model_change()
  inference()
