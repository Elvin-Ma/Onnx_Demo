from operator import mod
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import os
import time
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

def get_onnx_model(path):
  model = models.resnet50(pretrained=True)
  model.eval()

  x = torch.randn(1, 3, 224, 224, requires_grad=True)

  torch.onnx.export(model,
                    x,
                    path,
                    export_params=True,#导入参数
                    opset_version=10,
                    input_names=["input"], #指定输入的名称（key）
                    output_names=['output'],
                    dynamic_axes={'input':{0:'batchsize'}, 'output':{0:'batchsize'}}
                    )

def model_process(path):
  '''make dymamic batch model'''
  model = onnx.load(path)
  input_name = model.graph.input[0].name
  output_name = model.graph.output[0].name
  model.graph.input.pop()
  model.graph.output.pop()

  input_layer_value_info= onnx.helper.make_tensor_value_info(input_name, 1, ["batch", 3, 224, 224])
  model.graph.input.append(input_layer_value_info)
  output_layer_value_info= onnx.helper.make_tensor_value_info(output_name, 1, ["batch", 1000])
  model.graph.output.append(output_layer_value_info)

  model = onnx.shape_inference.infer_shapes(model)
  new_path = path[0:-5] + "_new.onnx"
  onnx.save(model, new_path)

  return new_path

def image_process(path):
  img = Image.open(path)
  img = img.resize((224, 224))
  img = np.array(img, dtype = np.float32)
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  means= np.array(mean).reshape((1, 1, -1))
  std = np.array(std).reshape((1, 1, -1))
  img = (img / 255.0 - means) / std

  # means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
  # img -= means

  img = np.transpose(img, [2, 0, 1]).astype(np.float32)
  return img

def image_process_int8(path):
  img = Image.open(path)
  img = img.resize((224, 224))
  img = np.array(img, dtype = np.float32)
  mean = [123.68, 116.28, 103.53]
  mean = np.array(mean).reshape((1, 1, -1))
  img = (img - mean)
  img = np.transpose(img, [2, 0, 1]).astype(np.float32)
  return img


def run(model_path, image_path):
  session = onnxruntime.InferenceSession(model_path)
  input_data = []
  input_data.append(image_process(image_path))
  input_name_1 = session.get_inputs()[0].name
  outputs = session.run([],{input_name_1:input_data})
  return outputs

if __name__ == "__main__":
  path_1 = "dog.jpg"
  path_2 = "dog2.jpg"
  path_3 = "ILSVRC2012_val_00000067.JPEG"

  # model_path = "resnet50_mlperf_equal_conv.onnx"
  model_path_1 = "resnet50_v1.onnx"
  model_path_2 = "resnet50_v1_cut.onnx"

  output_1 = run(model_path_1, path_2)
  output_2 = run(model_path_2, path_2)
  diff = (output_1-output_2).sum()
  print(diff)

  # print(outputs[0])
  # print("onnx 推理用时：", time.time() - a)
  # print(outputs[0].shape)
  # print(outputs[0][0].argmax()) # get the output result
  # print(outputs[0][0].argmax())
