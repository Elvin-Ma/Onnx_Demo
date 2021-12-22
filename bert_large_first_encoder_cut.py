import onnx
import onnxruntime
import netron
import torch
import torch.nn as nn
import numpy as np
from onnx import shape_inference
import netron
from pytorch_pretrained_bert import BertTokenizer

model = onnx.load("bert_large.onnx")
name_stay = []

dropout = [1001, 1003, 1005, 1007]
layernorm = ["LayerNormalization_2","LayerNormalization_4", "LayerNormalization_6"]
constant = ["position_01", "op_min_ends_expand_10", "start_expand_10","axes_expand_10","concat_shape_20",
            "concat_shape_50", "concat_shape_31", "concat_shape_38", "485", "528", "531", "526", "404", "406"]

for j in range(20):
    for node in model.graph.node:
        node_name = node.name  # got the node's name

        if (node_name == "Transpose_1003"):
          model.graph.node.remove(node)
          continue

        if (node.output[0] in constant):
          continue

        if (node.op_type == "LayerNormalization" and node_name not in layernorm):
          model.graph.node.remove(node)
          continue
        if (node.op_type == "Constant" and node_name not in constant):
          model.graph.node.remove(node)
          continue

        name_split = node_name.split("_")

        if (len(name_split) > 0 and name_split[-1].isdigit()):
          ordernum = name_split[-1]
          ordernum = int(ordernum)
        else:
          name_stay.append(node_name)

          continue

        if  ordernum > 147 and ordernum not in dropout :
            model.graph.node.remove(node)

for j in range(10):
    for output in model.graph.output:
        model.graph.output.remove(output)

inputnamelist=[] # all input list for graph
for node in model.graph.node:
    for inputname in node.input:
        if inputname not in inputnamelist:
            inputnamelist.append(inputname)


for i in range(10):
    for ini in model.graph.initializer:
        if ini.name not in inputnamelist:
            model.graph.initializer.remove(ini)
    print(len(model.graph.initializer))

intermediate_layer_value_info2= onnx.helper.make_tensor_value_info("551", 1, ["batch", "sequence", 1024])
model.graph.output.append(intermediate_layer_value_info2)

path_new = "bert_encoder.onnx"
onnx.save(model, path_new)

netron.start(path_new)
