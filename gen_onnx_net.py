import onnx
import numpy as np
import onnxruntime
input = onnx.helper.make_tensor_value_info("input",onnx.TensorProto.FLOAT,[1,1,3,3])
output = onnx.helper.make_tensor_value_info("output",onnx.TensorProto.FLOAT,[1,3,3,3])
conv_node = onnx.helper.make_node(
            'Conv',
            inputs=['input', 'conv1.weight','conv1.bias'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads = [1,1,1,1]
        )
weight_data = np.ones([3,1,3,3],dtype=np.float32)
bias_data = 2*np.ones([3],dtype=np.float32)
weight_tensor=onnx.helper.make_tensor("conv1.weight",onnx.TensorProto.FLOAT,[3,1,3,3],weight_data.tobytes(),raw=True)
bias_tensor=onnx.helper.make_tensor("conv1.bias",onnx.TensorProto.FLOAT,[3],bias_data.tobytes(),raw=True)



graph = onnx.helper.make_graph(nodes=[conv_node], name='test_graph',
                   inputs=[input],
                   outputs=[output],initializer=[weight_tensor,bias_tensor])

model = onnx.helper.make_model(graph)
onnx.checker.check_model(model)
onnx.save(model, "conv.onnx")


session = onnxruntime.InferenceSession("conv.onnx")
input = np.ones([1,1,3,3],dtype=np.float32)
result = session.run([],{"input":input})
print(result[0])
