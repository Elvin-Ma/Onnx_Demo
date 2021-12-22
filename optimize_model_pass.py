import onnx
import onnxoptimizer
import netron
# print('&&&load export_model_path = '+str(export_model_path))
original_model = onnx.load("resnet50_int8.onnx")

#get all the onnx optimizer pass supported
all_passes = onnxoptimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print('\t{}'.format(p))
print()
r"""
Available optimization passes:
eliminate_deadend
eliminate_duplicate_initializer
eliminate_identity
eliminate_if_with_const_cond
eliminate_nop_cast
eliminate_nop_dropout
eliminate_nop_flatten
eliminate_nop_monotone_argmax
eliminate_nop_pad
eliminate_nop_transpose
eliminate_unused_initializer
extract_constant_to_initializer
fuse_add_bias_into_conv
fuse_bn_into_conv
fuse_consecutive_concats
fuse_consecutive_log_softmax
fuse_consecutive_reduce_unsqueeze
fuse_consecutive_squeezes
fuse_consecutive_transposes
fuse_matmul_add_bias_into_gemm
fuse_pad_into_conv
fuse_transpose_into_gemm
lift_lexical_references
nop
split_init
split_predict
"""
#from transformers.convert_graph_to_onnx import convert
#convert(framework="pt",model=model_path,output=
passes = ['fuse_add_bias_into_conv', 'fuse_bn_into_conv']
optimized_model = onnxoptimizer.optimize(original_model, passes)

# print('The model after optimization:\n\n{}'.format(onnx.helper.printable_graph(optimized_model.graph)))

# save new model
onnx.save(optimized_model, "resnet50_optimized_model_path.onnx")
# torch.cuda.empty_cache()
netron.start("resnet50_optimized_model_path.onnx")
