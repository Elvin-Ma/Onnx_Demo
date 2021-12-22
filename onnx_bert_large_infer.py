# coding:utf8
import torch
import os
import onnx
import onnxruntime
import numpy as np
from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM
# from ipdb import set_trace

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
tokenized_text.extend(['[MASK]']*(512-14))

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids.extend([1]*(512-14))

mask = [1]*14
mask.extend([0]*(512-14))

token_tensors = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
mask = torch.tensor([mask])

session = onnxruntime.InferenceSession("bert_large.onnx")
print("Bert onnx model has loaded successful!")

input_name_1 = session.get_inputs()[0].name
input_name_2 = session.get_inputs()[1].name
input_name_3 = session.get_inputs()[2].name
label_name = session.get_outputs()[0].name

outputs = session.run([],{input_name_1:token_tensors.numpy(),input_name_2:segments_tensors.numpy(), input_name_3: mask.numpy()})

predictions = torch.from_numpy(outputs[0])
masked_index = 8
k = 3
probs,indices = torch.topk(torch.softmax(predictions[0,masked_index],-1),k)
# probs,indices = torch.topk(torch.softmax(predictions[0,masked_index],-1),k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
# print("輸入 tokens ：", tokenized_text[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokenized_text[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokenized_text), '...')

print('predictions successful')
