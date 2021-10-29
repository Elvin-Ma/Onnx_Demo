# coding:utf8
import torch
import os
import onnx
import onnxruntime
import numpy as np
from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

token_tensors = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

session = onnxruntime.InferenceSession("bert.onnx")
print("Bert onnx model has loaded successful!")
# set_trace()
input_name_1 = session.get_inputs()[0].name
input_name_2 = session.get_inputs()[1].name

outputs = session.run([],{input_name_1:token_tensors.numpy(),input_name_2:segments_tensors.numpy()})

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
