import torch
from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from ipdb import set_trace
import onnx
import onnxruntime

# load vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenized input
text = " [CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP] "
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# token convert to vocabulary index
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# define sentence A and B index
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# input --> pytorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# model = BertModel.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

onnx_path = 'bert_base_v13.onnx'
torch.onnx.export(model,(tokens_tensor,segments_tensors), onnx_path, opset_version = 13,export_params=True,verbose=False)

#put cuda and model on cuda
# token_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')
# torch.save(model,f='bert.pth')

with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)
    # predictions = predictions[0]
# masked_index = 2
k = 3
probs,indices = torch.topk(torch.softmax(predictions[0,masked_index],-1),k)
# probs,indices = torch.topk(torch.softmax(predictions[0,masked_index],-1),k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
# print("輸入 tokens ：", tokens[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokenized_text[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokenized_text), '...')

