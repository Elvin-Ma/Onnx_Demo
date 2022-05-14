import onnx
import onnxruntime
import numpy as np
from scipy.io import wavfile
from text import text_to_sequence

np.set_printoptions(threshold=np.inf)

def encoder_infer(input_data, encoder_model):
  session = onnxruntime.InferenceSession(encoder_model.SerializeToString())
  sequences = session.get_inputs()[0].name
  sequence_lengths = session.get_inputs()[1].name
  print("input_name_1: ", sequences)
  print("input_name_2: ", sequence_lengths)
  outputs = session.run(["memory", "processed_memory", "lens"],{sequences:input_data[0], sequence_lengths:input_data[1]})
  return outputs

def decoder_infer(decoder_inputs, decoder_model):
  session = onnxruntime.InferenceSession(decoder_model.SerializeToString())
  # 1. decoder_input
  # 2. attention_hidden
  # 3. attention_cell
  # 4. decoder_hidden
  # 5. decoder_cell
  # 6. attention_weights
  # 7. attention_weights_cum
  # 8. attention_context
  # 9. memory
  # 10. processed_memory
  # 11. mask
  input_names = []
  for input in session.get_inputs():
    input_names.append(input.name)

  output_names = []
  for output in session.get_outputs():
    output_names.append(output.name)

  # print("output names: ", output_names)

  inputs_dict = {}
  for i in range(11):
    inputs_dict[input_names[i]] = decoder_inputs[i]

  decoder_outputs = session.run([], inputs_dict)

  return decoder_outputs

def decoder_iter_infer(decoder_inputs, decoder_model, threshold = 0.5):
  count = 0
  mel_counts = [0] * decoder_inputs[0].shape[0]

  outputs = 0

  decoder_predicts = []

  gate_ones = np.ones(decoder_input.shape[0])

  while True:
    outputs = decoder_infer(decoder_inputs, decoder_model)
    assert(len(outputs) == 9)
    decoder_predicts.append(outputs[0])

    decoder_inputs[0] = outputs[0]
    for i in range(1, 8):
      decoder_inputs[i] = outputs[i + 1]

    gate = outputs[1]

    gate = 1/(1+np.exp(-gate)).flatten()

    gate = (gate < threshold).astype(np.int)
    gate_temp = gate_ones
    gate_ones = gate_ones * gate

    for i in range(len(gate_temp)):
      if gate_temp[i] == 1 and gate_ones[i] == 0:
        mel_counts[i] = count

    count += 1

    print("iter times : ", count)
    print("gate : ", gate_ones)
    print("The {} times gate is : {}".format(count, gate_ones))

    if gate_ones.sum() == 0:
      break

  return decoder_predicts, mel_counts

def postnet_infer(postnet_inputs, postnet_model):
  session = onnxruntime.InferenceSession(postnet_model.SerializeToString())
  mel_outputs = session.get_inputs()[0].name

  postnet_outputs = session.run([],{mel_outputs:postnet_inputs})
  return postnet_outputs

def waveglow_infer(waveglow_inputs, postnet_model):
  session = onnxruntime.InferenceSession(postnet_model.SerializeToString())
  mel = session.get_inputs()[0].name
  z = session.get_inputs()[1].name

  waveglow_outputs = session.run([], {mel : waveglow_inputs[0], z : waveglow_inputs[1]})
  return waveglow_outputs

def save_wav(wav, path, sr):
  wav = np.array(wav).astype(np.float)
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  wavfile.write(path, sr, wav.astype(np.int16))

def file_to_audio(audio_data, mel_len):
  sampling_rate = 22050
  result = audio_data.astype(np.float32)
  if mel_len > 0:
      # Cut to real mel_len.
      # For example, prediction report "60 gate = 0.998607" as latest, mel_len is 60
      result = result[0 : mel_len * 8 * 32]
  if len(result) > 0:
      save_wav(result, "tacotron2_onnx_models/mtn_test_batch8_process.wav", sampling_rate)

def change_model_input_batch(batch, model):
  names_to_change_list = ["sequences", "sequence_lengths",
                          "decoder_input", "attention_hidden", "attention_cell", "decoder_hidden",
                          "decoder_cell", "attention_weights", "attention_weights_cum", "attention_context",
                          "memory", "processed_memory", "mask",
                          "mel_outputs"
                         ]
  for input in model.graph.input:
    if input.name in names_to_change_list:
      dim_proto0 = input.type.tensor_type.shape.dim[0]
      dim_proto0.dim_value = batch

def change_model_output_batch(batch, model):
  names_to_change_list = ["memory", "processed_memory", "lens",
                          "decoder_output", "gate_prediction", "out_attention_hidden", "out_attention_cell",
                          "out_decoder_hidden", "out_decoder_cell", "out_attention_weights",
                          "out_attention_weights_cum", "out_attention_context",
                          "mel_outputs_postnet"
                          ]
  for output in model.graph.output:
    if output.name in names_to_change_list:
      dim_proto0 = output.type.tensor_type.shape.dim[0]
      dim_proto0.dim_value = batch

if __name__ == "__main__":
  # ============================= step 1 : input prepare ===============================
  words_index = []

  text = ["Bi Ren Technology is China's greatest chip company .",
          "Responsibility excellance collaboration innovation pragmatism empowering",
          "No boundaries, no challenges, pursuit excellence and Dare to be first",
          "Printing, in the only sense with which we are at present concerned",
          "differs from most if not from all the arts and crafts represented in the Exhibition",
          "in being comparatively modern.",
          "produced the block books, which were the immediate predecessors of the true printed book",
          "And it is worth mention in passing that, as an example of fine typography"
         ]

  for sentence in text:
    word_index_temp = text_to_sequence(sentence, ['english_cleaners'])
    words_index.append(word_index_temp)

  sentence_len = [len(item) for item in words_index]
  seq_len = max(sentence_len)

  for word_index in words_index:
    if len(word_index) < seq_len:
      word_index.extend([0]*(seq_len-len(word_index)))

  decoder_mask = []
  for length in sentence_len:
    mask_data = [0]*length + [1]*(seq_len - length)
    decoder_mask.append(mask_data)

  # root_path = "/home/mtn/suinfer_temp/"
  root_path = "/home/mtn/Projects/Onnx_process/tacotron2_onnx_models/"

  encoder_path = root_path + "encoder.onnx"
  decoder_path = root_path + "decoder_iter.onnx"
  postnet_path = root_path + "postnet.onnx"
  waveglow_path = root_path + "waveglow.onnx"

  # =============================step 2 : infer encoder ================================
  batch = len(words_index)

  encoder_model = onnx.load(encoder_path)

  change_model_input_batch(batch, encoder_model)
  change_model_output_batch(batch, encoder_model)

  sequence_data = np.array(words_index).astype(np.int64) # input1
  # sequence_lengths_data = np.array([seq_len] * batch).astype(np.int64) #input2
  sequence_lengths_data = np.array(sentence_len).astype(np.int64) #input2
  encoder_outputs =  encoder_infer([sequence_data, sequence_lengths_data], encoder_model) # infer

  #  =============================step 3 : infer decoder ==========================

  decoder_model = onnx.load(decoder_path)

  change_model_input_batch(batch, decoder_model)
  change_model_output_batch(batch, decoder_model)

  decoder_inputs = []

  # input 1
  decoder_input = np.array([0]*batch*80).astype(np.float16).reshape((batch, 80))
  decoder_inputs.append(decoder_input)
  # input 2
  attention_hidden = np.array([0]*batch*1024).astype(np.float16).reshape((batch, 1024))
  decoder_inputs.append(attention_hidden)
  # input 3
  attention_cell = np.array([0]*batch*1024).astype(np.float16).reshape((batch, 1024))
  decoder_inputs.append(attention_cell)
  # input 4
  decoder_hidden = np.array([0]*batch*1024).astype(np.float16).reshape((batch, 1024))
  decoder_inputs.append(decoder_hidden)
  # input 5
  decoder_cell = np.array([0]*batch*1024).astype(np.float16).reshape((batch, 1024))
  decoder_inputs.append(decoder_cell)
  # input 6
  attention_weights = np.array([0]*batch*seq_len).astype(np.float16).reshape((batch, seq_len))
  decoder_inputs.append(attention_weights)
  # input 7
  attention_weights_cum = np.array([0]*batch*seq_len).astype(np.float16).reshape((batch, seq_len))
  decoder_inputs.append(attention_weights_cum)
  # input 8
  attention_context = np.array([0]*batch*512).astype(np.float16).reshape((batch, 512))
  decoder_inputs.append(attention_context)
  # input 9
  memory = encoder_outputs[0] # form encpder:memory --> (batch, seq, 512)
  decoder_inputs.append(memory)
  # input 10
  processed_memory = encoder_outputs[1] # from encoder: processed_memory --> (batch, seq, 128)
  decoder_inputs.append(processed_memory)
  # input 11
  mask = np.array(decoder_mask).astype(np.bool)
  decoder_inputs.append(mask)

  # decoder inference
  decoder_outputs, mel_lens = decoder_iter_infer(decoder_inputs, decoder_model) # (batch, 80, mel_len) : mel_len is random

  decoder_outputs_unsequence = []
  for item in decoder_outputs:
    decoder_outputs_unsequence.append(item.reshape(batch, 80, 1))

  mel_outputs = np.concatenate(decoder_outputs_unsequence, axis=2)

  # ================================step 4 : infer postnet =================================

  postnet_model = onnx.load(postnet_path)

  change_model_input_batch(batch, postnet_model)
  change_model_output_batch(batch, postnet_model)

  postnet_outputs = postnet_infer(mel_outputs, postnet_model)

  # print(postnet_outputs[0].shape)

  # ================================= step 5 : infer waveglow ================================

  waveglow_model = onnx.load(waveglow_path)


  change_model_input_batch(batch, postnet_model)
  change_model_output_batch(batch, postnet_model)

  mel_size = postnet_outputs[0].shape[2] # 367
  stride = 256
  n_group = 8
  z_size = mel_size * stride # 367 * 256
  z_size = z_size // n_group # 367 * 256 / 32
  z = np.random.randn(batch, n_group, z_size).astype(np.float16)

  waveglow_outputs = waveglow_infer([postnet_outputs[0], z], waveglow_model)

  # ================================== step 6 : save to .wav ==================================
  mels_data = []
  for i in range(batch):
    mel_data = waveglow_outputs[0][i].flatten()
    mel_len = mel_lens[i]
    valid_mel_data = mel_data[0 : mel_len * 256]
    mels_data.extend(valid_mel_data.tolist())

  final_data = np.array(mels_data)
  file_to_audio(final_data, 0)

