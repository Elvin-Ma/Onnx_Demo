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

  outputs = 0

  decoder_predicts = []

  while True:
    outputs = decoder_infer(decoder_inputs, decoder_model)
    assert(len(outputs) == 9)
    decoder_predicts.append(outputs[0])

    decoder_inputs[0] = outputs[0]
    for i in range(1, 8):
      decoder_inputs[i] = outputs[i + 1]

    gate = outputs[1][0][0]

    gate = 1/(1+np.exp(-gate))

    count += 1

    # print("The {} times gate is : {}".format(count, gate))

    if gate > threshold:
      break;

  return decoder_predicts

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
      save_wav(result, "tacotron2_onnx_models/mtn_test_2.wav", sampling_rate)

if __name__ == "__main__":

  # ============================= step 1 : input prepare ===============================
  # word_index = [52, 11, 41, 49, 55, 45, 60, 42, 52, 49, 49, 2]
  word_index_0 = [60, 42, 49, 40, 52, 50, 42, 11, 57, 52, 11, 39, 42, 55, 42, 51, 2]

  text = "Bi Ren Technology is China's greatest chip company ."
  word_index_1 = text_to_sequence(text, ['english_cleaners'])

  word_index = [word_index_1]

  # root_path = "/home/mtn/suinfer_temp/"
  root_path = "/home/mtn/Projects/Onnx_process/tacotron2_onnx_models/"

  encoder_path = root_path + "encoder.onnx"
  decoder_path = root_path + "decoder_iter.onnx"
  postnet_path = root_path + "postnet.onnx"
  waveglow_path = root_path + "waveglow.onnx"

  # =============================step 2 : infer encoder ================================

  encoder_model = onnx.load(encoder_path)

  sentence_len = [len(item) for item in word_index]

  seq_len = max(sentence_len)

  bacth = len(word_index)

  sequence_data = np.array(word_index).astype(np.int64)
  sequence_lengths_data = np.array([seq_len]).astype(np.int64)
  encoder_outputs =  encoder_infer([sequence_data, sequence_lengths_data], encoder_model)

  #  =============================step 3 : infer decoder ==========================

  decoder_model = onnx.load(decoder_path)

  decoder_inputs = []

  # input 1
  decoder_input = np.array([0]*80).astype(np.float16).reshape((bacth, 80))
  decoder_inputs.append(decoder_input)
  # input 2
  attention_hidden = np.array([0]*1024).astype(np.float16).reshape((bacth, 1024))
  decoder_inputs.append(attention_hidden)
  # input 3
  attention_cell = np.array([0]*1024).astype(np.float16).reshape((bacth, 1024))
  decoder_inputs.append(attention_cell)
  # input 4
  decoder_hidden = np.array([0]*1024).astype(np.float16).reshape((bacth, 1024))
  decoder_inputs.append(decoder_hidden)
  # input 5
  decoder_cell = np.array([0]*1024).astype(np.float16).reshape((bacth, 1024))
  decoder_inputs.append(decoder_cell)
  # input 6
  attention_weights = np.array([0]*seq_len).astype(np.float16).reshape((bacth, seq_len))
  decoder_inputs.append(attention_weights)
  # input 7
  attention_weights_cum = np.array([0]*seq_len).astype(np.float16).reshape((bacth, seq_len))
  decoder_inputs.append(attention_weights_cum)
  # input 8
  attention_context = np.array([0]*512).astype(np.float16).reshape((bacth, 512))
  decoder_inputs.append(attention_context)
  # input 9
  memory = encoder_outputs[0] # form encpder:memory --> (batch, seq, 512)
  decoder_inputs.append(memory)
  # input 10
  processed_memory = encoder_outputs[1] # from encoder: processed_memory --> (batch, seq, 128)
  decoder_inputs.append(processed_memory)
  # input 11
  mask = np.array([0]*seq_len).astype(np.bool).reshape((bacth, seq_len))
  decoder_inputs.append(mask)

  # decoder inference
  decoder_outputs = decoder_iter_infer(decoder_inputs, decoder_model) # (batch, 80, mel_len) : mel_len is random

  decoder_outputs_unsequence = []
  for item in decoder_outputs:
    decoder_outputs_unsequence.append(item.reshape(bacth, 80, 1))

  # decoder_iter_count = 96
  # batch, length =  decoder_outputs_unsequence[0].shape[0], decoder_outputs_unsequence[0].shape[1]
  # padding = [np.zeros((batch, length, 1), dtype=np.float16)]*(decoder_iter_count - len(decoder_outputs_unsequence))
  # decoder_outputs_unsequence.extend(padding)

  mel_outputs = np.concatenate(decoder_outputs_unsequence, axis=2)

  # ================================step 4 : infer postnet =================================

  postnet_model = onnx.load(postnet_path)

  postnet_outputs = postnet_infer(mel_outputs, postnet_model)

  # print(postnet_outputs[0].shape)

  # ================================= step 5 : infer waveglow ================================

  waveglow_model = onnx.load(waveglow_path)

  mel_size = postnet_outputs[0].shape[2]
  stride = 256
  n_group = 8
  z_size = mel_size * stride
  z_size = z_size // n_group # 取整
  z = np.random.randn(bacth, n_group, z_size).astype(np.float16)

  waveglow_outputs = waveglow_infer([postnet_outputs[0], z], waveglow_model)

  # ================================== step 6 : save to .wav ==================================
  file_to_audio(waveglow_outputs[0].flatten(), 0)


