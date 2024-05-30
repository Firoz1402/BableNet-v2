import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle

# Load the models
encoder_model = load_model('../models/english_to_hindi_translator/encoder_model_e2h.h5')
decoder_model = load_model('../models/english_to_hindi_translator/decoder_model_e2h.h5')

# Load the tokenizers
with open('../models/english_to_hindi_translator/english_tokenizer_e2h.pkl', 'rb') as f:
    input_token_index = pickle.load(f)

with open('../models/english_to_hindi_translator/hindi_tokenizer_e2h.pkl', 'rb') as f:
    target_token_index = pickle.load(f)

with open('../models/english_to_hindi_translator/reverse_english_tokenizer_e2h.pkl', 'rb') as f:
    reverse_input_char_index = pickle.load(f)

with open('../models/english_to_hindi_translator/reverse_hindi_tokenizer_e2h.pkl', 'rb') as f:
    reverse_target_char_index = pickle.load(f)

max_length_src = 20  # Set this to the maximum length used during training
latent_dim = 300
num_decoder_tokens = len(target_token_index) + 1

def translate_sentence(input_sentence):
    input_seq = [input_token_index.get(word, 0) for word in input_sentence.split()]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_length_src, padding='post')

    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, '')

        # Exit condition: either hit max length or find stop character
        if sampled_char == '_END' or len(decoded_sentence) > max_length_src:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_char)

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return ' '.join(decoded_sentence)
