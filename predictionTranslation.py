from keras.models import load_model
from util import *
filename="char2encoding.pkl"
sentence="I work"
#saveChar2encoding("char2encoding.pkl",input_token_index,16,71,reverse_target_char_index,num_decoder_tokens,target_token_index)
input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(filename)
encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,16,71)
encoder_model= load_model('encoder_modelPredTranslation.h5')
decoder_model= load_model('decoder_modelPredTranslation.h5')

input_seq = encoder_input_data

decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
print('-')
print('Input sentence:', sentence)
print('Decoded sentence:', decoded_sentence)
