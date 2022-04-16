import transformers
import tensorflow as tf
from tensorflow.keras.layers import Layer, GRU, Embedding

from data.ShapeChecker import ShapeChecker


class SimpleLanguageEncoder(Layer):
    def __init__(self, enc_units, vocab_size):
        super(SimpleLanguageEncoder, self).__init__()
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding = Embedding(self.vocab_size, enc_units)

        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):

        shape_checker = ShapeChecker()
        # print(tokens['input_ids'])
        shape_checker(tokens['input_ids'], ('batch', 'seq'))

        # 2.    The embedding layer looks up the embedding vector for the each token
        emb_out = self.embedding(tokens['input_ids'])
        # print(emb_out)
        shape_checker(emb_out, ('batch', 'seq', 'embed_dim'))

        # 3.    The GRU processes the vectors,
        #       output shape: (batch, seq, enc_units)
        #       state shape: (batch, enc_units)
        output, state = self.gru(emb_out, initial_state=state)
        # shape_checker(output, ('batch', 'seq', 'enc_units'))
        # shape_checker(state, ('batch', 'enc_units'))

        # 4.    Returns the new sequence and state
        return output, state
