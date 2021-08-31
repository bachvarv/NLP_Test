import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding

from transformer.data.DecoderLayer import DecoderLayer
from transformer.data.EncoderLayer import EncoderLayer
from transformer.data.MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np


class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = \
            self.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self, inp, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(inp)[1]
        attention_weights = {}

        inp = self.embedding(inp)  # (batch_size, target_seq_len, d_model)
        inp *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inp += self.pos_encoding[:, :seq_len, :]

        inp = self.dropout(inp, training=training)

        for i in range(self.num_layers):
            inp, block1, block2 = self.dec_layers[i](inp, enc_output, training,
                                                        look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # inp.shape == (batch_size, target_seq_len, d_model)

        return inp, attention_weights

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])

        # print("Positional Encoding befor:", angle_rads.shape)
        pos_encoding = angle_rads[np.newaxis, ...]
        # print("Positional Encoding after:", pos_encoding.shape)

        return pos_encoding

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
        return pos * angle_rates
