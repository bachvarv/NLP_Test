import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding

from transformer.data.EncoderLayer import EncoderLayer
from transformer.data.MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np


class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training, mask):
        seq_len = x.shape[1]

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        print()
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
        return pos * angle_rates

