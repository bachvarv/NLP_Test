import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from transformer.data.MultiHeadAttentionLayer import MultiHeadAttentionLayer


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, layer_norm_eps=1e-6):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layer_norm2 = LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layer_norm2(out1 + ffn_output)

        return out2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return Sequential([Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                           Dense(d_model)  # (batch_size, seq_len, d_model)
                           ])
