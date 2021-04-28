import tensorflow as tf
from keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras import Sequential

from transformer.data.MultiHeadAttentionLayer import MultiHeadAttentionLayer


class DecoderLayer(Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mmha1 = MultiHeadAttentionLayer(d_model, num_heads)
        self.mha2 = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.norm_layer1 = LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = LayerNormalization(epsilon=1e-6)
        self.norm_layer3 = LayerNormalization(epsilon=1e-6)

        self.drop_layer1 = Dropout(rate)
        self.drop_layer2 = Dropout(rate)
        self.drop_layer3 = Dropout(rate)

    def call(self, x, enc_out, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_1 = self.mmha1(x, x, x, look_ahead_mask)   # (batch_size, input_seq_len_q, d_model)
        attn1 = self.drop_layer1(attn1, training=training)
        out1 = self.norm_layer1(attn1 + x)

        attn2, attn_weights_2 = self.mha2(enc_out, enc_out, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.drop_layer2(attn2, training=training)
        out2 = self.norm_layer2(attn2 + out1) # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.drop_layer3(ffn_output, training=training)
        out3 = self.norm_layer3(ffn_output + out2)

        return out3, attn_weights_1, attn_weights_2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return Sequential([Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                           Dense(d_model)  # (batch_size, seq_len, d_model)
                           ])