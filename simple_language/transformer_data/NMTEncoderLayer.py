from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention

from simple_language.transformer_data.NMTMultiHeadAttention import NMTMultiHeadAttention


class NMTEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, drop_rate, dff=1024):
        super(NMTEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.self_mha = NMTMultiHeadAttention(d_model, num_heads)
        self.bert_mha = NMTMultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.normalization_Layer = LayerNormalization(epsilon=1e6)
        self.normalization_Layer2 = LayerNormalization(epsilon=1e6)

        self.drop_layer1 = Dropout(drop_rate)
        self.drop_layer2 = Dropout(drop_rate)

    def call(self, inputs, bert_output, training, padding_mask=None):
        self_attention = self.self_mha(inputs, inputs, inputs, padding_mask)
        bert_enc_attention = self.bert_mha(inputs, bert_output, bert_output, padding_mask)
        attention_output = .5*(self_attention + bert_enc_attention)

        attention_output = self.drop_layer1(attention_output, training=training)
        out1 = self.normalization_Layer(attention_output + inputs)

        ffn_output = self.ffn(out1)
        ffn_output = self.drop_layer2(ffn_output, training=training)
        norm = self.normalization_Layer(ffn_output)

        return norm




    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return Sequential([Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                           Dense(d_model)  # (batch_size, seq_len, d_model)
                           ])