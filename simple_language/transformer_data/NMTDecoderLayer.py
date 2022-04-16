from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention

from simple_language.transformer_data.NMTMultiHeadAttention import NMTMultiHeadAttention


class NMTDecoderLayer(Layer):
    def __init__(self, d_model, num_heads, drop_rate, dff=1024):
        super(NMTDecoderLayer, self).__init__()
        self.self_mha = NMTMultiHeadAttention(d_model, num_heads)
        self.bert_dec_mha = NMTMultiHeadAttention(d_model, num_heads)
        self.enc_dec_mha = NMTMultiHeadAttention(d_model, num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.norm_layer = LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = LayerNormalization(epsilon=1e-6)
        self.norm_layer3 = LayerNormalization(epsilon=1e-6)

        self.drop_layer1 = Dropout(drop_rate)
        self.drop_layer2 = Dropout(drop_rate)
        self.drop_layer3 = Dropout(drop_rate)

    def call(self, x, bert_enc, encoder_enc, training, look_ahead_mask=None, padding_mask=None):
        self_attention = self.self_mha(x, x, x, look_ahead_mask)
        self_attention = self.drop_layer1(self_attention, training=training)
        self_attention = self.norm_layer(x + self_attention)

        bert_dec_attention = self.bert_dec_mha(self_attention, bert_enc, bert_enc, padding_mask)
        enc_dec_attention = self.enc_dec_mha(self_attention, encoder_enc, encoder_enc, padding_mask)
        second_attention = 0.5*(bert_dec_attention + enc_dec_attention)
        norm_out = self.drop_layer2(second_attention, training=training)
        norm_out = self.norm_layer2(self_attention + norm_out)

        ffn_out = self.ffn(norm_out)
        ffn_out = self.drop_layer3(ffn_out, training=training)
        layer_output = self.norm_layer3(ffn_out + norm_out)


        return layer_output

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return Sequential([Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                           Dense(d_model)  # (batch_size, seq_len, d_model)
                           ])
