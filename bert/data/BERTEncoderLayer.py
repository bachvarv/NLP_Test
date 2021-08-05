import tensorflow as tf
from transformer.data.EncoderLayer import EncoderLayer
from tensorflow.keras.layers import Dropout, Embedding, Layer


class BERTEncoderLayer(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, initializer_range=0.02, rate=0.1, layer_norm_eps=1e-12):
        super(BERTEncoderLayer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, self.d_model)

        self.segment_encoding = Embedding(2, self.d_model)

        self.pos_encoding = tf.Variable(
            initial_value=tf.random_normal_initializer(mean=1., stddev=initializer_range)(
                shape=[1, maximum_position_encoding, d_model]),
            trainable=True)

        self.dropout = Dropout(rate)

        self.encoder_layer = [EncoderLayer(self.d_model, num_heads, dff, rate, layer_norm_eps)
                              for _ in range(self.num_layers)]

    def call(self, x, segments, training, mask):
        x = self.embedding(x) + self.segment_encoding(segments)
        x += self.pos_encoding[:, :x.shape[1], :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, mask)

        return x