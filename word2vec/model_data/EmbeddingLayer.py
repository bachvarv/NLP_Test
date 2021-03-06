import tensorflow as tf
from tensorflow import keras

class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, corpus_size, prop_size, ind):
        super(EmbeddingLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(corpus_size, prop_size)),
                             trainable=True, name="W"+ind)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(prop_size)),
                             trainable=True, name="b" + ind)

    def call(self, input):
        # result = tf.tensordot(input, self.w, axes=0) + self.b
        # return tf.reduce_sum(result, axis=0, keepdims=True)
        return tf.tensordot(input, self.w, axes=1) + self.b


# x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# print(x)
# layer = EmbeddingLayer(len(x[0]), 10)
# print(layer.w)
# y = layer(x.transpose())
# print(y)