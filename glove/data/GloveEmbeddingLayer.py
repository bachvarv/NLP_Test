import keras
import tensorflow as tf

class GloveEmbeddingLayer(keras.layers.Layer):
    def __init__(self, corpus_size, prop_size, ind):
        super(GloveEmbeddingLayer, self).__init__()
        w_init = tf.random_normal_initializer()

        self.corpus_size = corpus_size
        self.prop_size = prop_size

        self.w = tf.Variable(initial_value=w_init(shape=(corpus_size, prop_size)),
                             trainable=True, name="W"+ind)
        self.corpus_size = corpus_size
        self.prop_size = prop_size

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(prop_size)),
                             trainable=True, name="b" + ind)

    def call(self, i):
        # result = tf.tensordot(input, self.w, axes=0) + self.b
        # return tf.reduce_sum(result, axis=0, keepdims=True)
        return tf.tensordot(i, self.w, axes=1) + self.b