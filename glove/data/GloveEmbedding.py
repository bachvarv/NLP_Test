from builtins import super

from tensorflow import keras
import numpy as np
import tensorflow as tf

# tf.executing_eagerly()


class GloveEmbeddingLayer(keras.layers.Layer):
    def __init__(self, corpus_size, prop_size):
        super(GloveEmbeddingLayer, self).__init__()
        self.corpus_size = corpus_size
        self.prop_size = prop_size
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(corpus_size, prop_size)),
                             trainable=True, name="W", )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(corpus_size)),
                             trainable=True, name="b")

    def call(self, i, j, **kwargs):
        print(i, j)
        w_i = tf.nn.embedding_lookup(self.w, i, name="w_0")
        w_j = tf.nn.embedding_lookup(self.w, j, name="w_1")
        b_i = tf.nn.embedding_lookup(self.b, i, name="b_0")
        b_j = tf.nn.embedding_lookup(self.b, j, name="b_1")
        # print(w_i, w_j)
        s = [ w_i, w_j, b_i, b_j]
        return s

    def compute_output_shape(self):
        return [(self.prop_size, 1), (self.prop_size, 1), (self.corpus_size, 1), (self.corpus_size, 1)]#, (1, self.corpus_size)]


emb = GloveEmbeddingLayer(5, 5)
# [(w_i, w_j), (b_i, b_j)] = emb([1, 2])
x = emb(tf.constant(1), tf.constant(2))


print(x[0])