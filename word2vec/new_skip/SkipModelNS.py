import tensorflow as tf
import numpy as np
import keras
from tensorflow.python.keras.layers import Embedding, Dot, Flatten


class SkipModelNS(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns=4):
        super(SkipModelNS, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=1,
                                  name="glove_embedding", )

        self.context_embedding = Embedding(vocab_size, embedding_dim,
                                           input_length=num_ns + 1,)

        self.dots = Dot(axes=(3, 2))

        self.flatten = Flatten()

    def call(self, input, **kwargs):
        target, context = input
        # print(target)
        # print(context)
        targets = self.target_embedding(target)
        contexts = self.target_embedding(context)
        d = self.dots([contexts, targets])
        fl = self.flatten(d)
        return fl

    def get_embedding_matrix(self):
        weights = np.array(self.target_embedding.get_weights())
        return weights

