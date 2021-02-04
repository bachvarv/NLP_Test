import numpy as np

from tensorflow.keras import Model
from keras.layers import Embedding, Dot, Flatten


class CbowModelNS(Model):

    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(CbowModelNS).__init__()

        self.embedding_layer = Embedding(vocab_size,
                                         embedding_dim,
                                         input_length=1,
                                         name="cbow_embedding")
        self.target_layer = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=num_ns + 1)

        self.dot = Dot(axes=(3,2))
        self.flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        context, target = inputs

        ce = self.embedding_layer(context)
        te = self.target_layer(target)
        dot = self.dot(te, ce)
        return self.flatten(dot)

    def get_embedding_matrix(self):
        weights = np.array(self.target_embedding.get_weights())
        return weights
