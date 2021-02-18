from tensorflow.keras import Model
from keras.layers import Embedding, Dot
from keras.backend import square, pow, log
from tensorflow import constant


class GloveModelN(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.x_max = constant(100)
        self.factor = constant(3/4)
        self.embedding_layer = Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=1,
                                  name="skip_embedding", )

        self.dot = Dot()

    def loss(self, y_pred, y_true):
        return pow((y_true/self.x_max), self.factor)*square((y_pred - log(y_true)))

    def call(self, inputs, training=None, mask=None):
        (target, context) = inputs
        te = self.embedding_layer(target)
        ce = self.embedding_layer(context)
        dot = self.dot([ce, te])
        return dot
