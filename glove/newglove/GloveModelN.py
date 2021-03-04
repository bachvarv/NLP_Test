from tensorflow.keras import Model
from keras.backend import mean
from keras.layers import Embedding, Dot, Flatten
from keras.backend import square, pow, log
from tensorflow import constant, GradientTape, cast, float64
from numpy import array

import tensorflow as tf


class GloveModelN(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModelN, self).__init__()
        self.x_max = constant(100, dtype=float64)
        self.factor = constant(3/4, dtype=float64)
        self.embedding_layer = Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=1,
                                  name="Glove_embedding", )

        self.context_layer = Embedding(vocab_size,
                                         embedding_dim,
                                         input_length=1,
                                         name="Glove_context_embedding", )

        self.dot = Dot(axes=(2, 2))
        self.flatten = Flatten()

    def loss(self, y_true, y_pred):
        # tf.print(y_true)
        # tf.print(y_pred)
        y_pred = cast(y_pred, dtype=float64)
        y_true = cast(y_true, dtype=float64)
        return pow((y_true/self.x_max), self.factor)*square((y_pred - log(y_true)))

    def call(self, inputs, training=None, mask=None):
        (target, context) = inputs
        # print(target, context)
        te = self.embedding_layer(target)
        ce = self.context_layer(context)
        # print(te)
        # print(ce)
        dot = self.dot([ce, te])
        # print(dot)
        fl = self.flatten(dot)
        # print(fl)
        return fl

    def train_step(self, data):

        # Unpack the data
        (ta, co), y_true = data

        with GradientTape() as tape:
            y_pred = self((ta, co), training=True)  # forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss(y_true, y_pred)
            # tf.print(loss)

            # Get the trainable variables
            trainable_vars = self.trainable_variables
            # Compute gradients
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics
            self.compiled_metrics.update_state(y_true, y_pred)
            # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def percentage_difference(y_true, y_pred):
        return mean(abs(y_pred/y_true - 1) * 100)

    def get_embedding_matrix(self):
        weights = array(self.embedding_layer.get_weights())
        return weights
