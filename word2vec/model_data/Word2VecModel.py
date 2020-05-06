from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.EmbeddingLayer import EmbeddingLayer


class Word2VecModel(keras.Model):
    def __init__(self, corpus_size, prop_size):
        super(Word2VecModel, self).__init__()
        # self.input_tensor = keras.Input(shape=(None, corpus_size), name='word')
        #
        self.input_layer = keras.layers.Dense(corpus_size)
        self.embed_layer = EmbeddingLayer(corpus_size, prop_size, "0")
        self.outp_layer = EmbeddingLayer(prop_size, corpus_size, "1")

        # print(self.embed_layer.trainable_weights)

        # self.optimizer = keras.optimizers.Adam()
        self.optimizer = keras.optimizers.SGD()

        # self.add(self.input_layer)
        # self.add(self.embed_layer)
        # self.add(self.outp_layer)
        # self.loss_func = keras.losses.mean_squared_error()

    def call(self, input_x):
        # x = self.input_tensor(input_x)
        # x = self.input_layer(x)
        x = self.embed_layer(input_x)
        x = tf.transpose(x)
        x = self.outp_layer(x)
        x = tf.nn.softmax(x)
        # x = tf.reduce_sum(tf.subtract(x, input_y), axis=0)
        return x

    def loss(self, y_pred, y):
        # y_ = self.call(x)
        # return tf.subtract(y_pred, y)
        # return keras.losses.binary_crossentropy(y_true=y, y_pred=y_pred)
        return keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)

    def grad(self, y_pred, y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(y_pred, y)
            return loss_value, tape.gradient(loss_value, self.trainable_variables)

    @tf.function
    def train(self, input, label):
        with tf.GradientTape() as tape:
            current_loss = self.loss(self.call(input), label)
            grads = tape.gradient(current_loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return grads

    def train_loop(self, input, label, r):
        start = timer()
        train_size = len(input)
        for _ in range(r):
            for i in np.random.permutation(train_size):
                self.train(input[i].transpose(), label[i])

        np.savetxt('test.out', self.embed_layer.w.numpy())
        end = timer()
        print('Training Time: {}'.format(end - start))