import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from tensorflow import keras
from keras import backend as K

from data.utils_functions import one_hot
from glove.data.GloveEmbeddingLayer import GloveEmbeddingLayer


class GloVeModelMultiInput(tf.keras.Model):
    def __init__(self, corpus_size, property_size):
        super(GloVeModelMultiInput, self).__init__()
        self.count_max = tf.constant([100], dtype=tf.float32)
        self.scaling_factor = tf.constant([3 / 4], dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.SGD()

        # Input
        self.w_i = keras.Input(shape=(1,), dtype=tf.int32)
        self.w_j = keras.Input(shape=(1,), dtype=tf.int32)
        self.y_true = keras.Input(shape=(1,), dtype=tf.int32)


        # TODO: Look at https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs

        self.embedding_layer = GloveEmbeddingLayer(corpus_size, property_size, 'One Layer')
        self.dot = keras.layers.Dot(axes=1)
        # self.pred = keras.backend.sum(self.dot)



    def loss(self, y_pred, y_true):
        # l = K.log(y_true)
        # print("L: ", l)
        l = K.pow(y_true/self.count_max, self.scaling_factor)*(K.square(y_pred - K.log(y_true)))
        return l

    def call(self, input):
        x = self.embedding_layer(input[0])
        x1 = tf.reshape(x[0], shape=(1, len(x[0])))
        x2 = tf.reshape(x[1], shape=(1, len(x[1])))
        x = self.dot([x1, x2])
        return x

    def train(self, input):
        with tf.GradientTape() as tape:
            prediction = self.call(input)
            loss = self.loss(prediction, tf.cast(input[1], tf.float32))
            gradients = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients((grad, var) for (grad, var)
                                           in zip(gradients, self.trainable_variables)
                                           if grad is not None)

        return prediction, gradients

    def f1(self, y_true):
        return np.power(y_true/self.count_max, self.scaling_factor)

    def trainLoop(self, EPOCHS, matrix):
        start = timer()
        print("Initiating Training...")
        size = len(matrix)
        for _ in range(EPOCHS):
            for row in range(len(matrix)):
                for column in range(len(matrix)):
                    if (matrix[row, column] != 0.0):
                        loss, _ = self.train([[one_hot(row, size), one_hot(column, size)], matrix[row, column]])
        end = timer()
        print('Training Time: {}'.format(end - start))
        return self.embedding_layer.w.numpy(), self.embedding_layer.b.numpy()