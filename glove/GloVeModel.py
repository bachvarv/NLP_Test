import tensorflow as tf
from tensorflow import keras
from keras import backend as K


from glove.data.GloveEmbedding import GloveEmbeddingLayer


class GloVeModel(tf.keras.Model):
    def __init__(self, corpus_size, property_size):
        super(GloVeModel, self).__init__()
        self.count_max = tf.constant([100], dtype=tf.float32)
        self.scaling_factor = tf.constant([3 / 4], dtype=tf.float32)
        self.w_i = keras.Input(shape=(1,), dtype=tf.int32)
        self.w_j = keras.Input(shape=(1,), dtype=tf.int32)
        self.y_true = keras.Input(shape=(1,))

        self.embedding_layer = GloveEmbeddingLayer(corpus_size, property_size)
        self.dot = keras.layers.Dot(axis=-1)
        self.pred = keras.backend.sum(self.dot)

    def loss(self, y_pred, y_true):
        loss = K.sum(K.pow((y_pred/self.count_max), self.scaling_factor))*K.square(y_pred - K.log(self.y_true))

        return loss

    def call(self, input):
        # x1 = self.w_i(input[0])
        # x2 = self.w_j(input[1])
        # y_true = self.y_true(input[2])
        vectors = self.embedding_layer(input[0], input[1])
        x = self.dot([tf.reshape(vectors[0], (len(vectors[0]), 1)), tf.reshape(vectors[1], (len(vectors[1]), 1))])
        print(x)
        x = tf.reduce_sum(x)
        print(x)
        # x = self.pred(x.numpy())
        loss = self.loss(x, vectors[2])

        # pred = tf.add(x, vectors[2], vectors[2])
        return x, loss



    # def train(self, input, epochs):
    #     '''
    #
    #     :param input: A Matrix, with all the
    #     :param epochs:
    #     :return:
    #     '''
    #
    #     # for i in range(len(input)):
    #     #     for j in range(len(input)):
    #     #         occ = input[i][j]
    #     #         if occ:
    #
    #
    #     return 1