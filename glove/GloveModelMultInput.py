import tensorflow as tf
from tensorflow import keras
from keras import backend as K


from glove.data.GloveEmbedding import GloveEmbeddingLayer


class GloVeModelMultiInput(tf.keras.Model):
    def __init__(self, corpus_size, property_size):
        super(GloVeModelMultiInput, self).__init__()
        self.count_max = tf.constant([100], dtype=tf.float32)
        self.scaling_factor = tf.constant([3 / 4], dtype=tf.float32)

        # Input
        self.w_i = keras.Input(shape=(1,))
        self.w_j = keras.Input(shape=(1,))
        self.y_true = keras.Input(shape=(1,))

        self.inp = keras.layers.Concatenate(axis=1, name='conc_input')([self.w_i, self.w_j, self.y_true])


        # TODO: Look at https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs

        self.embedding_layer = GloveEmbeddingLayer(corpus_size, property_size)(self.inp[0], self.inp[1])
        self.dot = keras.layers.Dot(axis=-1)(self.embedding_layer[0], self.embedding_layer[1])
        # self.pred = keras.backend.sum(self.dot)

    def loss(self, y_pred, y_true):
        loss = K.sum(K.pow((y_pred/self.count_max), self.scaling_factor))*K.square(y_pred - K.log(self.y_true))

        return loss

    def call(self, input):
        # x1 = self.w_i(input[0])
        # x2 = self.w_j(input[1])
        # y_true = self.y_true(input[2])
        # vectors = self.embedding_layer(input[0], input[1])
        # x = self.dot([tf.reshape(vectors[0], (len(vectors[0]), 1)), tf.reshape(vectors[1], (len(vectors[1]), 1))])
        # x = tf.reduce_sum(x)
        # loss = self.loss(x, vectors[2])
        x = self.input(input)
        print(x)
        # pred = tf.add(x, vectors[2], vectors[2])
        return x



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