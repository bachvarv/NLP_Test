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
        self.w_i = keras.Input(shape=(1,), dtype=tf.int32)
        self.w_j = keras.Input(shape=(1,), dtype=tf.int32)
        self.y_true = keras.Input(shape=(1,), dtype=tf.int32)

        self.inp = keras.layers.Concatenate(name='conc_input')([self.w_i, self.w_j, self.y_true])


        # TODO: Look at https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs

        self.embedding_layer = GloveEmbeddingLayer(corpus_size, property_size)
        self.dot = keras.layers.Dot(axes=-1)
        # self.pred = keras.backend.sum(self.dot)



    def loss(self, y_true, y_pred):
        l = K.sum(K.pow((y_pred/self.count_max), self.scaling_factor))*K.square(y_pred - K.log(y_true))
        return l

    def call(self, input):
        print(input)
        x = self.embedding_layer(input[0], input[1])
        x = self.dot([x[0], x[1]])
        ip = tf.cast(input[2], tf.float32)
        x = self.loss(ip, x)
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