import tensorflow as tf
from timeit import default_timer as timer
from tensorflow import keras
from keras import backend as K


from glove.data.GloveEmbedding import GloveEmbeddingLayer


class GloVeModel(tf.keras.Model):
    def __init__(self, corpus_size, property_size):
        super(GloVeModel, self).__init__()
        self.count_max = tf.constant(5, dtype=tf.float32)
        self.scaling_factor = tf.constant(3 / 4, dtype=tf.float32)
        self.w_i = keras.Input(shape=(1,), dtype=tf.int32)
        self.w_j = keras.Input(shape=(1,), dtype=tf.int32)
        self.y_true = keras.Input(shape=(1,))

        self.embedding_layer = GloveEmbeddingLayer(corpus_size, property_size)
        self.dot = keras.layers.Dot(axes=-1)
        # self.pred = keras.backend.sum(self.dot)

        ''' Optimizer '''
        self.optimizer = keras.optimizers.SGD()

    def loss(self, y_pred, y_true):
        loss = K.sum(K.pow((y_true/self.count_max), self.scaling_factor))*K.square(y_pred - K.log(y_true))

        return loss

    def call(self, input):
        # x1 = self.w_i(input[0])
        # x2 = self.w_j(input[1])
        # y_true = self.y_true(input[2])
        vectors = self.embedding_layer(input[0], input[1])
        v0 = tf.reshape(vectors[0], (len(vectors[0]), 1))
        v1 = tf.reshape(vectors[1], (len(vectors[1]), 1))
        x = self.dot([v0, v1]),

        x = tf.add(tf.reduce_sum(x), vectors[2], vectors[3])

        # pred = tf.add(x, vectors[2], vectors[2])
        return x

    def train(self, input):
        with tf.GradientTape() as tape:
            prediction = self.call(input)

            loss = self.loss(prediction, tf.cast(input[2], tf.float32))

            gradients = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients((grad, var) for (grad, var)
                                           in zip(gradients, self.trainable_variables)
                                           if grad is not None)

        return loss, gradients

    def trainLoop(self, EPOCHS, matrix):
        start = timer()
        print("Initiating Training...")
        for _ in range(EPOCHS):
            for row in range(len(matrix)):
                for column in range(len(matrix)):
                    if (matrix[row, column] != 0.0):
                        loss, _ = self.train([row, column, matrix[row, column]])

        end = timer()
        print('Training Time: {}'.format(end - start))
        return self.embedding_layer.w.numpy(), self.embedding_layer.b.numpy()
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