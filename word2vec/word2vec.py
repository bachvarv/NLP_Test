import tensorflow as tf
# import tensorflow.compat.v1 as tf
from data.EmbeddingLayer import EmbeddingLayer


class Word2Vec:

    def __init__(self, corpus_size, window_size):

        self.vector_dim = corpus_size
        self.properties = 10

        self.x = [tf.constant([1.0])] * corpus_size
        self.y_lab = [[tf.constant([1.0])] * corpus_size] * window_size

        self.model = tf.keras.Sequential()
        # # self.input_layer = tf.keras.layers.Dense(self.vector_dim, input_shape=(self.vector_dim,))
        # self.y_label = tf.keras.layers.Dense(self.vector_dim, input_shape=(window_size, self.vector_dim,))
        #
        # self.input_layer = tf.Variable(shape=(self.vector_dim,), trainable=False)
        # self.y_label = tf.Variable(
        #     shape=(window_size, self.vector_dim,), trainable=False)

        # self.inp_layer = tf.placeholder(tf.float32, shape=(None, self.vector_dim))

        self.model.add(EmbeddingLayer(corpus_size, self.properties))
        # self.model.add(self.input_layer)
        # self.model.add(self.hidden_layer)
        # self.model.add(self.output_layer)

        self.loss = tf.reduce_sum(tf.subtract(self.output_layer, self.y_lab))

        self.optimizer = tf.optimizers.Adam()
        # self.embedding = keras.layers.Dense(64, input_shape=(90,), name='embedding')
        # # self.embedding = keras.layers.Flatten(input_shape=(9, 10))
        # self.embedding(np.random.uniform(low=-1, high=1, size=(90,)))
        #
        # self.model = keras.Sequential([
        #     keras.layers.Input(shape=(9,), name='target'),
        #     self.embedding,
        #     keras.layers.Dense(20, activation='softmax', name='context')
        # ])
        #
        # self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=["accuracy"])