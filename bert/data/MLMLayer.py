import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras import Sequential


class MLMLayer(Layer):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MLMLayer, self).__init__(**kwargs)

        self.mlp = Sequential()
        self.mlp.add(Dense(num_hiddens, activation='relu'))
        self.mlp.add(LayerNormalization())
        self.mlp.add(Dense(vocab_size))

    def call(self, inputs):
        x, pred_pos = inputs
        num_pred_positions = pred_pos.shape[1]
        pred_pos = tf.reshape(pred_pos, [-1])
        batch_size = x.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = [[[x[batch_i][pos].numpy()] for pos in pred_pos] for batch_i in batch_idx ]
        masked_X = tf.reshape(masked_X, (batch_size, num_pred_positions, -1))
        mlm_Y = self.mlp(masked_X)
        return mlm_Y


