import time
import timeit

import tensorflow as tf
from _cffi_backend import new_struct_type

from bert.data.BERTEncoderLayer import BERTEncoderLayer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam

from bert.data.MLMLayer import MLMLayer
import numpy as np

class BERTModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_length, maximum_positional_encoding, type_vocab_size=2,
                 pad_token_id=0, rate=0.1, layer_norm_eps=1e-12):
        super(BERTModel, self).__init__()
        self.encoding = BERTEncoderLayer(num_layers, d_model, num_heads,
                                         dff, input_vocab_length, maximum_positional_encoding,
                                         rate=rate, layer_norm_eps=layer_norm_eps)

        self.hidden = Dense(d_model, activation='tanh')

        self.nsp_layer = Dense(2)

        self.mlm_layer = MLMLayer(input_vocab_length, d_model)

        self.softmax = Softmax()

        self.loss = CategoricalCrossentropy()

        self.checkpoint_path = 'checkpoint/learning/'

    def call(self, token_ids, segments, pred_positions = None, training=True):
        x = self.encoding(token_ids, segments, training, None)

        if pred_positions is not None:
            mlm_input = x, pred_positions
            mlm_y_hat = self.mlm_layer(mlm_input)
            mlm_y_hat = self.softmax(mlm_y_hat)
            # mlm_y_hat = self.flatten(mlm_y_hat)
        else:
            mlm_y_hat = None

        x = self.hidden(x)
        nsp_y_hat = self.nsp_layer(x[:, 0, :])
        return x, mlm_y_hat, nsp_y_hat

    def train_step(self, dataset, epochs):
        start_time = time.time()
        batch = 0
        for i in range(epochs):

            for item in dataset:
                if (batch == 100):
                    break
                try:
                    tokens_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_labels = item
                    # print(mlm_labels)
                    with tf.GradientTape() as tape:
                        x, mlm_y_hat, nsp_y_hat = self.call(tokens_ids, segments, pred_positions)

                        nsp_real = self._generate_nsp_labels(nsp_labels)
                        nsp_loss = self.loss(nsp_real, nsp_y_hat)


                        if mlm_y_hat is not None:
                            # print(mlm_y_hat)
                            mlm_y_predicted = tf.argmax(mlm_y_hat, axis=2)
                            # print(mlm_y_predicted)
                            # mask = np.cast['float32'](mlm_weights)
                            mlm_y_predicted *= mlm_weights
                            mlm_y_predicted = tf.cast(mlm_y_predicted, tf.float32)

                            mlm_loss = self.loss(mlm_labels, mlm_y_predicted)
                            # print(mlm_loss)

                        # print(nsp_y_hat)

                        # nsp_loss = self.loss(nsp_labels, nsp_y_hat)

                    if mlm_loss is not None:
                        print(mlm_loss)
                        print(nsp_loss)
                        gradients = tape.gradient([mlm_loss, nsp_loss], self.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients,
                                                           self.trainable_variables))
                    else:

                        gradients = tape.gradient(nsp_loss, self.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                except StopIteration:
                    print("Epoch ended")
                    continue

                batch+=1
            self.save_weights(self.checkpoint_path + 'BERT_Save_Epoch_%n'.format(i) + time.strftime("%d_%m_%Y_%H:%M:%S",
                                                                                          time.gmtime()))

        end_time = time.time()
        print(end_time - start_time)
        return mlm_y_hat, nsp_y_hat

    def _generate_nsp_labels(self, nsp_labels):
        labels = []
        for i in nsp_labels:
            if i:
                labels.append([0, 1])
            else:
                labels.append([0, 2])

        return labels