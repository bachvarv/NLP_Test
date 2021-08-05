import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense
from tensorflow.python import _pywrap_tf_optimizer

from transformer.data.Decoder import Decoder
from transformer.data.Encoder import Encoder


class Transformer(Model):

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                               pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size,
                               pe_target)

        self.dense = Dense(target_vocab_size)

        self.loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                                         reduction='none')
        # checkpoint system

    def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_out, attention_weights = self.decoder(tar, enc_output, training,
                                                  look_ahead_mask, dec_padding_mask)

        output = self.dense(dec_out)

        return output, attention_weights

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def accuracy_function(self, real, pred):
        # print("Real target:", real)
        # print("Prediction:", tf.argmax(pred, axis=2))
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        # accuracies = tf.equal(real, pred)

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)

        dec_padding_mask = self.create_padding_mask(inp)

        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)

        combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        # write a 1 when a value is 0 and 0 everywhere else
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        return mask  # (seq_len, seq_len)

    # @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = \
            self.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            prediction, _ = self.call(inp, tar_inp, True,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask)

            print(prediction)
            print(tar_real)

            loss = self.loss_function(tar_real, prediction)
            predicted = tf.argmax(prediction, axis=2)

        gradients = tape.gradient(loss,
                                  self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients,
                                           self.trainable_variables))

        self.train_loss(loss)
        # print("Train accuracy:", self.accuracy_function(tar_real, prediction))
        self.train_accuracy(self.accuracy_function(tar_real, prediction))

        return predicted

    def check_checkpoint(self):
        return self.checkpoint_manager.latest_checkpoint

    def restore_checkpoint(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def save_checkpoint(self):
        self.checkpoint_manager.save()
