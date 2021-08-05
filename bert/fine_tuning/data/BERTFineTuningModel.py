import os.path
import time

import numpy
import numpy as np
import tensorflow.keras.layers
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text


class BERTFineTuningModel(tf.keras.Model):
    def __init__(self, path_to_model, batch_size, max_sentence_size, target_vocab_size):
        super(BERTFineTuningModel, self).__init__()
        self.max_sentence_size = max_sentence_size
        self.batch_size = batch_size
        self.output_target_vocab_size = target_vocab_size
        # TODO: Change with the specific tokenizer
        # this is a multi_cased_preprocess/3
        # self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

        self.encoder_inputs = dict(
            input_word_ids=tf.keras.layers.Input(shape=(max_sentence_size), dtype=tf.int32),
            input_mask=tf.keras.layers.Input(shape=(max_sentence_size), dtype=tf.int32),
            input_type_ids=tf.keras.layers.Input(shape=(max_sentence_size), dtype=tf.int32),
        )

        self.encoder_model = hub.KerasLayer(path_to_model, trainable=True)

        self.encoder_output = self.encoder_model(self.encoder_inputs)

        self.dropout = tensorflow.keras.layers.Dropout(rate=0.1)
        self.dense = tf.keras.layers.Dense(self.output_target_vocab_size)
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
        # self.softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, inputs, training=None, mask=None):

        if inputs is None:
            return None
        x = self.encoder_model(inputs)
        # print(x)
        x = x['sequence_output']
        x = self.dropout(x)
        x = self.dense(x)
        # x = self.softmax(x)
        return x

    def train(self, inputs, lengths, epochs):
        start = time.time()
        for _ in range(epochs):
            for i in range(len(inputs)):

                # print(inputs[i])
                if len(inputs[i][0]) == 0 or len(inputs[i][1]) == 0:
                    continue
                x = self.call(inputs[i][0])

                if x is None:
                    continue

                label = inputs[i][1]['input_word_ids']
                # label = tf.cast(label_dict['input_word_ids'], dtype=tf.int64)


                # label = tf.keras.utils.to_categorical(label_dict['input_word_ids'],
                #                                       num_classes=self.output_target_vocab_size)
                # predicted = tf.keras.utils.to_categorical(predicted)
                # label = tf.keras.utils.to_categorical(label)
                # predicted = tf.cast(x, tf.float64)
                print(x)
                print(label)
                loss = self.loss(label, x)
                print(loss)

                with tf.GradientTape() as tape:
                    gradients = tape.gradient(loss,
                                              self.trainable_variables, unconnected_gradients='zero')
                    # print("Trainable: {}".format(self.trainable_variables))
                    # print(gradients)

                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                # print(loss)

        end = time.time()
        print("Time it took {}".format(end - start))
        return x

    def _pre_process_data(self, input):
        if self.batch_size == 1:
            array = input.numpy().astype(int)
            array_size = len(array)
            input_mask = np.zeros((array_size, self.max_sentence_size)).astype(int)
            processed_input = np.zeros((array_size, self.max_sentence_size)).astype(int)
            processed_input[:array_size, :array.shape[1], ] = array
            input_mask[:array_size, :array.shape[1], ] = np.ones(array.shape)
            input_type_ids = np.zeros((self.batch_size, self.max_sentence_size)).astype(int)
        else:
            array = input.numpy()
            array_size = len(array)
            input_mask = np.zeros((array_size, self.max_sentence_size)).astype(int)
            processed_input = np.zeros((array_size, self.max_sentence_size)).astype(int)
            input_type_ids = np.zeros((array_size, self.max_sentence_size)).astype(int)
            for i in range(len(array)):
                if len(array[i]) < self.max_sentence_size:
                    processed_input[i][:len(array[i]), ] = array[i]
                    input_mask[i] [:len(array[i]), ] = np.ones(array[i].shape)
                else:
                    return dict()

        return {'input_mask': tf.convert_to_tensor(input_mask),
                'input_type_ids':
                    tf.convert_to_tensor(input_type_ids),
                'input_word_ids':
                    tf.convert_to_tensor(processed_input)}
