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
        print(target_vocab_size)
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
        self.dense = tf.keras.layers.Dense(self.output_target_vocab_size, activation=None)
        self.accuracy_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Definition of the model
        net = self.encoder_output['sequence_output']
        net = self.dropout(net)
        self.output_layer = self.dense(net)
        self.model = tf.keras.Model(self.encoder_inputs, self.output_layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.accuracy_metric])

        # self.softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, inputs, training=None, mask=None):
        if inputs is None:
            return None

        inp, out = inputs
        output = self.model(inp)

        # x = self.encoder_model(inputs)
        # x = x['sequence_output']
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = self.softmax(x)
        return output

    def train(self, inputs, epochs):
        start = time.time()
        self.model.fit(x=inputs[0], y=inputs[1], batch_size=1, epochs=1)
        # for _ in range(epochs):

            # for i in range(len(inputs)):
            #
            #     if len(inputs[i][0]) == 0 or len(inputs[i][1]) == 0:
            #         continue

                #
                # x = self.call(inputs[i])
                #
                # if x is None:
                #     continue
                #
                # label = inputs[i][1]
                # label = tf.cast(label_dict['input_word_ids'], dtype=tf.int64)


                # label = tf.keras.utils.to_categorical(label_dict['input_word_ids'],
                #                                       num_classes=self.output_target_vocab_size)
                # predicted = tf.keras.utils.to_categorical(predicted)
                # label = tf.keras.utils.to_categorical(label)
                # predicted = tf.cast(x, tf.float64)
                # print(x)
                # print(label)
                # loss = self.model.loss(label, x)
                # print(loss)
                # with tf.GradientTape() as tape:
                    # gradients = tape.gradient(loss,
                    #                           self.trainable_variables)
                    # gradients = tape.gradient(loss, self.model.trainable_variables)
                    # print(gradients_encoder)
                    # print("Trainable: {}".format(self.trainable_variables))
                    # print("Gradients")
                    # print(gradients)

                # self.model.optimizer.apply_gradients(zip(gradients,
                #                                          self.model.trainable_variables))
                # print(loss)

        end = time.time()
        print("Time it took {}".format(end - start))
        return 'Finished'

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
