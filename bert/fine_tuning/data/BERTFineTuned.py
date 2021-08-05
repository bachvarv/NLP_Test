import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np


class BERTFineTuned(tf.keras.Model):
    def __init__(self, latest_checkpoint, *args, **kwargs):
        super(BERTFineTuned, self).__init__(*args, **kwargs)

        model_name = '.\\ted_hrlr_translate_pt_en_converter'
        # tf.keras.utils.get_file(f"{model_name}.zip",
        #                         f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        #                         cache_dir='.', cache_subdir='', extract=True)
        tokenizer = tf.saved_model.load(model_name)
        self.en_preprocessing = tokenizer.en
        self.pt_tokenizer = tokenizer.pt
        self.bert = self.create_bert_model(self.pt_tokenizer.get_vocab_size())
        self.softmax = tf.keras.layers.Softmax()

        self.checkpoint = tf.train.Checkpoint(self.bert)
        self.load_saved_mode(latest_checkpoint)

    def call(self, inputs, training=None, mask=None):
        x = self.en_preprocessing.tokenize(inputs)
        print(x)
        x = self.prepare_inputs(x)
        x = self.bert(x)
        x = self.softmax(x)
        prediction = tf.argmax(x, axis=2)
        print(prediction)
        return self.pt_tokenizer.detokenize(prediction)

    def load_saved_mode(self, path):
        self.bert.load_weights(path)

    @staticmethod
    def create_bert_model(vocab_size):
        pre_trained_model = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'

        inputs = dict(
            input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
            input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
            input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        )

        encoder_input = hub.KerasLayer(pre_trained_model, trainable=True, name="BERT_Encoder")

        outputs = encoder_input(inputs)

        net = outputs['sequence_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(vocab_size, activation=None)(net)

        model = tf.keras.Model(inputs, net)

        return model

    @staticmethod
    def prepare_inputs(tokenized):
        array = tokenized.numpy()
        array_size = len(array)
        print(array.shape)
        input_mask = np.zeros((array_size, 128)).astype(int)
        input_word_ids = np.zeros((array_size, 128)).astype(int)
        input_type_ids = np.zeros((array_size, 128)).astype(int)
        input_mask[:, :array.shape[1]] = np.ones(array.shape)
        input_word_ids[:, :array.shape[1]] = array

        return dict(input_mask=input_mask, input_type_ids=input_type_ids, input_word_ids=input_word_ids)
