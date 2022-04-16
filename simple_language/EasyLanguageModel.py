
import tensorflow as tf
import transformers
import transformers as ts


class EasyLanguageModel(tf.keras.Model):
    def __init__(self, path_to_embedding_model, cfg, *args, **kwargs):
        super(EasyLanguageModel, self).__init__(*args, **kwargs)
        self.embedding = transformers.TFBertModel.from_pretrained(path_to_embedding_model)
        self.lstm_layer = tf.keras.layers.LSTM(cfg['hidden_layer_size'], return_sequences=True)
        self.dense = tf.keras.layers.Dense(cfg['vocab_size'], activation='sigmoid')
        self.soft = tf.keras.layers.Softmax()

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = x[0]
        print(x.shape)
        x = self.lstm_layer(x)
        print(x)
        x = self.dense(x)
        x = self.soft(x)
        return x
