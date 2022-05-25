import tensorflow as tf
from tensorflow.keras.layers import Layer, AdditiveAttention, Dense

from simple_language.seq2seq_data.bahdanau.ShapeChecker import ShapeChecker


class BahdanauAttention(Layer):
    def __init__(self, d_model):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(d_model)    # Query
        self.W2 = Dense(d_model)    # Value

        self.attention = AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'd_model'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'd_model'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(inputs=[w1_query, value, w2_key],
                                                           mask=[query_mask, value_mask],
                                                           return_attention_scores=True)
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights
