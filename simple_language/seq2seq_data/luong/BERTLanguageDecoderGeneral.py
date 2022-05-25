import typing

from tensorflow.keras.layers import Layer, Embedding, GRU, Dense
from simple_language.seq2seq_data.luong.ShapeChecker import ShapeChecker
import tensorflow as tf

from typing import Any

from simple_language.seq2seq_data.luong.LuongAttention import LuongAttention


class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights:Any


# Using Luong's multiplicative attention
class BertLanguageDecoderGeneral(Layer):
    def __init__(self, vocab_size, d_model):
        super(BertLanguageDecoderGeneral, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # For Step 1. The embedding layer converts token IDs to vectors
        self.embedding = Embedding(self.vocab_size, d_model)

        # For Step 2. The RNN keeps track of wath's been generated so far.
        self.gru = GRU(d_model,
                       return_state=True,
                       return_sequences=True,
                       recurrent_initializer='glorot_uniform')

        # For Step 3. The RNN output will be the query for the attention layer
        # self.attention = BahdanauAttention(self.d_model)
        self.attention = LuongAttention(self.d_model)
        # For Step 4. converting `ct` to `at`
        self.Wc = Dense(self.d_model, activation=tf.math.tanh,
                        use_bias=False)

        # For Step 5. This fully connected layer produces the logits for each output token
        self.fc = Dense(self.vocab_size)

    def call(self,
             inputs: DecoderInput,
             state=None) -> typing.Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()
        # print('new', inputs.new_tokens)
        shape_checker(inputs.new_tokens, ('batch_size', 'inp_seq'))
        shape_checker(inputs.enc_output, ('batch_size', 'seq', 'd_model'))
        shape_checker(inputs.mask, ('batch_size', 'seq'))

        if state is not None:
            shape_checker(state, ('batch_size', 'd_model'))

        # Step 1. Lookup the embeddings
        emb_out = self.embedding(inputs.new_tokens)
        shape_checker(emb_out, ('batch_size', 'inp_seq', 'd_model'))

        # Step 2. Process the embedding vector
        rnn_output, state = self.gru(emb_out, initial_state=state)
        shape_checker(rnn_output, ('batch_size', 'inp_seq', 'd_model'))
        shape_checker(state, ('batch_size', 'd_model'))

        # Step 3. Use the RNN output as the query for the attention over the encoder output
        # context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output, mask=(inputs.mask != 0))
        # context_vector, attention_weights = self.attention([rnn_output, inputs.enc_output], mask=(inputs.mask != 0))
        context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output, mask=(inputs.mask != 0))
        # print(attention_weights)
        shape_checker(context_vector, ('batch_size', 'inp_seq', 'd_model'))
        shape_checker(attention_weights, ('batch_size', 'inp_seq', 'seq'))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #   [ct; ht] shape: (batch_size, inp_seq, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct;ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch_size', 'inp_seq', 'd_model'))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch_size', 'inp_seq', 'vocab_size'))
        return DecoderOutput(logits, attention_weights), state
