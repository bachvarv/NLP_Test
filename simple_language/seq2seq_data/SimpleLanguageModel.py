import numpy as np
import tensorflow as tf

from tensorflow.keras import Model

from simple_language.seq2seq_data.bahdanau.ShapeChecker import ShapeChecker
from simple_language.seq2seq_data.SimpleLanguageEncoder import SimpleLanguageEncoder
from simple_language.seq2seq_data.SimpleLanguageDecoder import SimpleLanguageDecoder, DecoderInput


class SimpleLanguageModel(Model):
    def __init__(self, d_model, vocab_size, tokenizer):
        super(SimpleLanguageModel, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.encoder = SimpleLanguageEncoder(self.d_model, self.vocab_size)
        self.decoder = SimpleLanguageDecoder(self.vocab_size, self.d_model)
        self.tokenizer = tokenizer
        self.sequence_length = 128

        self.start_token = tokenizer.convert_tokens_to_ids('[CLS]')
        self.end_token = tokenizer.convert_tokens_to_ids(['[SEP]'])

        token_ids = tokenizer.convert_tokens_to_ids(['', '[UNK]', '[CLS]'])
        token_mask = np.zeros([vocab_size], dtype=bool)
        token_mask[token_ids] = True
        self.token_mask = token_mask

        self.shape_checker = ShapeChecker()


    def call(self, inputs, max_length=128):
        inp, target_text = inputs
        predicted_text = []
        done = tf.zeros([1, 1], dtype=tf.bool)

        (input_tokens, input_mask) = self._preprocess(inp)
        # (target_tokens, target_mask) = self._preprocess(target_text)
        target_tokens = self._preprocess_for_decoder(target_text)

        encoding, state = self.encoder(input_tokens)
        self.shape_checker(encoding, ('batch_size', 'seq_length', 'd_model'))
        self.shape_checker(state, ('batch_size', 'd_model'))
        dec_state = state

        for _ in range(max_length):

            dec_input = DecoderInput(#new_tokens=target_tokens['input_ids'],
                                     new_tokens=target_tokens,
                                     enc_output=encoding,
                                     mask=input_mask)

            dec_pred, dec_state = self.decoder(dec_input, dec_state)
            target_tokens = self._sample_token(dec_pred.logits)

            done = done | (target_tokens[0] in self.end_token)
            target_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), target_tokens)
            predicted_text.append(target_tokens[0])

            if tf.reduce_all(done):
                predicted_text=predicted_text[:-1]
                break

        text = tf.concat(predicted_text, axis=-1)
        text = self.tokenizer.convert_ids_to_tokens(text)
        return dec_pred, dec_state, text


    def _sample_token(self, logits):
        shape_checker = ShapeChecker()
        # 't' is usually 1 here.
        shape_checker(logits, ('batch', 't', 'vocab'))
        shape_checker(self.token_mask, ('vocab',))

        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

        logits = tf.where(token_mask, -np.inf, logits)

        new_token = tf.argmax(logits, axis=-1)

        return new_token

    def _preprocess(self, input_text):
        # self.shape_checker(input_text, ('batch_size'))
        # self.shape_checker(target_text, ('batch_size'))
        input_tokens = self.tokenizer(input_text,  max_length=self.sequence_length,
                                    padding='max_length', truncation=True, return_tensors='tf')
        self.shape_checker(input_tokens['input_ids'], ('batch_size', 'seq_length'))
        # target_tokens = self.tokenizer(input_text,  max_length=self.sequence_length,
        #                             padding='max_length', truncation=True, return_tensors='tf')
        # self.shape_checker(target_tokens['input_ids'], ('batch_size', 'seq_length'))

        input_mask = input_tokens['attention_mask']
        self.shape_checker(input_mask, ('batch_size', 'seq_length'))
        # target_mask = target_tokens['attention_mask']
        # self.shape_checker(target_mask, ('batch_size', 'seq_length'))

        return input_tokens, input_mask

    def _preprocess_for_decoder(self, target_text):
        target_tokens = tf.constant([self.tokenizer.convert_tokens_to_ids(target_text)])
        self.shape_checker(target_tokens, ('batch_size', 'tar_length'))

        return target_tokens

    def train_step(self, inputs):
        inp, tar = inputs


        # (input_tokens, input_mask) = self._preprocess(input_text)

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(inp)
            self.shape_checker(enc_output, ('batch_size', 'seq_length', 'd_model'))
            self.shape_checker(enc_state, ('batch_size', 'd_model'))

            dec_state = enc_state
            loss = tf.constant(0.0)
            tokens, target_mask = self._preprocess(tar.numpy()[0].decode('utf-8'))
            # tokens, target_mask = self._preprocess_for_decoder(tar.numpy()[0].decode('utf-8'))
            tokens = tokens['input_ids']
            max_length = tf.shape(tokens)[1]
            for t in tf.range(max_length-1):
                new_tokens = tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, inp['attention_mask'],
                                                       enc_output, dec_state)
                loss = loss + step_loss

            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

            # loss, dec_state = self._loop_step(tar, inp['attention_mask'],
            #                                   enc_output, dec_state)

        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'batch_loss': average_loss}

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        new_input, target = new_tokens[:, 0:1], new_tokens[:, 1:2]
        # print(text[:1])
        # input_token = tf.concat([tokens['input_ids'][:, :1], tf.zeros(shape=(tf.shape(tokens['input_ids'])[0], 127), dtype=tf.int32)], -1)
        # print(input_token.shape)

        # target_token = tf.concat([tokens['input_ids'][:, 1:], tf.zeros(shape=(tf.shape(tokens['input_ids'])[0], 1), dtype=tf.int32)], -1)
        # print(target_token.shape)
        # input_token = inp_text['input_ids'][:, 0:2]
        # target_token = inp_text['input_ids'][:, 2:]

        # print(input_token)
        # print(target_token)

        decoder_input = DecoderInput(new_tokens=new_input,
                                     enc_output=enc_output,
                                     mask=input_mask)
        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        self.shape_checker(dec_result.logits, ('batch_size', 'inp_seq', 'logits'))
        self.shape_checker(dec_result.attention_weights, ('batch', 'inp_seq', 'seq'))
        self.shape_checker(dec_state, ('batch', 'd_model'))

        y = target
        y_pred = dec_result.logits
        loss = self.loss(y, y_pred)
        return loss, dec_state
