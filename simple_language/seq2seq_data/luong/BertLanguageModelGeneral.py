import tensorflow as tf
import transformers

from tensorflow.keras import Model

from data.ShapeChecker import ShapeChecker
from simple_language.seq2seq_data.SimpleLanguageDecoder import DecoderInput
from simple_language.seq2seq_data.luong.BERTLanguageDecoderGeneral import BertLanguageDecoderGeneral


class BertLanguageModelGeneral(Model):
    def __init__(self, d_model, vocab_size, tokenizer, path_to_model=None, bert_trainable=True):
        super(BertLanguageModelGeneral, self).__init__()
        self.d_model = d_model
        self.path_to_Bert_Model = path_to_model
        self.vocab_size = vocab_size
        self.encoder = transformers.TFBertModel.from_pretrained(self.path_to_Bert_Model, trainable=bert_trainable)
        # self.encoder = SimpleLanguageEncoder(self.path_to_Bert_Model, self.d_model)
        self.decoder = BertLanguageDecoderGeneral(self.vocab_size, self.d_model)
        self.tokenizer = tokenizer
        self.sequence_length = 128

        self.shape_checker = ShapeChecker()

    def call(self, inputs):

        inp, target_text = inputs

        (input_tokens, input_mask) = self._preprocess(inp)
        (target_tokens, target_mask) = self._preprocess(target_text)

        encoding = self.encoder(input_tokens)
        enc_output, state = encoding.last_hidden_state, encoding.pooler_output
        batch, length = tf.shape(enc_output)[0], tf.shape(enc_output)[1]
        # print(enc_output)
        # print(state)
        self.shape_checker(enc_output, ('batch_size', 'seq_length', 'd_model'))
        self.shape_checker(state, ('batch_size', 'd_model'))

        # if self.path_to_Bert_Model is not None:
        #     dec_input = DecoderInput(new_tokens=target_tokens['input_ids'],
        #                              enc_output=encoding,
        #                              mask=input_mask)
        # else:
        # self.decoder.attention.setup_memory(enc_output)
        dec_input = DecoderInput(new_tokens=target_tokens['input_ids'],
                                 enc_output=enc_output,
                                 mask=input_mask)
        dec_state = state
        dec_pred, dec_state = self.decoder(dec_input, dec_state)

        return dec_pred, dec_state

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

    def train_step(self, inputs):
        inp, tar = inputs


        # (input_tokens, input_mask) = self._preprocess(input_text)

        with tf.GradientTape() as tape:

            encoding = self.encoder(inp)
            enc_output, enc_state = encoding.last_hidden_state, encoding.pooler_output
            # print(enc_output)
            self.shape_checker(enc_output, ('batch_size', 'seq_length', 'd_model'))
            self.shape_checker(enc_state, ('batch_size', 'd_model'))
            # self.decoder.attention.setup_memory(enc_output)
            dec_state = enc_state
            loss = tf.constant(0.0)
            tokens, target_mask = self._preprocess(tar.numpy()[0].decode('utf-8'))
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

