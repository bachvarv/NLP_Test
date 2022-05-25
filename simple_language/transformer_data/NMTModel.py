import time

import tensorflow as tf
import transformers

from simple_language.transformer_data.NMTDecoderLayer import NMTDecoderLayer
from simple_language.transformer_data.NMTEncoderLayer import NMTEncoderLayer
from tensorflow.keras.layers import Embedding

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class NMTModel(tf.keras.Model):
    def __init__(self, model_name, hidden_layer_size, transformer_heads, seq_len, vocab_size):
        super(NMTModel, self).__init__()
        self.heads = transformer_heads
        self.bert_layer = transformers.TFBertModel.from_pretrained(model_name)
        self.enc_embedding_layer = Embedding(vocab_size, hidden_layer_size, input_length=seq_len)

        self.encoder_layers = [NMTEncoderLayer(hidden_layer_size,
                                               transformer_heads,
                                               1e-3) for _ in range(transformer_heads)]

        self.decoder_emb = Embedding(vocab_size, hidden_layer_size, input_length=seq_len)
        self.decoder_layers = [NMTDecoderLayer(hidden_layer_size,
                                               transformer_heads,
                                               1e-3) for _ in range(transformer_heads)]

        self.dense_layer = tf.keras.layers.Dense(vocab_size)
        self.softmax_layer = tf.keras.layers.Softmax()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs, training):
        inp, target = inputs
        # print(enc_self_att)
        # print(inp['attention_mask'])
        enc_padding_mask, dec_padding_mask, look_ahead_mask = self.create_masks(inp['input_ids'], target)

        enc_out = self.bert_layer(inp).last_hidden_state
        # print(enc_out.shape)
        enc_self_att = self.encoder_layers[0](enc_out, enc_out, training, enc_padding_mask)
        for i in range(1, self.heads):
            enc_self_att = self.encoder_layers[i](enc_self_att, enc_out, training, enc_padding_mask)

        target_enc = self.decoder_emb(target)
        target_enc = self.decoder_layers[0](target_enc, enc_out, enc_self_att, training, look_ahead_mask, dec_padding_mask)
        for i in range(self.heads):
            target_enc = self.decoder_layers[i](target_enc, enc_out, enc_self_att, training, look_ahead_mask, dec_padding_mask)

        output_dense = self.dense_layer(target_enc)
        output = self.softmax_layer(output_dense)

        new_token = tf.argmax(output, axis=-1)
        return new_token, output

    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)

        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, dec_padding_mask, look_ahead_mask

    def train_step(self, dataset, epochs=5):
        losses = list()
        start = time.time()
        for i in range(epochs):
            epoch_start = time.time()
            for step, element in enumerate(dataset):
                _, target = element
                print(element)
                with tf.GradientTape() as tape:
                    _, prediction = self.call(element, True)

                    loss = self.loss(target, prediction)
                    losses.append(loss)
                grads = tape.gradient(loss, self.trainable_weights)

                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # Log every 200 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * 1))
            end = time.time()
            print('Epoch #%d' % (i+1))
            print('Execution time was %d s' % (end - epoch_start))
        print('Execution time was %d s' % (end - start))
        return losses


