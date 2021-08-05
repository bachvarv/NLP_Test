import logging
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text

from transformer.data.Transformer import Transformer


def tokenize_pairs(pt, en):
    pt_token = tokenizers.pt.tokenize(pt)
    pt_token = pt_token.to_tensor()

    en_token = tokenizers.en.tokenize(en)
    en_token = en_token.to_tensor()

    return pt_token, en_token


def make_batches(ds):
    return (ds.cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def get_angles(pos, i, d_model):
    angle_rates = 1/ np.power(10000, 2*(i//2)/ np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding


def create_padding_mask(seq):
    # write a 1 when a value is 0 and 0 everywhere else
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask     # (seq_len, seq_len)


def scaled_dot_product_attention(q, k , v, mask):
    """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
      """
    matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)

    #scale mat_mul
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled attention_logits:
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v) # (..., seq_len_q, seq_len_v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


BUFFER_SIZE = 20000
BATCH_SIZE = 2
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
print(train_examples.batch(1).take(1))
# print(examples)

# The Interesting Part about this is that
# the pt_examples and en_examples after their initialization here
# they stay initialized and are callable later in the program
# Which is very bizarre
for pt_examples, en_examples in train_examples.batch(BATCH_SIZE).take(5):
    # for pt in pt_examples.numpy():
    #     print(pt.decode('utf-8'))

    # print()

    pt_examp = pt_examples
    en_examp = en_examples

    # for en in en_examples.numpy():
    #     print(en.decode('utf-8'))


model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(f"{model_name}.zip",
                        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
                        cache_dir='.', cache_subdir='', extract=True)
tokenizers = tf.saved_model.load(model_name)
encoded = tokenizers.en.tokenize(en_examples)
pt_encoded = tokenizers.pt.tokenize(pt_examples)

train_examples = train_examples.take(4)
# print(len(train_examples))
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# print(train_batches)
# print(val_batches)
# for row in encoded.to_list():
#     print(row)

round_trip = tokenizers.en.detokenize(encoded)
#
# for line in round_trip.numpy():
#     print(line.decode('utf-8'))

# testing positional encoding
n, d = 2048, 512

pos_encoding = positional_encoding(2048, 512)
# print(pos_encoding.shape)

# hyper-parameters for Transformer Model
EPOCHS = 1
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
input_vocab_size = tokenizers.en.get_vocab_size()
target_vocab_size = tokenizers.pt.get_vocab_size()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

transformer = Transformer(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          input_vocab_size,
                          target_vocab_size,
                          1000,
                          1000
                          )

transformer.compile(optimizer='adam',
                    loss=transformer.loss_object,
                    metrics=[transformer.loss_function,
                             transformer.accuracy_function])

#checkpoint Manager

checkpoint_path = './checkpoint/learning'
checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=transformer.optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

for i in range(EPOCHS):
    start = time.time()

    transformer.train_loss.reset_states()
    transformer.train_accuracy.reset_states()

    # inp -> english, tar -> portuguese
    for (batch, (tar, inp)) in enumerate(train_batches):
        # print("Detokenized Input:", tokenizers.en.detokenize(inp))
        # print("Tokenized Input:", inp)
        # print("Input:", tokenizers.en.detokenize(inp))
        prediction = transformer.train_step(inp, tar)

        if batch % 2 == 0:
            print(
                f'Epoch {i + 1} Batch {batch} Loss {transformer.train_loss.result():.4f} '
                f'Accuracy {transformer.train_accuracy.result():.4f}')

            checkpoint_manager.save()

    print(f'Epoch {i + 1} Loss {transformer.train_loss.result():.4f} Accuracy {transformer.train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

