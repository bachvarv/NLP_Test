from keras.layers import TextVectorization

import platform
import os
from bert.fine_tuning.data.BERTFineTuningModel import BERTFineTuningModel
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from official.nlp import optimization, bert
import official.nlp.bert.tokenization
import tensorflow_hub as hub

def pre_process_input(input, batch_size, max_sentence_size):
    if batch_size == 1:
        array = input.numpy().astype(int)
        array_size = len(array)
        input_mask = np.zeros((array_size, max_sentence_size)).astype(int)
        processed_input = np.zeros((array_size, max_sentence_size)).astype(int)
        processed_input[:array_size, :array.shape[1], ] = array
        input_mask[:array_size, :array.shape[1], ] = np.ones(array.shape)
        input_type_ids = np.zeros((batch_size, max_sentence_size)).astype(int)
    else:
        # array = input.numpy()
        array = input
        array_size = len(array)
        input_mask = np.zeros((array_size, max_sentence_size)).astype(int)
        processed_input = np.zeros((array_size, max_sentence_size)).astype(int)
        input_type_ids = np.zeros((array_size, max_sentence_size)).astype(int)
        for i in range(len(array)):
            if len(array[i]) < max_sentence_size:
                processed_input[i][:len(array[i]), ] = array[i]
                input_mask[i][:len(array[i]), ] = np.ones(array[i].shape)
            else:
                return None, None, None

    return input_mask, input_type_ids, processed_input


def encode_sentence(s):

    tokenized_sentences = list(act_tokenizer.tokenize(s.numpy()[0]))
    tokenized_sentences.insert(0, '[CLS]')
    tokenized_sentences.append('[SEP]')
    ids = act_tokenizer.convert_tokens_to_ids(tokenized_sentences)
    return ids

#TODO: make a better preprocessing of the dataset
pre_trained_model = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
sequence_size = 128
batch_size = 2

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)


train_examples, val_examples = examples['train'], examples['validation']

if(platform.system() == 'Linux'):
    model_checkpoint = 'checkpoint/bert_multi_cased_L-12_H-768_A-12_4'
    vocab_path = os.path.join(model_checkpoint, "assets/vocab.txt")
else:
    model_checkpoint = 'E:\\Masterarbeit\\pre-trained bert\\bert_multi_cased_L-12_H-768_A-12_4'
    vocab_path = os.path.join(model_checkpoint, "assets\\vocab.txt")

# TODO: new Tokenizer remove the old one
act_tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)


# train_pt_examples, train_en_examples = train_examples.batch(30000).take(5)
#
# print(train_pt_examples[0][0])
# print(train_en_examples[0][0])

# preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

cls = act_tokenizer.convert_tokens_to_ids(['[CLS]'])
train_inp_ex = []
train_label_ex = []
over = False
for train_pt_examples, train_en_examples in train_examples.batch(1):
    # train_en_examples = train_en_examples.numpy()
    # train_pt_examples = train_pt_examples.numpy()
    for example in train_pt_examples:

        one_sent_pt = len(example.numpy())
        if one_sent_pt > sequence_size:
            over = True
            break
    for example in train_en_examples:
        one_sent_en = len(example.numpy())
        if one_sent_en > sequence_size:
            over = True
            break

    if over:
        over = False
        continue
    # TODO: implement the act_tokenizer
    train_en_example = encode_sentence(train_en_examples)
    train_pt_example = encode_sentence(train_pt_examples)
    train_inp_ex.append(train_en_example)
    train_label_ex.append(train_pt_example)
    # train_en_examples = act_tokenizer.tokenize(train_en_examples.numpy())
    # train_pt_examples = act_tokenizer.tokenize(train_pt_examples.numpy())
    # train_en_examples = act_tokenizer.convert_tokens_to_ids(train_en_examples)
    # train_pt_examples = act_tokenizer.convert_tokens_to_ids(train_pt_examples)
    # input_mask, input_type_ids, input_word_ids = pre_process_input(train_en_example, batch_size, sequence_size)


inp_data = []
lab_data = []
input_word_ids = []
input_type_ids = []
input_mask = []
for i in range(len(train_inp_ex)):
    inp_length = len(train_inp_ex[i])
    lab_length = len(train_label_ex[i])
    if inp_length > sequence_size or lab_length > sequence_size:
        continue

    input_word_ids.append(np.concatenate(
        (train_inp_ex[i], np.zeros(sequence_size - inp_length, dtype='int'))
    ))
    input_type_ids.append(np.zeros(sequence_size, dtype='int'))
    input_mask.append(np.concatenate(
        (np.ones_like(train_inp_ex[i]), np.zeros(sequence_size - inp_length, dtype='int'))
    ))

    lab_data.append(np.concatenate(
        (train_label_ex[i], np.zeros(sequence_size - inp_length, dtype='int'))
    ))

lab_data = tf.ragged.constant(lab_data).to_tensor()
input_mask = tf.ragged.constant(input_mask).to_tensor()
input_word_ids = tf.ragged.constant(input_mask).to_tensor()
input_type_ids = tf.ragged.constant(input_type_ids).to_tensor()

print(f'Input shape: {input_word_ids.shape}')
print(f'Label shape: {lab_data.shape}')

inp_data = dict(
    input_word_ids=input_word_ids,
    input_type_ids=input_type_ids,
    input_mask=input_mask
)

data = (inp_data, lab_data)

# This way doesn't work
# en_examples = []
# pt_examples = []
#
# for label, input in train_examples.batch(1):
#     label = list(act_tokenizer.tokenize(label.numpy()[0]))
#     input = list(act_tokenizer.tokenize(input.numpy()[0]))
#     one_sent = len(label)
#     if one_sent > sequence_size:
#         break
#
#     one_sent = len(input)
#     if one_sent > sequence_size:
#         break
#     label.insert( 0, '[CLS]')
#     label.append('[SEP]')
#     input.insert(0, '[CLS]')
#     input.append('[SEP]')
#     print(input)
#     print(label)
#
#     label = act_tokenizer.convert_tokens_to_ids(label)
#     input = act_tokenizer.convert_tokens_to_ids(input)
#
#     en_examples.append(input)
#     pt_examples.append(label)
#
# input_word_ids = []
# input_mask = []
# input_type_ids = []
# labels = []
# for i in range(len(en_examples)):
#     en_length = len(en_examples[i])
#     pt_length = len(pt_examples[i])
#     if en_length > sequence_size or pt_length > sequence_size:
#         continue
#
#     input_word_ids.append(np.concatenate(
#         (en_examples[i], np.zeros((sequence_size - en_length), dtype='int'))
#     ))
#     input_mask.append(np.concatenate(
#         (np.ones(en_length, dtype='int'), np.zeros((sequence_size - en_length), dtype='int'))
#     ))
#     input_type_ids.append(np.zeros(sequence_size, dtype='int'))
#
#     labels.append(np.concatenate(
#         (pt_examples[i], np.zeros((sequence_size - pt_length), dtype='int'))
#     ))
#
# labels = tf.ragged.constant(labels).to_tensor()
# input_word_ids = tf.ragged.constant(input_word_ids)
# input_type_ids = tf.ragged.constant(input_type_ids)
# input_mask = tf.ragged.constant(input_mask)
#
vocab_size = len(act_tokenizer.vocab)
#
# input_ds = dict(
#     input_word_ids=input_word_ids.to_tensor(),
#     input_type_ids=input_type_ids.to_tensor(),
#     input_mask=input_mask.to_tensor()
# )

# print(vocab_size)

# sentences = tf.constant(["Hello Tensorflow!"])
bert = BERTFineTuningModel(pre_trained_model, batch_size, sequence_size, vocab_size)
#
# # from_logits=True if there was no softmax applied beforehand
# # from_logits=True tells the loss function that an activation function (e.g. softmax) was not applied on the last layer,
# # in which case your output needs to be as the number of classes. This is equivalent to using a softmax and
# # from_logits=False. However, if you end up using sparse_categorical_crossentropy, make sure your target values are 1D.
# # E.g. [1, 1, 0, 1, ...] (and not [[1], [1], [0], [1], ...]). On the other hand, if you use categorical_crossentropy
# # and your target values are 1D, you need to apply tf.keras.utils.to_categorical(targets)
# # on them first to convert them to 2D.
bert.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

bert.train(data, 1)


# Use sparse categorical crossentropy when your classes are mutually exclusive
# (e.g. when each sample belongs exactly to one class) and categorical crossentropy when one sample can have multiple
# classes or labels are soft probabilities (like [0.5, 0.3, 0.2]).
#
# Formula for categorical crossentropy (S - samples, C - classess, s∈c - sample belongs to class c) is:
#
# −1N∑s∈S∑c∈C1s∈clogp(s∈c)
#
# For case when classes are exclusive, you don't need to sum over them - for each sample only non-zero value is
# just −logp(s∈c) for true class c.
#
# This allows to conserve time and memory. Consider case of 10000 classes when they are mutually exclusive -
# just 1 log instead of summing up 10000 for each sample, just one integer instead of 10000 floats.
#
# Formula is the same in both cases, so no impact on accuracy should be there.
# If your targets are one-hot encoded, use categorical_crossentropy. Examples of one-hot encodings:
# [1,0,0]
# [0,1,0]
# [0,0,1]
# But if your targets are integers, use sparse_categorical_crossentropy.
# Examples of integer encodings (for the sake of completion):
#
# 1
# 2
# 3
