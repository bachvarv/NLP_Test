import os

import numpy
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from keras.utils.metrics_utils import Reduction
from numpy import dtype
from official import nlp
from official.nlp import optimization, bert
import official.nlp.bert.tokenization

import tensorflow_text


def pre_process_input(input, batch_size, max_sentence_size):
    if batch_size == 1:

        array = input
        size = len(array)

        if size > max_sentence_size:
            return dict()

        input_mask = np.zeros((1, max_sentence_size)).astype(int)
        processed_input = np.zeros((1, max_sentence_size)).astype(int)
        processed_input[:1, :size, ] = array
        input_mask[:1, :size, ] = np.ones(size)
        input_type_ids = np.zeros((batch_size, max_sentence_size)).astype(int)
    else:
        array = input.numpy()
        array_size = len(array)
        input_mask = np.zeros((array_size, max_sentence_size)).astype(int)
        processed_input = np.zeros((array_size, max_sentence_size)).astype(int)
        input_type_ids = np.zeros((array_size, max_sentence_size)).astype(int)
        for i in range(len(array)):
            if len(array[i]) < max_sentence_size:
                processed_input[i][:len(array[i]), ] = array[i]
                input_mask[i][:len(array[i]), ] = np.ones(array[i].shape)
            else:
                return dict()

    return {'input_mask': tf.convert_to_tensor(input_mask),
            'input_type_ids':
                tf.convert_to_tensor(input_type_ids),
            'input_word_ids':
                tf.convert_to_tensor(processed_input)}


def different_pre_process_input(input, max_sentence_size):
    if input is None:
        return None, None, None

    size = len(input)
    input_mask = np.zeros((size, max_sentence_size)).astype(int)
    processed_input = np.zeros((size, max_sentence_size)).astype(int)
    input_type_ids = np.zeros((size, max_sentence_size)).astype(int)


    return input_mask, input_type_ids, processed_input

## Dataset
batch_size = 8
max_sentence_size = 128

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

dataframe = tfds.as_dataframe(examples['train'].take(-1), metadata)

print(dataframe.head(2))
ds = tf.data.Dataset.from_tensor_slices((dataframe['en'], dataframe['pt']))
print(ds)

train_examples, val_examples = examples['train'], examples['validation']
en_examples = []
pt_examples = []

model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(f"{model_name}.zip",
                                    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
                                    cache_dir='.', cache_subdir='', extract=True)
tokenizer = tf.saved_model.load(model_name)
en_tokenizer = tokenizer.en
pt_tokenizer = tokenizer.pt

model_checkpoint = 'E:\\Masterarbeit\\pre-trained bert\\bert_multi_cased_L-12_H-768_A-12_4'

# TODO: new Tokenizer remove the old one
act_tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(model_checkpoint,
                                                                        "assets\\vocab.txt"), do_lower_case=True)

# print("Vocab Size: ", len(act_tokenizer.vocab))

# train_pt_examples, train_en_examples = train_examples.batch(30000).take(5)
#
# print(train_pt_examples[0][0])
# print(train_en_examples[0][0])

# preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")

train_label = []
train_input = dict(input_mask=[],
                   input_type_ids=[],
                   input_word_ids=[])
train_input_array = []
train_label_array = []
one_sent = ''
index = 0
over = False

#TODO: Add the CLS and SEP token
for train_pt_examples, train_en_examples in train_examples.batch(1):
    # train_pt_examples = train_pt_examples.numpy()
    # train_en_examples = train_en_examples.numpy()
    train_pt_examples = list(act_tokenizer.tokenize(train_pt_examples.numpy()[0]))
    train_en_examples = list(act_tokenizer.tokenize(train_en_examples.numpy()[0]))
    for example in train_pt_examples:
        one_sent = len(example.split())
        if one_sent > max_sentence_size:
            over = True
            break
    if over is False:
        for example in train_en_examples:
            one_sent = len(example.split())
            if one_sent > max_sentence_size:
                over = True
                break
#
    if over is True:
        over = False
        continue
#
    train_pt_examples.insert(0, '[CLS]')
    train_pt_examples.append('[SEP]')
    train_en_examples.insert(0, '[CLS]')
    train_en_examples.append('[SEP]')
    train_en_examples = act_tokenizer.convert_tokens_to_ids(train_en_examples)
    train_pt_examples = act_tokenizer.convert_tokens_to_ids(train_pt_examples)
    train_input_array.append(train_en_examples)
    train_label_array.append(train_pt_examples)


    # input_mask, input_type_ids, input_word_ids = different_pre_process_input(train_en_examples, max_sentence_size)

    # if input_mask is None:
    #     continue

    # _, _, pt_ex = different_pre_process_input(train_pt_examples, max_sentence_size)

    # if pt_ex is None:
    #     continue
    #
    # if pt_ex.shape == 0:
    #     continue
    # train_input.append(en_ex)
    # train_label.append(pt_ex)
    index+=1
    # train_input.get("input_mask").append(input_mask)
    # train_input.get("input_type_ids").append(input_type_ids)
    # train_input.get("input_word_ids").append(input_word_ids)
    # train_input_array.append(dict(input_mask=input_mask, input_type_ids=input_type_ids, input_word_ids=input_word_ids))
    if index == 3:
        break

# TODO: create the right input data
# input_word_ids = [i for i in train_input_array]
# input_mask = [[1 for _ in range(len(i))] for i in train_input_array]
# input_type_ids = [[1 for _ in range(len(i))]  for i in train_input_array]

print()
en_text = dataframe['en'].values
pt_text = dataframe['pt'].values
input_words_tokenized = [act_tokenizer.convert_tokens_to_ids(act_tokenizer.tokenize(text)) for text in en_text]
label_words_tokenized = [act_tokenizer.convert_tokens_to_ids(act_tokenizer.tokenize(text)) for text in pt_text]
print(input_words_tokenized)
print(label_words_tokenized)

input_word_ids = []
input_mask = []
input_type_ids = []
labels = []
# for i in range(len(input_word_ids)):
#     size = len(input_word_ids[i]
for i in range(len(input_words_tokenized)):
    if len(input_words_tokenized[i]) > max_sentence_size or len(label_words_tokenized[i]) > max_sentence_size:
        continue

    input_word_ids.append(np.concatenate(
        (input_words_tokenized[i], np.zeros(shape=max_sentence_size - len(input_words_tokenized[i]), dtype='int'))
    ))

    input_mask.append(np.concatenate(
        (np.ones_like(input_words_tokenized[i]),
        np.zeros(shape=max_sentence_size - len(input_words_tokenized[i]), dtype='int'))
    ))

    input_type_ids.append(np.zeros(shape=max_sentence_size, dtype='int'))

    labels.append(np.concatenate(
        (label_words_tokenized[i], np.zeros(shape=max_sentence_size - len(label_words_tokenized[i]), dtype='int')))
    )

# print(input_word_ids)
# print(labels)

labels = tf.ragged.constant(labels).to_tensor()
input_word_ids = tf.ragged.constant(input_word_ids)
input_type_ids = tf.ragged.constant(input_type_ids)
input_mask = tf.ragged.constant(input_mask)

input_items = dict(
    input_word_ids=input_word_ids.to_tensor(),
    input_type_ids=input_type_ids.to_tensor(),
    input_mask=input_mask.to_tensor()
)

for key, value in input_items.items():
  print(f'{key:15s} shape: {value.shape}')

# inp_arr = different_pre_process_input(train_input_array, max_sentence_size)
# print(inp_arr)

# print(f'Training array: {train_array}')
# print(f'Training input: {train_input_array}')

validation_input = []
validation_label = []
over = False
# print(val_examples)
for val_pt_examples, val_en_examples in val_examples.batch(1):
    val_pt_examples = val_pt_examples.numpy()
    val_en_examples = val_en_examples.numpy()
    val_pt_examples = list(act_tokenizer.tokenize(val_pt_examples[0]))
    val_en_examples = list(act_tokenizer.tokenize(val_en_examples[0]))

    val_pt_examples.insert(0, '[CLS]')
    val_pt_examples.append('[SEP]')
    val_en_examples.insert(0, '[CLS]')
    val_en_examples.append('[SEP]')

    validation_en_examples = act_tokenizer.convert_tokens_to_ids(val_pt_examples)
    validation_pt_examples = act_tokenizer.convert_tokens_to_ids(val_en_examples)
    validation_en_examples = pre_process_input(validation_en_examples, 1, max_sentence_size)
    validation_pt_examples = pre_process_input(validation_pt_examples, 1, max_sentence_size)

    if len(validation_pt_examples) != 0 and len(validation_en_examples) != 0:
        validation_input.append(validation_en_examples)
        validation_label.append(validation_pt_examples['input_word_ids'])



# dataset = tf.data.Dataset().from_tensor_slices(train_ex)
# print("Train label: ", len(train_label))
# print("Train Input: ", len(train_input))
vocab_size = len(act_tokenizer.vocab)

# train_input = train_input[:10]
# train_label = train_label[:10]
#
# validation_label = validation_label[:10]
# validation_input = validation_input[:10]
# defining the model

pre_trained_model = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'

inputs = dict(
    input_word_ids=tf.keras.layers.Input(shape=max_sentence_size, dtype=tf.int32),
    input_mask=tf.keras.layers.Input(shape=max_sentence_size, dtype=tf.int32),
    input_type_ids=tf.keras.layers.Input(shape=max_sentence_size, dtype=tf.int32)
)

encoder_input = hub.KerasLayer(pre_trained_model, trainable=True, name="BERT_Encoder")

outputs = encoder_input(inputs)

net = outputs['sequence_output']
net = tf.keras.layers.Dropout(0.1)(net)
output = tf.keras.layers.Dense(vocab_size, activation=None)(net)

model = tf.keras.Model(inputs, output)

# bert_raw_result = model(train_input_array[0])

tf.keras.utils.plot_model(model)
#loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalCrossentropy()

init_lr = 5e-5
steps_per_epoch = len(labels)
num_warmup_steps = int(0.1 * steps_per_epoch)
# the optimizer
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=steps_per_epoch,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[metrics])
# print(tf.nn.softmax(bert_raw_result))

tf.keras.utils.plot_model(model, to_file='model.png')

# Checkpoint

path_to_checkpoint = 'training/checkpoints'
ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=3)

print(f'Training model with {pre_trained_model}')
print(f'Training label: {np.shape(labels)}')
print(f'Training input: {input_items["input_word_ids"].shape}')
if ckpt_manager.latest_checkpoint:
    print(f'Loading checkpoint from {ckpt_manager.latest_checkpoint}')
    ckpt.restore(ckpt_manager.latest_checkpoint)

# for i in range(len(train_input_array)):
model.fit(x=input_items, y=labels, batch_size=8, epochs=1)

ckpt_manager.save()
# #
# #
# print(len(validation_input))
for i in range(len(validation_input)):
    print("Validating")
    print(model.evaluate(x=validation_input[i], y=validation_label[i],verbose=1))
#
model.save('saved_model/')
