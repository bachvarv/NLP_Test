from keras.layers import TextVectorization

from bert.fine_tuning.data.BERTFineTuningModel import BERTFineTuningModel
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
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

    return {'input_mask': input_mask,
            'input_type_ids': input_type_ids,
            'input_word_ids': processed_input}


pre_trained_model = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
sequence_size = 128
batch_size = 2

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

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


# train_pt_examples, train_en_examples = train_examples.batch(30000).take(5)
#
# print(train_pt_examples[0][0])
# print(train_en_examples[0][0])

# preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
train_ex = []
one_sent = ''
over = False
for train_pt_examples, train_en_examples in train_examples.batch(batch_size):
    for example in train_pt_examples:
        one_sent = len(example.numpy().split())
        if one_sent > sequence_size:
            over = True
            break
    if over is False:
        for example in train_en_examples:
            one_sent = len(example.numpy().split())
            if one_sent > sequence_size:
                over = True
                break

    if over is True:
        over = False
        continue

    train_en_examples = en_tokenizer.tokenize(train_en_examples)
    train_pt_examples = pt_tokenizer.tokenize(train_pt_examples)

    en_ex = pre_process_input(train_en_examples, batch_size, sequence_size)
    pt_ex = pre_process_input(train_pt_examples, batch_size, sequence_size)
    train_ex.append((en_ex, pt_ex))


# print(train_ex[0])
vocab_size = pt_tokenizer.get_vocab_size()

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


tf.keras.utils.plot_model(bert, show_shapes=True, dpi=48, to_file='model_1.png')

train_example_size = len(en_examples)
bert.train(train_ex, train_example_size, 1)



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
