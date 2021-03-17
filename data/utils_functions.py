import re
from collections import Counter

import numpy as np
from keras_preprocessing.text import Tokenizer
from keras.utils import np_utils as npu
import tensorflow as tf

def read_text(path, window):
    temp = []
    ts = []
    d = dict()
    file = open(path)
    words = file.read().lower()
    # print(words)
    words = re.sub(r'[^\w\s]', '', words)
    words = words.replace('_', '')
    words = words.split()
    # print(words)
    hot_index = 0
    for i in words:
        if i not in temp:
            temp.append(i)

    for i in temp:
        d[i] = np.zeros(shape=(len(temp),)).astype(float)
        d[i][hot_index] = float(1)
        hot_index += 1

    size = len(words)
    keys = list(words)
    for i in range(size):
        t = list()
        t.append([keys[i]])
        labels = list()
        for b in range(i - window, i):
            if b < 0 or b == i:
                continue
            labels.append(keys[b])
        for f in range(i, i + (window + 1)):
            if f >= size or f == i:
                continue
            labels.append(keys[f])

        t.append(labels)
        ts.append(t)

    return d, ts


def tokenize(corpus):
    """
    src: http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/

    Tokenize the corpus text.

        :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
        :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus (the example above would return [[1, 2, 3, 4]])
        :return V: size of vocabulary
    """

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V, tokenizer.word_index


def corpus2io(corpus_tokenized, V, window_size=10):
    """
    src: http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/

    Converts corpus text into context and center words.

        :param corpus_tokenized: corpus text
        :param window_size: size of context window
        :return: context and center words (arrays)
    """
    train_set = []
    for words in corpus_tokenized:
        L = len(words)

        for index, word in enumerate(words):
            s = index - window_size
            e = index + window_size + 1
            contexts = [words[i]-1 for i in range(s, e) if 0 <= i < L and i != index]
            labels = list()
            # contexts.append([words[i]-1 for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word-1)
            x = npu.to_categorical(contexts, V)
            # print(x)
            y = npu.to_categorical(labels, V)
            train_set.append([np.sum(x, axis=0), y.ravel()])
            # print(train_set)
            # yield (x, y.ravel())
    return train_set

def read_from_file(path):
    """
    Read a matrix from file into an array
    :param path:
    :return:
    """
    file = open(path, mode='r')

    for line in file.readlines():
        print(line)


def cosine_sim(A, B):
    """
    Return the cosine similarity between two vectors.
    :param A:
    :param B:
    :return:
    """
    dot_prod = np.dot(A, B)
    mag_A = np.sqrt(np.dot(A, A))
    mag_B = np.sqrt(np.dot(B, B))
    mag_mul = mag_A * mag_B
    cos_AB = dot_prod / mag_mul
    return cos_AB


def fill_dict(arr_lab, arr_val):
    """
    Create a dictionary with keys from arr_lab and values from arr_val

    :param arr_lab:
    :param arr_val:
    :return:
    """
    d = {}
    i = 0
    for s in arr_lab:

        d[s] = arr_val[i]
        i += 1
    return d


def create_training_arrays(training_set, dictionary):
    """
    Create the training array

    :param training_set:
    :param dictionary:
    :return:
    """
    train_x = []
    train_y = []
    for item in training_set:
        # print(item)
        train_x.append([dictionary[x] for x in item[0]])
        train_y.append([dictionary[y] for y in item[1]])

    return np.array(train_x), np.array(train_y)
    # print(train_x, train_y)


# Glove utils
def cooccur_mat(vocab, corpus, window_size=10, min_count=None):
    """
    Create Co-Occurrence matrix

    :param vocab: an array of the vocabulary
    :param corpus: the text
    :param window_size: the size of the context window
    :param min_count: gives a minimal count of word occurrence
    :return: an array of tuples in the form of (i_main, i_context, co-occurrence)
    """
    vocab_size = len(vocab)
    mat = np.zeros(shape=(vocab_size, vocab_size))
    for line in corpus:
        words = line.lower().strip().split(' ')
        # print(words)
        for ind, w in enumerate(words):
            w = re.sub(r'[^\s\w]', '', w)
            (w_ind, _) = vocab[w]
            for l in range(max(0, ind - window_size), ind):
                co_word = re.sub(r'[^\s\w]', '', words[l])
                l_ind, _ = vocab[co_word]
                mat[w_ind][l_ind] += 1

            for r in range(ind + 1, min(ind + window_size + 1, len(words))):
                right_word = re.sub(r'[^\s\w]', '', words[r])
                r_ind, _ = vocab[right_word]
                mat[w_ind][r_ind] += 1
    return mat



def build_vocab_Glove(corpus):
    """
    Build the vocabulary for Glove
    src:

    :param corpus: the path for the file
    :return: dictionary with id and occurrence in the corpus for the word
    """
    vocab = Counter()
    # corpus = open(corpus, 'r')
    for line in corpus:
        # print(line)
        tokens = line.lower().strip().split(' ')
        vocab.update(tokens)
    index = 0
    dic = dict()
    for word, freq in vocab.items():
        dic.__setitem__(re.sub(r'[^\s\w]', '', word), (index, freq))

    dic = {word: (i, freq) for i, (word, (_, freq)) in enumerate(dic.items())}
    return dic


def one_hot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return vec


def create_skip_dataset(corpus_tokenized, corpus_size, window_size, num_ns, seed):

    targets, contexts, labels = [], [], []

    # sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=(corpus_size + 1))
    # print(sampling_table)

    for sequence in corpus_tokenized:
        # print(sequence)
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=corpus_size,
          window_size=window_size,
          negative_samples=0)

        # print(positive_skip_grams)
        for target_word, context_word in positive_skip_grams:

            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=corpus_size,
                    seed=seed,
                    name="negative_sampling"
                )

            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

        return targets, contexts, labels


def create_cbow_dataset(corpus_tokenized, corpus_size, window_size, num_ns, seed):
    targets, contexts, labels = [], [], []

    # sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=(corpus_size + 1))
    # print(sampling_table)

    for sequence in corpus_tokenized:
        # print(sequence)
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=corpus_size,
          window_size=window_size,
          negative_samples=0)


        arr = dict()

        for i in positive_skip_grams:
            if i[0] in arr.keys() and i[1] not in arr.values():
                arr[i[0]].append(i[1])
            else:
                arr[i[0]] = [i[1]]

        for (k, v) in arr.items():
            while len(v) < window_size * 2:
                arr[k].append(k)
            if len(v) > window_size * 2:
                arr[k] = arr[k][:window_size*2]

        for target_word, context_word in arr.items():

            target_class = tf.expand_dims(
                tf.constant([target_word], dtype="int64"), 1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=target_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=corpus_size,
                    seed=seed,
                    name="negative_sampling"
                )

            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            target = tf.concat([target_class, negative_sampling_candidates], 0)
            # context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            targets.append(target)
            contexts.append(context_word)
            labels.append(label)

        return targets, contexts, labels


