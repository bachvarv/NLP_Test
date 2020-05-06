import os

import numpy as np

from data.Dictionary import Dictionary
from word2vec.model_data.Word2VecModel import Word2VecModel
from data.utils_functions import read_text, fill_dict, create_training_arrays

current_path = os.path.dirname(os.path.realpath(__file__))



dictionary, training_set = read_text("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 2)
window = 2

# training_arr = learning_set(dictionary, window)
# print(dictionary)
# print(dictionary)
corpus_size = len(dictionary)
train_size = len(training_set)
# print(dictionary.get(training_set[0]))
# print(corpus_size)

print(training_set)

#
train_x, train_y = create_training_arrays(training_set, dictionary)

print(train_x[0])
print(train_y[0])
# train_x, train_y = [None] * train_size, [None] * train_size
# index = 0
# for item in training_set:
#     train_x[index] = [dictionary[x] for x in item[0]]
#     train_y[index] = [dictionary[y] for y in item[1]]
#     index = index + 1

# train_x = np.array(train_x)
# train_y = np.array(train_y)
# print(training_set)

# word2vec = Word2Vec(corpus_size, window)
# v = tf.Variable(train_y[0])
# v = tf.cast(v, tf.float32)
# w1 = tf.nn.softmax(tf.Variable(tf.random.truncated_normal([corpus_size,])))
# red_sum = tf.reduce_sum(tf.subtract(w1, v), axis=0)
# tf.print(w1)
# tf.print(v)
#
# tf.print(red_sum)

w2v = Word2VecModel(corpus_size, 10)

print(w2v.embed_layer.w.numpy())
w2v.train_loop(train_x, train_y, 1000)

mat = w2v.embed_layer.w.numpy()
print('Matrix {}'.format(mat))


# lookup_table = fill_dict(dictionary.keys(), mat)
# print(lookup_table)

# test_dic = Dictionary(lookup_table)

# keys = lookup_table.keys()

# print(list(keys))
# print(test_dic.d['natural'])

# res = test_dic.sum_of_values(test_dic.sub_of_values('fun', 'exciting')[0], 'processing')
#
# # res = (test_dic.d.get('language') - test_dic.d.get('processing')) + test_dic.d.get('machine')
#
# print('Result: {}'.format(res))
# print('fun: {}'.format(test_dic.d.get('is')))
# print('fun: {}'.format(test_dic.d.get('language')))

# print('Result in string: {}'.format(test_dic.get_similar(res, 3)))
# print(test_dic.get_sim_by_key('exciting', 2))

