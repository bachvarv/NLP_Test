import os


from data.Dictionary import Dictionary
from glove.GloVeModel import GloVeModel
from glove.GloveModelMultInput import GloVeModelMultiInput
from word2vec.model_data.Word2VecModel import Word2VecModel
from data.utils_functions import read_text, fill_dict, create_training_arrays, tokenize, corpus2io, build_vocab_Glove, \
    cooccur_mat

current_path = os.path.dirname(os.path.realpath(__file__))


# dictionary, training_set = read_text("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 2)
window = 3
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')
text = file.readlines()

''' 
    Word2Vec Corpus prep
'''
# corpus_tokenized, V = tokenize(text)
# skip - x contains labels(middle word) and y contains context(surrounding words)
# train_y, train_x = zip(*corpus2io(corpus_tokenized, V, window))
# cbow - x contains context(surrounding words) and y contains labels(middle word)
# train_x, train_y = zip(*corpus2io(corpus_tokenized, V, window))
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# print(train_x[0], train_y[0])

'''
    GloVe Corpus prep
'''

EPOCHS = 100
dictionary = build_vocab_Glove(text)
matrix = cooccur_mat(dictionary, text, window)
V = len(dictionary)

# w2v = Word2VecModel(V, 300)
glove = GloVeModel(V, 300)
#
glove.compile(optimizer='SGD',
              loss=glove.loss,
              metrics=['accuracy'])
# # print('the Message', msg)
# x, loss = glove(np.array([1, 1, 2]))
# print(x)

# print(glove.embedding_layer.w)
# print(glove.embedding_layer.b)
#
# grads = glove.train(np.array([1, 2, 2]))
# grads = glove.train(np.array([1, 2, 2]))
# grads = glove.train(np.array([1, 2, 2]))
#

w, b = glove.trainLoop(EPOCHS, matrix)

# print(w)
# print(b)

# print(dictionary)

lookup_table = fill_dict(dictionary.keys(), w)
# print(lookup_table)

test_dic = Dictionary(lookup_table)

test_dic.save("Win_2_Glove_10_Epoch")

# print(matrix)

            # print(row, column, matrix[row, column])
#
# print(glove.embedding_layer.w)
# print(glove.embedding_layer.b)



# print("Loss function returns", loss)

# for _ in range(EPOCHS):
#     for i in range(V):
#         for j in range(V):
#             occ = matrix[i][j]
#             if occ:
#                 acc = glove.fit([i, j], occ, batch_size=1)
                # result = glove([i,j, occ])
                # print(result)

# print(w2v(train_x[0]))
#
# print(w2v.embed_layer.w.numpy())
# w2v.train_loop(train_x, train_y, 100)
# # #
# mat = w2v.embed_layer.w.numpy()
# print('Matrix {}'.format(mat))
# #
# #
# print("result for {} and expected {} = {}".format(train_x[1], train_y[1], w2v(train_x[1])))



# print(corpus_tokenized[0])

# *training_set)
# print(training_set[0])
# print(train_x)
# print(train_y)

# training_arr = learning_set(dictionary, window)
# print(dictionary)
# print(dictionary)
# corpus_size = len(dictionary)
# train_size = len(training_set)
# # print(dictionary.get(training_set[0]))
# # print(corpus_size)
#
# print(training_set)
#
# #
# train_x, train_y = create_training_arrays(training_set, dictionary)
#
# print(train_x[0])
# print(train_y[0])
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

'''
GloveModel with multiple Inputs
'''

# glove = GloVeModelMultiInput(V, 10)
#
# msg = glove.compile(optimizer='SGD',
#               loss=glove.loss,
#               metrics=['accuracy'])

# glove.fit([1,1,2], y=2, batch_size=1)

# print('the Message', msg)
# x = glove(np.array([1, 1, 2]))

# x = glove.fit([1, 1, 2], batch_size=1)
