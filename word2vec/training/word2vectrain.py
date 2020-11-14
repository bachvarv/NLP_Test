# dictionary, training_set = read_text("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 2)
import numpy as np

from data.Dictionary import Dictionary
from data.utils_functions import corpus2io, tokenize, fill_dict, build_vocab_Glove
from word2vec.model_data.Word2VecModel import Word2VecModel

window = 10
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')
big_corp = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt", 'r')
text = big_corp.readlines()
EPOCHS = 10
dictionary = build_vocab_Glove(text)

''' 
    Word2Vec Corpus prep
'''
corpus_tokenized, V = tokenize(text)
# print(corpus_tokenized)
# skip - x contains labels(middle word) and y contains context(surrounding words)
# train_y, train_x = zip(*corpus2io(corpus_tokenized, V, window))
# cbow - x contains context(surrounding words) and y contains labels(middle word)
train_x, train_y = zip(*corpus2io(corpus_tokenized, V, window))
train_x = np.array(train_x)
train_y = np.array(train_y)
# print(train_x[0], train_y[0])

w2v = Word2VecModel(V, 300)
# print(w2v(train_x[0]))

print(w2v.embed_layer.w.numpy())
w2v.train_loop(train_x, train_y, EPOCHS)
print(w2v.embed_layer.w)
w = w2v.embed_layer.w

lookup_table = fill_dict(dictionary.keys(), w)
# print(lookup_table)

test_dic = Dictionary(lookup_table)
test_dic.save("Win10_W2VCBOW300_Epoch10")
