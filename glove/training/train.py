import nltk
# nltk.download('punkt')
from data.Dictionary import Dictionary
from data.utils_functions import build_vocab_Glove, cooccur_mat, fill_dict
from glove.GloVeModel import GloVeModel

window = 10
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')
file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt", 'r')
text = file.readlines()

# arr = nltk.word_tokenize(text)
# print(arr)

EPOCHS = 10
dictionary = build_vocab_Glove(text)

print(dictionary)
print(len(dictionary))
print(len(dictionary.items()))
i = 0

matrix = cooccur_mat(dictionary, text, window)
V = len(dictionary)
print(V)

glove = GloVeModel(V, 300)
# #
glove.compile(optimizer='SGD',
              loss=glove.loss,
              metrics=['accuracy'])
#

w, b = glove.trainLoop(EPOCHS, matrix)
print(w)
#
lookup_table = fill_dict(dictionary.keys(), w)
#
test_dic = Dictionary(lookup_table)
#
# test_dic.save("Win10_Glove300_Epoch1")