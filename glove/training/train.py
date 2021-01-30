
# nltk.download('punkt')
from data.Dictionary import Dictionary
from data.utils_functions import build_vocab_Glove, cooccur_mat, fill_dict
from glove.GloVeModel import GloVeModel
from glove.GloveModelMultInput import GloVeModelMultiInput

window = 10
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')
file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt", 'r')
text = file.readlines()

# arr = nltk.word_tokenize(text)
# print(arr)

EPOCHS = 1
dictionary = build_vocab_Glove(text)

# print(dictionary)
# print(len(dictionary))
# print(len(dictionary.items()))
i = 0

matrix = cooccur_mat(dictionary, text, window)
V = len(dictionary)
# print(V)

# Normal Model
# glove = GloVeModel(V, 200)

# Glove with multi input
glove = GloVeModelMultiInput(V, 200)
glove.compile(optimizer=glove.optimizer,
              loss=glove.loss,
              metrics=['accuracy'])
#

print(glove.embedding_layer.w.numpy())
w, b = glove.trainLoop(EPOCHS, matrix)
print(w)
#
lookup_table = fill_dict(dictionary.keys(), w)
#
test_dic = Dictionary(lookup_table)
#
test_dic.save("Win10_GloveMultiInput200_Epoch1_Test")