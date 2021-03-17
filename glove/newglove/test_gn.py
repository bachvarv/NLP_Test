from data.Dictionary import Dictionary
from data.utils_functions import build_vocab_Glove, cooccur_mat, fill_dict
from glove.newglove.GloveModelN import GloveModelN
from tensorflow import data, constant
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard


# FILE = "/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt"
# FILE = "/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt"
FILE = "/home/bachvarv/Abschlussarbeit/Corpus/Albert.txt"

file = open(FILE, 'r')
EPOCHS = 10
text = file.readlines()
window = 4
dictionary = build_vocab_Glove(text)
V = len(dictionary)
# print(dictionary)
matrix = cooccur_mat(dictionary, text, window)

targets, contexts, labels = [], [], []

u = 0
for i in matrix:
    for v, coocc in enumerate(i):
        if coocc == 0.0: continue
        targets.append(constant(u, shape=(1,)))
        contexts.append(constant(v, shape=(1, )))
        labels.append(constant(coocc, shape=(1,)))
    u = u + 1

# print(targets)
# print(contexts)
# print(labels)

dataset = data.Dataset.from_tensor_slices(((targets, contexts), labels))
# print(dataset)
dataset = dataset.shuffle(V).batch(1, drop_remainder=True)

model = GloveModelN(V, 200)

model.compile(optimizer="adam",
              loss=model.loss,
              metrics=[model.loss])

matrix = model.get_embedding_matrix()
print(matrix)

tensorboard_callback = TensorBoard(log_dir="logs")
model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])

matrix = model.get_embedding_matrix()
print(matrix[0])

lookup_table = fill_dict(dictionary.keys(), matrix[0])
dict = Dictionary(lookup_table)
dict.save("Albert_Glove_10EPOCHS_200VECTORSIZE")
