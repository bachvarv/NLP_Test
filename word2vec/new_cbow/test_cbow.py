from data.Dictionary import Dictionary
from data.utils_functions import tokenize, create_cbow_dataset, fill_dict
import tensorflow as tf

from word2vec.new_cbow.CbowModelNS import CbowModelNS

SEED = 42

file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt", "r")
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')

text = file.readlines()

corpus_tokenized, V, dic = tokenize(text)
V = V + 1
inv_dic = {i: word for word, i in dic.items()}
window = 4
num_ns = 4

targets, contexts, labels = create_cbow_dataset(corpus_tokenized, V, window, num_ns, SEED)



print("Targets:", targets[0])
print("Contexts:", contexts[0])
print("Labels:", labels[0])

dataset = tf.data.Dataset.from_tensor_slices(((contexts, targets), labels))

print(dataset)
dataset = dataset.shuffle(V).batch(1, drop_remainder=True)

model = CbowModelNS(V, 200, num_ns)
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.fit(dataset, epochs=10, callbacks=[tensorboard_callback])
# model()

matrix = model.get_embedding_matrix()

lookup_table = fill_dict(dic.keys(), matrix[0][1:])
dic = Dictionary(lookup_table)

dic.save("CBOWNS_10EPOCHS_200VECTORSIZE")


