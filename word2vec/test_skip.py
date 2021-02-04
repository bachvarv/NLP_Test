import tensorflow as tf

from data.Dictionary import Dictionary
from data.utils_functions import build_vocab_Glove, tokenize, create_skip_dataset, fill_dict
from word2vec.new_skip.SkipModelNS import SkipModelNS

SEED = 42


# print(dataset)
file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus.txt")
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small2.txt", 'r')
# file = open("/home/bachvarv/Abschlussarbeit/Corpus/corpus_small.txt", 'r')
text = file.readlines()
corpus_tokenized, V, dic = tokenize(text)
V = V + 1
inv_dic={i:word for word, i in dic.items()}
window = 4
num_ns = 4

# print(inv_dic)
#
# print(corpus_tokenized[0])
# arr = corpus_tokenized[0]
# print(arr)
#
# window = 4
#
# positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams\
#     (arr,
#      vocabulary_size=V,
#      window_size=window,
#      negative_samples=0)

#
# target_word, context_word = positive_skip_grams[0]
#
# print(inv_dic[target_word])
# print(inv_dic[context_word])
#
# num_ns = 4
#
# context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
# negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
#     true_classes=context_class,
#     num_true=1,
#     num_sampled=num_ns,
#     unique=True,
#     range_max=V,
#     seed=SEED,
#     name="negative_sampling"
# )
# negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
#
# context = tf.concat([context_class, negative_sampling_candidates], 0)
#
# # for i in context:
# #     if i != 0:
# #         context_v = tf.concat(i, -1)
# #
# # context_v = tf.concat(context[-1], -1)
# label = tf.constant([1] + [0]*num_ns, dtype="int64")
#
# print(context)
#
# # Reshape target to shape (1,) and context and label to (num_ns+1,).
# target = tf.squeeze(target_word)
# context = tf.squeeze(context)
# label = tf.squeeze(label)
#
# print(f"target_index    : {target}")
# print(f"target_word     : {inv_dic[target_word]}")
# print(f"context_indices : {context}")
# print(f"context_words   : {[inv_dic[c.numpy()] for c in context if c != 0]}")
# print(f"label           : {label}")
#
# target = tf.cast(tf.reshape(target, [1, 1]), dtype=tf.int32)
# context = tf.cast(tf.reshape(context, [1, 5]), dtype=tf.int64)
# label = tf.cast(tf.reshape(label, [1, 5]), dtype=tf.int64)
#
# dataset = tf.data.Dataset.from_tensor_slices(((target, context), label))
# # dataset = ((target, context), label)
# # dataset.shuffle(4).batch(1, drop_remainder=False)
# print(dataset)

targets, contexts, labels = create_skip_dataset(corpus_tokenized, V, window, num_ns, SEED)
# Training

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

print(dataset)
dataset = dataset.shuffle(V).batch(1, drop_remainder=True)
# print(dataset)


model = SkipModelNS(V, 200)
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.fit(dataset, epochs=10, callbacks=[tensorboard_callback])

# weights = model.get_layer('glove_embedding').get_weights()
# print(weights)

matrix = model.get_embedding_matrix()
print(matrix.shape)

lookup_table = fill_dict(dic.keys(), matrix[0][1:])
dic = Dictionary(lookup_table)

dic.save("SKIPNS_10EPOCHS_200VECTORSIZE")

# print(dataset[0])
# result = model(dataset[0])

# print(result)