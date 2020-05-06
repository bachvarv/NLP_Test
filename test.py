import numpy as np
import tensorflow as tf

from word2vec.word2vec import Word2Vec

vec = np.array([-0.490796, -0.229903, 0.065460])
other_vec = np.array([0.23074, -0.368008, 0.422434])

# print(np.transpose(vec) * other_vec)
print(np.dot(vec, other_vec))
print(np.dot(np.transpose(vec), other_vec))

model = Word2Vec()

# print(tf.reduce_sum(tf.random.normal([1000, 1000])))