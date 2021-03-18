import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
# print(train_examples.batch(1).take(1))
# print(examples)

# The Interesting Part about this is that
# the pt_examples and en_examples after their initialization here
# they stay initialized and are callable later in the program
# Which is very bizarre
for pt_examples, en_examples in train_examples.batch(5).take(1):
    # for pt in pt_examples.numpy():
    #     print(pt.decode('utf-8'))

    # print()

    pt_examp = pt_examples
    en_examp = en_examples

    # for en in en_examples.numpy():
    #     print(en.decode('utf-8'))


model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(f"{model_name}.zip",
                        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
                        cache_dir='.', cache_subdir='', extract=True)
tokenizers = tf.saved_model.load(model_name)
encoded = tokenizers.en.tokenize(en_examples)
pt_encoded = tokenizers.pt.tokenize(pt_examples)

# for row in encoded.to_list():
#     print(row)

round_trip = tokenizers.en.detokenize(encoded)

for line in round_trip.numpy():
    print(line.decode('utf-8'))