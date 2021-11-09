import os
import tarfile
from io import BytesIO
from os import path
import platform

import numpy as np
import requests
import transformers
import tensorflow as tf
import tensorflow_hub
from official.nlp import bert
import official.nlp.bert.tokenization
import subprocess
from keras.models import load_model
import tensorflow_hub as hub


def git(*args):
    return subprocess.check_call(['git'] + list(args))


cfg = dict(
    max_sentence=128
)

# Clone the

if not os.path.isdir('bert_en_cased_L-12_H-768_A-12_4'):
    file = 'bert_en_cased_L-12_H-768_A-12_4.tar.gz'
    url = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4?tf-hub-format=compressed'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file, 'wb') as f:
            f.write(response.raw.read())
    file = tarfile.open(file)
    file.extractall('./bert_en_cased_L-12_H-768_A-12_4')
    file.close()
    # model = ZipFile(BytesIO(url_open.read()))
    # model.extractall()

if not os.path.isdir('bert-base-german-cased'):
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)

# Path to vocabulary
model_dir = "bert-base-german-cased"
path_to_vocab = os.path.join(model_dir, "vocab.txt")
path_to_weights = os.path.join(model_dir, "tf_model.h5")
path_to_model = os.path.join(os.curdir, model_dir)

# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)
text = "Die Steinbrech-Felsennelke ist eine überwinternd grüne, ausdauernde krautige Pflanze " \
       "und erreicht Wuchshöhen von 10 bis 30 Zentimetern"

# Testing the tokenizer
tokens = tokenizer.tokenize(text)
pre_model = tokenizer.prepare_for_model(tokens)
input_ids = tokenizer(text, return_tensors='tf')

print(tokens)
print(pre_model)
print(input_ids)
# print(tokens)

# Creating the model

inp_layer = dict(
    input_ids=tf.keras.layers.Input(shape=33, dtype=tf.int32),
    token_type_ids=tf.keras.layers.Input(shape=33, dtype=tf.int32),
    attention_mask=tf.keras.layers.Input(shape=33, dtype=tf.int32)
)

encoder_input = transformers.TFBertModel.from_pretrained(path_to_model)
outputs = encoder_input(inp_layer)
net = outputs.last_hidden_state
net = tf.keras.layers.Dropout(0.1)(net)
output = tf.keras.layers.Dense(tokenizer.vocab_size, activation=None)(net)

model = tf.keras.Model(inp_layer, output)
# out = encoder_input(input_ids).last_hidden_state
# pooler_output = encoder_input(input_ids).pooler_output

test = model(input_ids)
guess = tf.argmax(test, axis=-1)
print(guess)
print(tokenizer.convert_ids_to_tokens(guess[0]))

import os

path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
with open(path_to_corpus, 'r') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        print('Candidate: ')
        print(tokenizer.tokenize(x))
        print('Expected: ')
        print(tokenizer.tokenize(y))


# print(out.last_hidden_state)
# encoder_input.set_weights()
# encoder_output = encoder_input(inp)

# print(encoder_output)
