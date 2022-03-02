import os
import platform
import random
from io import BytesIO
from os.path import exists
from zipfile import ZipFile
from urllib.request import urlopen
import tensorflow as tf
import transformers
import numpy as np

from corpus_prep.corpus_gen import create_csv

path_to_wikitext = os.path.join(os.curdir, "wikitext-2")

if not exists(path_to_wikitext):
    path = os.curdir
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    corpus = urlopen(url)
    zip_file = ZipFile(BytesIO(corpus.read()))
    zip_file.extractall(path)


train_file = os.path.join(path_to_wikitext, "wiki.train.tokens")

text = []

once = 5
with open(train_file, 'r', encoding='utf8') as f:
    lines = f.readlines()

    paragraphs = [line.strip().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    text = paragraphs
    create_csv(text[:50])

# model_dir = "bert-base-german-cased"
# path_to_model = os.path.join(os.pardir, model_dir)
# # The Tokenizer
# tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)
#
#
#
# input_sequence = []
# line = ['Er ist im Garten spazierengegangen und hat ein Eichhörnchen gesehen.', 'Ich mache alles, was ich muss, um die Planeten zu sehen!']
# for l in line:
#     token_list = tokenizer.tokenize(l)
#     for i in range(1, len(token_list)):
#         n_gram_sequence = token_list[:i+1]
#         input_sequence.append(n_gram_sequence)
#
# print(input_sequence)
#
# predictors = [i[:-1] for i in input_sequence]
# label = [i[-1] for i in input_sequence]
# print(predictors)
# print(label)
# print(label)
# paragraphs = _read_text(dir)