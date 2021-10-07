import os
import platform
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import tensorflow as tf
import requests
from os.path import exists

from bert.corpus_utils.utils import download_data_wiki, _tokenize
from bert.data.BERTModel import BERTModel

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if(platform.system() == 'Linux'):
    dir = os.curdir + "/data/corpus/wikitext-2/"
else:
    dir = 'E:\Masterarbeit\Corpus\wikitext-2'

if not exists(os.curdir + '/data/corpus/wikitext-2/'):
    path = os.curdir + '/data/corpus/'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    corpus = urlopen(url)
    zip_file = ZipFile(BytesIO(corpus.read()))
    zip_file.extractall(path)
# paragraphs = _read_text(dir)

cfg = {
    'batch_size': 16,
    'input_max_len': 64,
    'num_layers': 12,
    'd_model': 768,
    'num_heads': 12,
    'depth_FF_Layers': 1024
    }


dataset_iter, vocab = download_data_wiki(
    cfg['batch_size'], cfg['input_max_len'], 5, dir)

print(_tokenize([['Hello Tensorflow']]))
words = _tokenize([['Hello Tensorflow']])[0]
print(vocab[words])


bert = BERTModel(cfg['num_layers'], cfg['d_model'],
                 cfg['num_heads'], cfg['depth_FF_Layers'],
                 len(vocab), cfg['input_max_len'])

bert.compile(optimizer='adam',
             loss=bert.loss)

iterator = dataset_iter.as_numpy_iterator()



tokens_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_labels = iterator.next()

bert(tokens_ids, segments, pred_positions)
# tf.keras.utils.plot_model(bert)
bert.train_step(iterator, 1)

if(platform.system() == 'Linux'):
    checkpoint_path = 'checkpoint/training/'
else:
    checkpoint_path ="E:\\Masterarbeit\\pre-trained bert\\bert_multi_cased_L-12_H-768_A-12_4.tar\\variables\\variables.index"

