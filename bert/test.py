import platform
import tensorflow as tf

from bert.corpus_utils.utils import download_data_wiki, _tokenize
from bert.data.BERTModel import BERTModel

if(platform.system() == 'Linux'):
    dir = '/home/bachvarv/Abschlussarbeit/Corpus/wikitext-2-v1/wikitext-2/'
else:
    dir = 'E:\Masterarbeit\Corpus\wikitext-2-v1\wikitext-2'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# paragraphs = _read_text(dir)

cfg = {
    'batch_size': 64,
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
tf.keras.utils.plot_model(bert)
# bert.train_step(iterator, 1)

checkpoint_path ="E:\\Masterarbeit\\pre-trained bert\\bert_multi_cased_L-12_H-768_A-12_4.tar\\variables\\variables.index"

