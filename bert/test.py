
from bert.corpus_utils.utils import download_data_wiki
from bert.data.BERTModel import BERTModel

dir = '/home/bachvarv/Abschlussarbeit/Corpus/wikitext-2-v1/wikitext-2/'

# paragraphs = _read_text(dir)

cfg = {
    'batch_size': 2,
    'input_max_len': 64,
    'num_layers': 4,
    'd_model': 64,
    'num_heads': 4,
    'hidden_size': 4,
    'depth_FF_Layers': 1024
                    }

dataset_iter, vocab = download_data_wiki(
    cfg['batch_size'], cfg['input_max_len'], 5, '/home/bachvarv/Abschlussarbeit/Corpus/wikitext-2-v1/wikitext-2/')


bert = BERTModel(cfg['num_layers'], cfg['d_model'],
                 cfg['num_heads'], cfg['depth_FF_Layers'],
                 len(vocab), cfg['input_max_len'])

bert.compile(optimizer='adam',
             loss=bert.loss)

iterator = dataset_iter.as_numpy_iterator()

bert.train_step(iterator, 1)