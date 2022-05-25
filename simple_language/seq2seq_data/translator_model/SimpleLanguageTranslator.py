from enum import Enum
import tensorflow as tf
import transformers
from nltk.translate.bleu_score import corpus_bleu

from simple_language.seq2seq_data.SimpleLanguageModel import SimpleLanguageModel
from simple_language.seq2seq_data.bahdanau.BertLanguageModel import BertLanguageModel
from simple_language.seq2seq_data.luong.BertLanguageModelGeneral import BertLanguageModelGeneral


class ModelType(Enum):
    bahdanau = 'Bahdanau'
    luong = 'Luong'
    simple = 'Simple'


cfg = dict(
    max_sentence=128,
    hidden_layer_size=768,
    batch_size=1,
    transformer_heads=12,
    head_size=64
)

class SimpleLanguageTranslator():
    def __init__(self, model_checkpoint, tokenizer_path, model_type: ModelType, list_corpus):
        self.model_type = model_type
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_path)
        self.model_name = model_checkpoint

        test_input = (['Hallo, Welt!'], ['[CLS]'])
        if self.model_type == ModelType.simple:
            self.model = SimpleLanguageModel(cfg['hidden_layer_size'], self.tokenizer.vocab_size, self.tokenizer)
            # self.model(test_input)
        elif self.model_type == ModelType.bahdanau:
            self.model = BertLanguageModel(cfg['hidden_layer_size'],
                                           self.tokenizer.vocab_size,
                                           self.tokenizer,
                                           path_to_model=tokenizer_path,
                                           bert_trainable=False)
            # self.model(test_input)

        elif self.model_type == ModelType.luong:
            self.model = BertLanguageModelGeneral(cfg['hidden_layer_size'],
                                                  self.tokenizer.vocab_size,
                                                  self.tokenizer,
                                                  path_to_model=tokenizer_path,
                                                  bert_trainable=False)
            # self.model(test_input)

        self.checkpoint = tf.train.Checkpoint(self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, model_checkpoint, max_to_keep=3)
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        self.corpus_items = list_corpus


    def predict(self, text):
        input = ([text], ['[CLS]'])
        _, _, text = self.model(input)
        
        return text


    def print_bleu_score(self):
        print(f'Testing {self.model_name}')
        input_x = []
        label_y = []
        for file in self.corpus_items:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    x, y = line.split(sep='\t')
                    y = self.tokenizer.tokenize(y)
                    input_x.append(x)
                    label_y.append([y])

        outputs = []
        for x in input_x:
            print(f'Candidate: {x}')
            output = self.predict(x)
            #output = self.tokenizer.tokenize(output)
            outputs.append(output) 
            print(f'Prediction: {output}')

        bleu_score = corpus_bleu(label_y, outputs)
        print(bleu_score)
        return bleu_score
