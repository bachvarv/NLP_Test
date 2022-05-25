import nltk
import transformers
from nltk.translate.bleu_score import corpus_bleu, ngrams
import os
from matplotlib import pyplot as plt
from simple_language.seq2seq_data.translator_model.SimpleLanguageTranslator import SimpleLanguageTranslator, ModelType
import csv

path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'einfache_sprache.csv')
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'spd_programm_einfache_sprache_v1.csv')
path_to_corpus_3 = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'simple_language_openAI.csv')

path_to_corpus_4 = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'test_corpus_ready.csv')

corpus_list = [path_to_corpus, path_to_corpus_2, path_to_corpus_3]
checkpoints = list()
scores = list()

path_to_tokenizer = os.path.join(os.curdir, os.path.join('simple_language', 'bert-base-german-cased'))
# path_to_checkpoint = os.path.join(os.curdir, os.path.join('simple_language', 'BSLM_v1_lr_1e-3'))
# SLM
# path_to_checkpoint = os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr1e-3')
#checkpoints.append(os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr1e-3'))
#checkpoints.append(os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr1e-4'))
#checkpoints.append(os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr2e-5'))
#checkpoints.append(os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_EP45_lr1e-3'))
# path_to_checkpoint = os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr1e-4')
# path_to_checkpoint = os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_lr2e-5')

#BSLM Luong
# path_to_checkpoint = os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-3')
# path_to_checkpoint = os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-4')
# path_to_checkpoint = os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr2e-5')
# path_to_checkpoint = os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-5')
checkpoints.append(os.path.join(os.curdir, 'BLM_HPC_lr2e-5_Bahdanau_50EP'))
checkpoints.append(os.path.join(os.curdir, 'BSLM_HPC_General_EP50_r2e-5'))
#checkpoints.append(os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-3'))
#checkpoints.append(os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-4'))
#checkpoints.append(os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr2e-5'))
#checkpoints.append(os.path.join(os.curdir, 'BSLM_HPC_General_v1_lr1e-5'))

#BSLM Bahdanau
# path_to_checkpoint = os.path.join(os.curdir, 'BLM_HPC_v1_lr1e-3_Bahdanau_100EP')
# path_to_checkpoint = os.path.join(os.curdir, 'BLM_HPC_v1_lr1e-4_Bahdanau_100EP')
# path_to_checkpoint = os.path.join(os.curdir, 'BLM_HPC_v1_lr2e-5_Bahdanau')
# path_to_checkpoint = os.path.join(os.curdir, 'BLM_HPC_v1_lr2e-5_Bahdanau_15EP')
#checkpoints.append(os.path.join(os.curdir, 'BLM_HPC_v1_lr1e-3_Bahdanau_100EP'))
#checkpoints.append(os.path.join(os.curdir, 'BLM_HPC_v1_lr1e-4_Bahdanau_100EP'))
#checkpoints.append(os.path.join(os.curdir, 'BLM_HPC_v5_lr2e-5_Bahdanau_100EP'))
#checkpoints.append(os.path.join(os.curdir, 'BLM_HPC_v1_lr1e-5_Bahdanau_50EP'))

tokenizer = transformers.BertTokenizer.from_pretrained(path_to_tokenizer)
columns = ['Model Name', 'BLEU-Score']

print(f"We have {len(checkpoints)} checkpoints")
for i, ckpt in enumerate(checkpoints):
    print(f'Printing score of Checkpoint: {ckpt}')
    if i < 3:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.simple, corpus_list)
    if i > 2 and i < 6:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.luong, corpus_list)
    if i > 5:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.bahdanau, corpus_list)
    scores.append(translator.print_bleu_score())
    print('_____________________________________________________________________________')

print(f"We must have simmilar {len(scores)} BLEU-scores")
plt.bar(checkpoints, scores)
plt.savefig('bleu_scores_new_4.png')


rows = []
for i in range(len(checkpoints)):
    rows.append([checkpoints[i], str(scores[i])])
filename = 'bleu_scores_table_4.csv'


with open(filename, 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(columns)
    csvwriter.writerows(rows)
 
filename_unseen = 'bleu_scores_unseen_4.csv'
scores_unseen = []
unseen_corpus = [path_to_corpus_4]
print(f"We have {len(checkpoints)} checkpoints")

for i, ckpt in enumerate(checkpoints):
    print(f'Printing score of Checkpoint: {ckpt}')
    if i < 3:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.simple, unseen_corpus)
    if i > 2 and i < 6:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.luong, unseen_corpus)
    if i > 5:
        translator = SimpleLanguageTranslator(ckpt, path_to_tokenizer, ModelType.bahdanau, unseen_corpus)
    scores_unseen.append(translator.print_bleu_score())
    print('_____________________________________________________________________________')

rows_unseen = []
for i in range(len(checkpoints)):
    rows_unseen.append([checkpoints[i], str(scores_unseen[i])])

print(f"We must have simmilar {len(scores)} BLEU-scores")
plt.bar(checkpoints, scores_unseen)
plt.savefig('bleu_scores_unseen_4.png')

with open(filename_unseen, 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(columns)
    csvwriter.writerows(rows_unseen)


# with open(path_to_corpus, 'r', encoding='utf-8') as file:
#
#     lines = file.readlines()
#     for line in lines:
#
#         x, y = line.split(sep='\t')
#         y = tokenizer.tokenize(y)
#         corpus.append(y)
#
# with open(path_to_corpus_2, 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         x, y = line.split(sep='\t')
#         y = y.replace('\n', '').split(' ')
#         corpus.append(y)
#
#
# with open(path_to_corpus_3, 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         x, y = line.split(sep='\t')
#         y = y.replace('\n', '').split(' ')
#
#         corpus.append(y)
#
# print(corpus[0])
#
# hypothese = [['Herr', 'oder', 'Frau', '.']]
# score = 0
# # score = corpus_bleu([corpus], [['In', 'welchem', 'Land', 'sind', 'Sie', 'geboren?']])
#
# score = corpus_bleu([[corpus[0]]], hypothese)
#
#
# print(score)