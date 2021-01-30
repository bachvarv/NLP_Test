import csv

import numpy as np

# import tensorflow as tf

from data.Dictionary import Dictionary
from data.utils_functions import one_hot

FILE1 = '../glove/training/Win3_Glove10_Epoch10.csv'

FILE22 = '../word2vec/training/Win10_W2VSKIP300_Epoch10.csv'
FILES500 = '../word2vec/training/Win10_W2VSKIP200_Epoch500.csv'

FILE3 = '../glove/training/Win10_Glove300_Epoch10.csv'
FILEG50 = '../glove/training/Win10_Glove200_Epoch50.csv'
FILEG500 = '../glove/training/Win10_Glove300_Epoch500.csv'

FILE41 = '../word2vec/training/Win10_W2VCBOW300_Epoch10.csv'
FILEC500 = '../word2vec/training/Win10_W2VSCBOW200_Epoch500.csv'

my_dict={}

''' Glove'''
# dictionary = Dictionary()
# dictionary.open_file(FILE1)
skip_dic = Dictionary()
skip_dic.open_file(FILES500)

cbow_dic = Dictionary()
cbow_dic.open_file(FILEC500)

glove_dic = Dictionary()
glove_dic.open_file(FILEG500)

# glove_dic.display_pca_scatterplot_2D()
# cbow_dic.display_pca_scatterplot_2D()
# skip_dic.display_pca_scatterplot_2D()

# glove_dic.plotWords()
# cbow_dic.plotWords()
skip_dic.plotWords()

# print('glove bulgaria:', glove_dic.get_sim_by_key('bulgaria', 5))
# print('skip bulgaria:', skip_dic.get_sim_by_key('bulgaria', 5))
# print('cbow bulgaria:', cbow_dic.get_sim_by_key('bulgaria', 5))
# # print('glove:', dictionary.sum_of_values('language', 'natural'))
# print('glove romanized+balgariya:', glove_dic.sum_of_values('romanized', 'balgariya'))
# print('skip romanized+balgariya:', skip_dic.sum_of_values('romanized', 'balgariya'))
# print('cbow romanized+balgariya:', cbow_dic.sum_of_values('romanized', 'balgariya'))
#
# print('skip republic:', skip_dic.get_sim_by_key('republic', 5))
# print('glove republic:', glove_dic.get_sim_by_key('republic', 5))
# print('cbow republic:', cbow_dic.get_sim_by_key('republic', 5))
#
# print('glove war:', glove_dic.get_sim_by_key('war', 5))
# print('skip war:', skip_dic.get_sim_by_key('war', 5))
# print('cbow war:', cbow_dic.get_sim_by_key('war', 5))
#
# print('skip war + russoturkish:', skip_dic.sum_of_values('war', 'russoturkish', 5))
# print('sofia + capital:', skip_dic.sum_of_values('capital', 'sofia')[0])
# print('cbow war + russoturkish:', cbow_dic.sum_of_values('war', 'russoturkish', 5))
# print('glove war + russoturkish:', glove_dic.sum_of_values('war', 'russoturkish', 5))
#
# print('glove war:', glove_dic.get_sim_by_key('asen', 5))
# print('skip war:', skip_dic.get_sim_by_key('asen', 5))
# print('cbow war:', cbow_dic.get_sim_by_key('asen', 5))

# print('glove:', dictionary.sum_of_values('processing', dictionary.sub_of_values('fun', 'exciting')[0]))
# print('w2v:', w2v_dic.sum_of_values('processing', dictionary.sub_of_values('fun', 'exciting')[0]))

#
# print(dictionary.get_similar(
#     dictionary.sum_of_values(dictionary.sum_of_values('natural', 'language'),
#                               dictionary.sub_of_values('fun', 'exciting')),2))
