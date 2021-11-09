import os

path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
with open(path_to_corpus, 'r') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
