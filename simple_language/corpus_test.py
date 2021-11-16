import os

import transformers

path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
model_dir = "bert-base-german-cased"
path_to_model = os.path.join(os.curdir, model_dir)
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)

input_arr = []
label_arr = []
# read the .csv file
with open(path_to_corpus, 'r', encoding='unicode_escape') as file:
    lines = file.readlines()
    print(lines)
    for line in lines:
        x, y = line.split(sep='\t')
        token_x = tokenizer(x)
        token_y = tokenizer(y)

        input_arr.append(token_x)
        label_arr.append(token_y)

print(input_arr)
print(label_arr)



