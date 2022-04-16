# TODO: Fix git lfs
#   Pull the changes that were not pulled inside this branch [+]
#   push the stuff [+]
#   Test the model that is trained if it does it's job!!! []
#   Write my professor about the progress! []
import os
import random

import tensorflow as tf
import transformers

from simple_language.EasyLanguageModel import EasyLanguageModel

model_dir = "bert-base-german-cased"
path_to_vocab = os.path.join(model_dir, "vocab.txt")
path_to_weights = os.path.join(model_dir, "tf_model.h5")
path_to_model = os.path.join(os.curdir, model_dir)

# Load tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)

cfg = dict(
    max_sentence=128,
    hidden_layer_size=768,
    path_to_model=path_to_model,
    vocab_size=tokenizer.vocab_size
)


# Create Model
model = EasyLanguageModel(cfg['path_to_model'], cfg)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False)])


# Preprocess Dataset

path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus'), 'spd_programm_einfache_sprache.csv')
input_arr = []
label_arr = []
eval_arr = []
eval_label_arr = []

with open(path_to_corpus, 'r') as file:
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        x, y = line.split(sep='\t')
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 20:
            input_arr.append(x)
            label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append(y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

        # size_x = token_x['input_ids'].shape[1]
        # size_y = token_y['input_ids'].shape[1]
        # if size_x > size_y:
        #     if size_x > longest_cand:
        #         longest_cand = size_x
        # else:
        #     if size_y > longest_cand:
        #         longest_cand = size_y


with open(path_to_corpus_2, 'r') as file:
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        x, y = line.split(sep='\t')
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 20:
            input_arr.append(x)
            label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append(y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

# Create Dataset
arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')

arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')

dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_inp['input_ids'],
         token_type_ids=arr_inp['token_type_ids'],
         attention_mask=arr_inp['attention_mask']), arr_lab['input_ids'].numpy()))

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_eval['input_ids'],
         token_type_ids=arr_eval['token_type_ids'],
         attention_mask=arr_eval['attention_mask']), arr_eval_label['input_ids'].numpy()))

input_ids = tf.reshape(arr_inp['input_ids'][0], shape=(1, 128))
token_type_ids = tf.reshape(arr_inp['token_type_ids'][0], shape=(1, 128))
attention_mask = tf.reshape(arr_inp['attention_mask'][0], shape=(1, 128))
print(input_ids)
inp = dict(input_ids=input_ids,
           token_type_ids=token_type_ids,
           attention_mask=attention_mask)
out = model(inp)
print(out)
