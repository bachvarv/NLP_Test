import os
import subprocess
import tarfile
import random

import requests
import tensorflow as tf

import transformers


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def pad_array(arr, size, item):
    # arr_to_append = [item for _ in range(size - len(arr))]
    # for _ in range(size - len(arr)):
    if size > arr.shape[1]:
        arr = tf.concat([arr, tf.constant([[item for _ in range(size - arr.shape[1])]], dtype='int32')], axis=1)
    return arr


cfg = dict(
    max_sentence=128,
    hidde_layer_size=768
)

# Download the pre-trained model and the tokenizer
if not os.path.isdir('bert-base-german-cased'):
    git_lfs_url = 'https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz'
    file = 'git-lfs-linux-amd64-v3.0.2.tar.gz'
    response = requests.get(git_lfs_url, stream=True)
    if response.status_code == 200:
        with open(file, 'wb') as f:
            f.write(response.raw.read())
            file = tarfile.open(file)
            file.extractall('./git_lfs')
            file.close()
    subprocess.run(['./install.sh'], cwd='git_lfs')

    subprocess.run(['git', 'lfs', 'install'], cwd='git_lfs')

    print("Cloning Bert-Base-German-cased!")
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)

    subprocess.run(['git', 'lfs', 'pull'], cwd='bert-base-german-cased')


# Path to vocabulary
model_dir = "bert-base-german-cased"
path_to_vocab = os.path.join(model_dir, "vocab.txt")
path_to_weights = os.path.join(model_dir, "tf_model.h5")
path_to_model = os.path.join(os.curdir, model_dir)

# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)

# Preprocess the corpus
input_arr = []
label_arr = []
eval_arr = []
eval_label_arr = []
path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus'), 'spd_programm_einfache_sprache.csv')
with open(path_to_corpus, 'r') as file:
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        x, y = line.split(sep='\t')
        tokenized_x = tokenizer.tokenize(x)
        print(tokenized_x)
        tokenized_y = tokenizer.tokenize(y)
        if random.randint(0, 100) > 20:
            input_arr.append(x)
            label_arr.append(y)


        else:
            eval_arr.append(x)
            eval_label_arr.append(y)



with open(path_to_corpus_2, 'r') as file:
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        x, y = line.split(sep='\t')
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)

        if random.randint(0, 100) > 20:
            input_arr.append(x)
            label_arr.append(y)


        else:
            eval_arr.append(x)
            eval_label_arr.append(y)



# arr = np.reshape(input_arr, newshape=[1, len(input_arr)])
arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# print(arr_inp)

arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')

dataset = tf.data.Dataset.from_tensor_slices((arr_inp['input_ids'], arr_lab['input_ids']))

eval_dataset = tf.data.Dataset.from_tensor_slices(arr_eval['input_ids'])

# Creating the model
encoder_input = tf.keras.Input(shape=(None, cfg['max_sentence']))
encoder = tf.keras.layers.LSTM(cfg['hidde_layer_size'], return_state=True)
encoder_output, encoder_final_memory_state, encoder_final_carry_state = encoder(encoder_input)
encoder_states = [encoder_final_memory_state, encoder_final_carry_state]

decoder_input = tf.keras.Input(shape=(None, cfg['max_sentence']))
decoder_lstm = tf.keras.layers.LSTM(cfg['hidde_layer_size'], return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')
output = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_input, decoder_input], output)

optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,
              loss_function= loss_function,
              metrics=[metric])


