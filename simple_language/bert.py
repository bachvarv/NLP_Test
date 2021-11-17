import os
import random
import tarfile

import requests
import transformers
import tensorflow as tf
import subprocess


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def pad_array(arr, size, item):
    # arr_to_append = [item for _ in range(size - len(arr))]
    # for _ in range(size - len(arr)):
    if size > arr.shape[1]:
        arr = tf.concat([arr, tf.constant([[item for _ in range(size - arr.shape[1])]], dtype='int32')], axis=1)
    return arr


cfg = dict(
    max_sentence=128
)

# Clone the

# if not os.path.isdir('bert_en_cased_L-12_H-768_A-12_4'):
#     file = 'bert_en_cased_L-12_H-768_A-12_4.tar.gz'
#     url = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4?tf-hub-format=compressed'
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(file, 'wb') as f:
#             f.write(response.raw.read())
#     file = tarfile.open(file)
#     file.extractall('./bert_en_cased_L-12_H-768_A-12_4')
#     file.close()
#     model = ZipFile(BytesIO(url_open.read()))
#     model.extractall()

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
text = "Die Steinbrech-Felsennelke ist eine überwinternd grüne, ausdauernde krautige Pflanze " \
       "und erreicht Wuchshöhen von 10 bis 30 Zentimetern"

# Testing the tokenizer
tokens = tokenizer.tokenize(text)
pre_model = tokenizer.prepare_for_model(tokens)
test_item = tokenizer(text, return_tensors='tf')

# print(tokens)
# print(pre_model)
# print(test_item)

# TODO: preprocess the inputs to be in three different dictionaries
#   one with all the ids
#   one with all the token_type_ids
#   and one with all the attention_masks
# The arrays and the dictionary for the input and the label
inp_dict = dict()
input_ids = []
token_type_ids = []
attention_masks = []
labels = []

eval_ids = []
eval_type_ids = []
eval_masks = []
eval_labels = []

input_arr = []
eval_arr = []
label_arr = []
eval_label_arr = []
path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus'), 'einfache_sprache.csv')
longest_cand = 0
with open(path_to_corpus, 'r') as file:
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        x, y = line.split(sep='\t')
        token_x = tokenizer(x, return_tensors='tf')
        token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
        if random.randint(0, 100) > 10:
            input_arr.append(token_x)
            label_arr.append(token_y)

        else:
            eval_arr.append(token_x)
            eval_label_arr.append(token_y)

        size_x = token_x['input_ids'].shape[1]
        size_y = token_y['input_ids'].shape[1]
        if size_x > size_y:
            if size_x > longest_cand:
                longest_cand = size_x
        else:
            if size_y > longest_cand:
                longest_cand = size_y

# print(len(input_arr))
for i in range(len(input_arr)):
    # print(pad_array(input_arr[i]['input_ids'], longest_cand, 0))
    input_ids.append(pad_array(input_arr[i]['input_ids'], longest_cand, 0))
    token_type_ids.append(pad_array(input_arr[i]['token_type_ids'], longest_cand, 0))
    attention_masks.append(pad_array(input_arr[i]['attention_mask'], longest_cand, 1))
    labels.append(pad_array(label_arr[i]['input_ids'], longest_cand, 0))

for i in range(len(eval_arr)):
    eval_ids.append(pad_array(eval_arr[i]['input_ids'], longest_cand, 0))
    eval_type_ids.append(pad_array(eval_arr[i]['token_type_ids'], longest_cand, 0))
    eval_masks.append(pad_array(eval_arr[i]['attention_mask'], longest_cand, 1))
    eval_labels.append(pad_array(eval_label_arr[i]['input_ids'], longest_cand, 0))
    # pad_array(input_arr[i]['input_ids'], longest_cand, 0)
    # pad_array(input_arr[i]['token_type_ids'], longest_cand, 0)
    # pad_array(input_arr[i]['attention_mask'], longest_cand, 1)
    # pad_array(label_arr[i]['input_ids'], longest_cand, 0)
    # pad_array(label_arr[i]['token_type_ids'], longest_cand, 0)
    # pad_array(label_arr[i]['attention_mask'], longest_cand, 1)

# print(longest_cand)

print(len(input_ids))
print(len(eval_ids))
inp_arr = dict(input_ids=input_ids,
               token_type_ids=token_type_ids,
               attention_mask=attention_masks)

eval_inp_arr = dict(input_ids=eval_ids,
                    token_type_ids=eval_type_ids,
                    attention_mask=eval_masks)

# print(inp_arr)
# creating the dataset
dataset = tf.data.Dataset.from_tensor_slices((inp_arr, labels))
eval_dataset = tf.data.Dataset.from_tensor_slices((eval_inp_arr, eval_labels))
# print(dataset)
# print(eval_dataset)

# Creating the model
inp_layer = dict(
    input_ids=tf.keras.layers.Input(shape=longest_cand, dtype=tf.int32),
    token_type_ids=tf.keras.layers.Input(shape=longest_cand, dtype=tf.int32),
    attention_mask=tf.keras.layers.Input(shape=longest_cand, dtype=tf.int32)
)

encoder_input = transformers.TFBertModel.from_pretrained(path_to_model)
outputs = encoder_input(inp_layer)
net = outputs.last_hidden_state
net = tf.keras.layers.Dropout(0.1)(net)
output = tf.keras.layers.Dense(tokenizer.vocab_size, activation=None)(net)
model = tf.keras.Model(inp_layer, output)
# out = encoder_input(input_ids).last_hidden_state
# pooler_output = encoder_input(input_ids).pooler_output

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[metric])

# Checkpoint Manger and Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'model_checkpoint')
ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')
# model.fit(dataset, batch_size=1, epochs=2)


model.evaluate(eval_dataset, batch_size=1)
model.evaluate(dataset, batch_size=1)
# ckpt_manager.save()^

saved_model = os.path.join(os.curdir, 'saved_model')
model.save(saved_model)
# test = model(input_arr[10])
# print(input_arr[10])
# # print(test)
# guess = tf.argmax(test, axis=-1)
# # print(guess)
# print(input_arr[10]['input_ids'])
# print(tokenizer.convert_ids_to_tokens(input_arr[10]['input_ids'][0]))
# print(tokenizer.convert_ids_to_tokens(guess[0]))


# print(out.last_hidden_state)
# encoder_input.set_weights()
# encoder_output = encoder_input(inp)

# print(encoder_output)
