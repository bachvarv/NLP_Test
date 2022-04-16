import os
import random
import subprocess
import time

import transformers
import tensorflow as tf
from matplotlib import pyplot as plt

from simple_language.seq2seq_data.bahdanau.BertLanguageModel import BertLanguageModel
from simple_language.transformer_data.MaskedLoss import MaskedLoss


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def save_history(history, file):

    with open(file, 'a+') as f:
        for i in range(len(history.history['accuracy'])):
            line = f"{history.history['accuracy'][i]}\t{history.history['loss'][i]}\n"
            f.writelines(line)
        f.close()

model_name = os.path.join(os.curdir, 'bert-base-german-cased')
path_to_history = os.path.join(os.curdir, 'history_nmt_mix_v1.csv')

if not os.path.isdir(model_name):
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)



# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens(['[GO]'])
vocab_size = tokenizer.vocab_size

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Config variables
cfg = dict(
    max_sentence=128,
    hidden_layer_size=768,
    batch_size=1,
    transformer_heads=12,
    head_size=64
)

# Corpus preparations
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
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus'), 'spd_programm_einfache_sprache_v1.csv')
path_to_corpus_3 = os.path.join(os.path.join(os.curdir, 'corpus'), 'simple_language_openAI.csv')
longest_cand = 0
with open(path_to_corpus, 'r', encoding='utf-8') as file:

    lines = file.readlines()
    # print(len(lines))
    for line in lines:

        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10 or x == 'Anrede':
            input_arr.append(x)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v4
            eval_label_arr.append( y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)
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


with open(path_to_corpus_2, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v4
            eval_label_arr.append(y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

with open(path_to_corpus_3, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            # label_arr.append(y)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)

            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v2
            eval_label_arr.append(y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)

# creating the dataset

#
# def tokenize_function(examples):
#     return tokenizer(examples, max_length=128, return_tensors='tf')


# arr = np.reshape(input_arr, newshape=[1, len(input_arr)])
arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# print(arr_inp)

arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')

dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_inp['input_ids'],
         token_type_ids=arr_inp['token_type_ids'],
         attention_mask=arr_inp['attention_mask']),
    label_arr)).batch(cfg['batch_size'])

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_eval['input_ids'],
         token_type_ids=arr_eval['token_type_ids'],
         attention_mask=arr_eval['attention_mask']),
    eval_label_arr)).batch(1)

# Model
model = BertLanguageModel(cfg['hidden_layer_size'], vocab_size, tokenizer, path_to_model=model_name, bert_trainable=False)

model.compile(optimizer=tf.keras.optimizers.Adam(
                                                     #learning_rate=1e-4
                                                     learning_rate=2e-5
                                                 ),
              loss=MaskedLoss(),
              metrics=['accuracy'])

# Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'BLM_v2_lr2e-5_Bahdanau_15EP')
# SLM_v4 lr=1e-4
# SLM_v5 lr=1e-3 changed loss to be calculated from logits
# SLM_v6_with_VGA lr=1e-3 to run with gpu
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')
# BLM_v1_train_pair_add_dot_for_end_symbol and the learning rate is 2e-5
# BLM_v1 the learning rate is 1e-3
# BLM_v1_lr_1e-3 lr 1e-3 5 EP
# BLM_v1_freeze_BERT_lr1e-4_Bahdanau was trained for 5 Iterations

ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')

result, _ = model((['Geburtsort'], ['[CLS]']))
prediction = tf.argmax(result.logits, axis=-1)
target = tokenizer(['In welchem Land sind Sie geboren?'])['input_ids']
print(target)
print(prediction.numpy())
# print(result.logits)
s = tokenizer.decode(prediction[0])
print(s)
model.summary()

start = time.time()
losses = []
for i in range(1, 16):
    for inp, tar in dataset:
        logs = model.train_step((inp, tar))
        print(logs)
        losses.append(logs['batch_loss'].numpy())
    print(f'Step {i}')
end = time.time()
plt.plot(losses)
plt.show()


print(end - start)

ckpt_manager.save()

result, _ = model((['Anrede'], ['[CLS]']))

prediction = tf.argmax(result.logits, axis=-1)
target = tokenizer(['Herr oder Frau.'])['input_ids']
print(target)
# print(result.logits)
print(prediction)
s = tokenizer.decode(prediction[0])
print(s)
# for inp, tar in dataset:
#     # print(inp, tar)
#     model.train_step((inp, tar))
#     break
