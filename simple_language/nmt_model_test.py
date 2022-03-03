import os
import random
import subprocess
import transformers
import tensorflow as tf

from simple_language.transformer_data.NMTModel import NMTModel


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
vocab_size = tokenizer.vocab_size

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

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
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            label_arr.append('[CLS]' + y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append('[CLS]' + y)
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
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            label_arr.append('[CLS]' + y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append('[CLS]' + y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

with open(path_to_corpus_3, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            label_arr.append('[CLS]' + y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append('[CLS]' + y)

# creating the dataset

#
# def tokenize_function(examples):
#     return tokenizer(examples, max_length=128, return_tensors='tf')


# arr = np.reshape(input_arr, newshape=[1, len(input_arr)])
arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# print(arr_inp)

arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# print(arr_eval)
### Old Way
# print(non_tokenized)

dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_inp['input_ids'],
         token_type_ids=arr_inp['token_type_ids'],
         attention_mask=arr_inp['attention_mask']), arr_lab['input_ids'].numpy())).batch(cfg['batch_size'])

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_eval['input_ids'],
         token_type_ids=arr_eval['token_type_ids'],
         attention_mask=arr_eval['attention_mask']), arr_eval_label['input_ids'].numpy())).batch(1)

model = NMTModel(model_name, cfg['hidden_layer_size'], cfg['transformer_heads'], cfg['max_sentence'], vocab_size)

model.compile(optimizer=model.optimizer,
              loss=model.loss,
              metrics=['accuracy'])

# Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'model_nmt_mix_v1')
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')

ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')


test_input = tokenizer('Anrede', max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
test_target = tokenizer('[CLS]', max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')

inputs = (dict(input_ids=test_input['input_ids'],
              token_type_ids=test_input['token_type_ids'],
              attention_mask=test_input['attention_mask']), test_target['input_ids'])

output = model(inputs, False)
print(output.shape)
print(output)
output = tf.argmax(output, axis=-1)
print(output)
print(tokenizer.decode(output[0]))

model.summary()



# history = model.fit(dataset, epochs=5)
# model.evaluate(eval_dataset)
model.train_step(dataset, epochs=70)
ckpt_manager.save()

sentence = ['Anrede']
pred = ['[CLS]']
token = tokenizer(sentence, max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
pred_token = tokenizer(pred, max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
test_inp = (dict(input_ids=token['input_ids'],
              token_type_ids=token['token_type_ids'],
              attention_mask=token['attention_mask']), pred_token['input_ids'])
predicted = model(test_inp, False)
print(predicted.shape)
arg_max = tf.argmax(predicted, axis=-1)
print(arg_max)
print(tokenizer.decode(arg_max[0]))

# save_history(history, path_to_history)


