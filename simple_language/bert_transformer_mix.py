import os
import subprocess
import tensorflow as tf
import transformers
import random

from simple_language.transformer_data.NMTDecoderLayer import NMTDecoderLayer
from simple_language.transformer_data.NMTEncoderLayer import NMTEncoderLayer
from transformer.data.DecoderLayer import DecoderLayer
from transformer.data.EncoderLayer import EncoderLayer


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def save_history(history, file):

    with open(file, 'a+') as f:
        for i in range(len(history.history['accuracy'])):
            line = f"{history.history['accuracy'][i]}\t{history.history['loss'][i]}\n"
            f.writelines(line)
        f.close()


model_name = os.path.join(os.path.curdir, 'bert-base-german-cased')
path_to_json_file = os.path.join(os.curdir, "history_model_bert_transformer_v6.csv")

if not os.path.isdir(model_name):
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)


# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Config variables
cfg = dict(
    max_sentence=128,
    hidden_layer_size=768,
    batch_size=1,
    transformer_heads=7,
    head_size=32
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
        print(tokenized_x)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
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
            label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append(y)
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
            label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            eval_label_arr.append(y)

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

# The model
# TODO: Incorporate the target mask and target input_ids, to the attention
#   Follow the guide on the tensorflow translation page on the official website of tensorflow

input_layer = dict(
    input_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='input_ids_layer'),
    token_type_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='token_type_ids_layer'),
    attention_mask=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='attention_mask_layer')
)
embedding_layer = transformers.TFBertModel.from_pretrained(model_name)

# Changed the encoder and decoder layer so this will not work!!!
encoder_layers = [NMTEncoderLayer(cfg['hidden_layer_size'],
                                  cfg['head_size'],
                                  cfg['transformer_heads'],
                                  1e-3) for _ in range(cfg['transformer_heads'])]
decoder_layers = [NMTDecoderLayer(cfg['hidden_layer_size'],
                                  cfg['head_size'],
                                  cfg['transformer_heads'],
                                  1e-3) for _ in range(cfg['transformer_heads'])]

dense_layer = tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')

emb_out = embedding_layer(input_layer).last_hidden_state
enc_out = encoder_layers[0](emb_out, emb_out)
for i in range(1, cfg['transformer_heads']):
    enc_out = encoder_layers[i](enc_out, emb_out)

print(enc_out.shape)
dec_out = decoder_layers[0](enc_out, emb_out, enc_out)

for i in range(1, cfg['transformer_heads']):
    dec_out = decoder_layers[i](dec_out, emb_out, enc_out)

dense_out = dense_layer(dec_out)

model = tf.keras.Model(input_layer, dense_out)

# Optimizer, loss function and metrics
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# Testing if it works
sentence = ['Geburtsdatum']
token = tokenizer(sentence, max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
out = model(token)

model.summary()

# Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'model_bert_transformer_v6')
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')
ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')

history = model.fit(dataset, epochs=5)
model.evaluate(eval_dataset)

ckpt_manager.save()

sentence = ['Anrede']
token = tokenizer(sentence, max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
predicted = model(token)
print(predicted.shape)
arg_max = tf.argmax(predicted, axis=2)
print(arg_max)
print(tokenizer.decode(arg_max[0]))

save_history(history, path_to_json_file)

# os.system("shutdown /s /t 1")

