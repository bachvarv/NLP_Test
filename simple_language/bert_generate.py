import json
import random

import tensorflow as tf
import transformers
import subprocess
import os


# from simple_language.bert import git
def git(*args):
    return subprocess.check_call(['git'] + list(args))


def save_history(history, file):

    with open(file, 'a+') as f:
        for i in range(len(history.history['accuracy'])):
            line = f"{history.history['accuracy'][i]}\t{history.history['loss'][i]}\n"
            f.writelines(line)
        f.close()


model_name = os.path.join(os.path.curdir, 'bert-base-german-cased')
model_dir = "bert-base-german-cased"
path_to_vocab = os.path.join(model_dir, "vocab.txt")
path_to_weights = os.path.join(model_dir, "tf_model.h5")
path_to_model = os.path.join(os.curdir, model_dir)
path_to_json_file = os.path.join(os.curdir, "history_model_text_gen_gnw_12heads_48Each_decoder_v1.csv")

if not os.path.isdir(model_name):
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)


# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Configuration variables
cfg = dict(
    max_sentence=128,
    hidde_layer_size=768,
    batch_size=4,
    heads=12
)

# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)
vocab_size = tokenizer.vocab_size

# Procesisng the data

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
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus'), 'spd_programm_einfache_sprache_v1.csv')
path_to_corpus_3 = os.path.join(os.path.join(os.curdir, 'corpus'), 'simple_language_openAI.csv')
longest_cand = 0
with open(path_to_corpus, 'r', encoding='utf-8') as file:

    lines = file.readlines()
    # print(len(lines))
    for line in lines:

        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        tokenized_y = tokenizer.tokenize(y)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:

            # text = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text = x + " [SEP] "
            for token in tokenized_y:
                # print(text)
                input_arr.append(text)
                label_arr.append(token)
                text = text + " " + token
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:

            # text_eval = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text_eval = x + " [SEP] "
            for token in tokenized_y:

                eval_arr.append(text_eval)
                eval_label_arr.append(token)
                text_eval = text_eval + " " + token
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
            # text = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text = x + " [SEP] "
            for token in tokenized_y:
                input_arr.append(text)
                label_arr.append(token)
                text = text + " " + token
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            # text_eval = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text_eval = x + " [SEP] "
            for token in tokenized_y:
                eval_arr.append(text_eval)
                eval_label_arr.append(token)
                text_eval = text_eval + " " + token
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
            # text = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text = x + " [SEP] "
            for token in tokenized_y:
                input_arr.append(text)
                label_arr.append(token)
                text = text + " " + token
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            # text_eval = x + " [ C L S ]" # model_text_gen_guess_next_word_v2
            text_eval = x + " [SEP] "
            for token in tokenized_y:
                eval_arr.append(text_eval)
                eval_label_arr.append(token)
                text_eval = text_eval + " " + token


print(input_arr[:5])
print(label_arr[:5])

print(eval_arr[:5])
print(eval_label_arr[:5])

arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_lab = tokenizer.convert_tokens_to_ids(label_arr)

arr_lab = tf.keras.utils.to_categorical(arr_lab, num_classes=vocab_size)
print(arr_lab.shape)
arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer.convert_tokens_to_ids(eval_label_arr)
arr_eval_label = tf.keras.utils.to_categorical(arr_eval_label, num_classes=vocab_size)
print(arr_eval_label.shape)
dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_inp['input_ids'],
         token_type_ids=arr_inp['token_type_ids'],
         attention_mask=arr_inp['attention_mask']), arr_lab)).batch(cfg['batch_size'])

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_eval['input_ids'],
         token_type_ids=arr_eval['token_type_ids'],
         attention_mask=arr_eval['attention_mask']), arr_eval_label)).batch(1)


# Model
inp_layer = dict(
    input_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='input_ids_layer'),
    token_type_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='token_type_ids_layer'),
    attention_mask=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='attention_mask_layer')
)

encoding_layer = transformers.TFBertModel.from_pretrained(model_name)

lstm_stacks = tf.keras.layers.StackedRNNCells(
    [tf.keras.layers.LSTMCell(int(cfg['hidde_layer_size']/16)) for _ in range(cfg['heads'])
     ]
)
bidirectional_layer = tf.keras.layers.RNN(lstm_stacks, return_state=True)

# bidirectional_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(cfg['hidde_layer_size']/12), # 'model_text_gen_guess_next_word_v4'
# bidirectional_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(cfg['hidde_layer_size']), # 'model_text_gen_guess_next_word_v4'
# bidirectional_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(cfg['hidde_layer_size'])/2, # 'model_text_gen_guess_next_word_v1'
#                                                                          return_sequences=True))
dropout_layer = tf.keras.layers.Dropout(0.01)
# second_lstm_layer = tf.keras.layers.LSTM(int(cfg['hidde_layer_size']/12)) #, return_sequences=True) not categorical
# second_lstm_layer = tf.keras.layers.LSTM(cfg['hidde_layer_size']) #, return_sequences=True) not categorical
dense_layer = tf.keras.layers.Dense(vocab_size, activation='sigmoid')
softmax_layer = tf.keras.layers.Softmax()
encoding_out = encoding_layer(inp_layer).last_hidden_state

bidi_out = bidirectional_layer(encoding_out)
drop_out = dropout_layer(bidi_out[0])
# second_hidden_out = second_lstm_layer(bidi_out)
dense_out = dense_layer(drop_out)
softmax_out = softmax_layer(dense_out)


model = tf.keras.Model(inp_layer, softmax_out)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)# model_text_gen_guess_next_word_v4
# opt = tf.keras.optimizers.Adam(learning_rate=1e-3) #model_text_gen_guess_next_word_higher_lr_v2
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

test = tokenizer(['Leben Sie allein, sind unter 2. keine weiteren Angaben erforderlich. Bitte weiter bei Abschnitt 3.'],
                 max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')

out = model(test)
print(f'Output of the model: {out}')
print(out.shape)
arg_max = tf.argmax(out, axis=1)
print(arg_max)
# print(tokenizer.decode(arg_max[0]))

model.summary()

# Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'model_text_gen_gnw_12heads_48Each_decoder_v1')
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')
ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')

# Train the model
history = model.fit(dataset, epochs=1)
model.evaluate(eval_dataset)

save_history(history, path_to_json_file)

ckpt_manager.save()

# See the model in action
# seed_text = 'Geburtsort [ C L S ]' # model_text_gen_guess_next_word_v2
seed_text = 'Anrede'
for _ in range(10):
    token_list = tokenizer([seed_text], max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
    predicted = model(token_list)
    arg_max = tf.argmax(predicted, axis=1)
    print(arg_max)
    output_word = ""
    output_word += " " + tokenizer.decode(arg_max)
    seed_text += " " + output_word

print(seed_text)