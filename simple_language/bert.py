import os
import random
import transformers
import tensorflow as tf
import subprocess
# import official.nlp
# from official.nlp import optimization
# from datasets import load_dataset


config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        lr = cfg['hidde_layer_size']**(-0.5) * min((batch+1)**-(0.5), (batch+1)*(warmup_steps**(-1.5)))
        print(lr)
        easy_model.optimizer.lr = lr


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def set_learn_rate(step):
    step = step + 1
    print(step)
    lr = cfg['hidde_layer_size']**(-0.5) * min(step**-(0.5), step*(warmup_steps**(-1.5)))
    print(lr)
    return lr


def pad_array(arr, size, item):
    # arr_to_append = [item for _ in range(size - len(arr))]
    # for _ in range(size - len(arr)):
    if size > arr.shape[1]:
        arr = tf.concat([arr, tf.constant([[item for _ in range(size - arr.shape[1])]], dtype='int32')], axis=1)
    return arr


cfg = dict(
    max_sentence=128,
    hidde_layer_size=768,
    batch_size=2
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
    #     git_lfs_url = 'https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz'
    #     file = 'git-lfs-linux-amd64-v3.0.2.tar.gz'
    #     response = requests.get(git_lfs_url, stream=True)
    #     if response.status_code == 200:
    #         with open(file, 'wb') as f:
    #             f.write(response.raw.read())
    #             file = tarfile.open(file)
    #             file.extractall('./git_lfs')
    #             file.close()
    #     subprocess.run(['./install.sh'], cwd='git_lfs')
    #
    #     subprocess.run(['git', 'lfs', 'install'], cwd='git_lfs')
    #
    #     print("Cloning Bert-Base-German-cased!")
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)
#
#     subprocess.run(['git', 'lfs', 'pull'], cwd='bert-base-german-cased')


# Path to vocabulary
model_dir = "bert-base-german-cased"
path_to_vocab = os.path.join(model_dir, "vocab.txt")
path_to_weights = os.path.join(model_dir, "tf_model.h5")
path_to_model = os.path.join(os.curdir, model_dir)

# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_model)

# print(tokens)
# print(pre_model)
# print(test_item)

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
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus'), 'spd_programm_einfache_sprache.csv')
longest_cand = 0
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
# print(arr_lab['input_ids'].numpy())

# print(dataset)
# print(dataset)
# print(input_dataset)

# Creating the model

# Create Model
# m = transformers.TFAutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=tokenizer.vocab_size)

# m = transformers.TFAutoModelForQuestionAnswering.from_pretrained(path_to_model)

# Different Way to create a model
inp_layer = dict(
    input_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='input_ids_layer'),
    token_type_ids=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='token_type_ids_layer'),
    attention_mask=tf.keras.layers.Input(shape=(cfg['max_sentence']), dtype=tf.int32, name='attention_mask_layer')
)
model = transformers.TFBertModel.from_pretrained(path_to_model)

output = model(inp_layer)
net = output.last_hidden_state
# Test with a lstm layer
# lstm_layer = tf.keras.layers.LSTM(cfg['hidde_layer_size'], return_state=True, return_sequences=True)
# dense_layer = tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')
# do_layer = tf.keras.layers.Dropout(0.1)(net)
lstm_layer = tf.keras.layers.LSTM(1024, return_state=True, return_sequences=True)
lstm_out, _, _ = lstm_layer(net)
dense_layer = tf.keras.layers.Dense(tokenizer.vocab_size, activation='sigmoid')
dense_output = dense_layer(lstm_out)
soft_max_layer = tf.keras.layers.Softmax()
softmax_out = soft_max_layer(dense_output)
# lstm_output, _, _ = lstm_layer(net)
# dense_output = dense_layer(lstm_output)
# lstm_out, lstm_h, lstm_c = lstm_layer(net)
# print(dense_output)
easy_model = tf.keras.Model(inp_layer, softmax_out)

# d_w = dense_layer.weights
# print(d_w)
# print(dense_layer.weights)
# outputs = encoder_input.bert(inp_layer)
# print(outputs)
# net = outputs.last_hidden_state
# net = tf.keras.layers.Dropout(0.1)(net)
# output = tf.keras.layers.Dense(tokenizer.vocab_size, activation=None)(net)
# model = tf.keras.Model(inp_layer, output)

# Optimizer
num_steps = len(input_arr)
warmup_steps = int(num_steps*0.15)
# opt = optimization.create_optimizer(init_lr=10e-9,
#                                     num_train_steps=num_steps,
#                                     num_warmup_steps=warmup_steps,
#                                     optimizer_type='adamw')
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# opti = transformers.AdamWeightDecay(learning_rate=1e-9)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Changed fromLogits to False
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalCrossentropy()

easy_model.compile(optimizer=opt,
              loss=loss,
              metrics=[metric])

# m.compile(optimizer=opt,
#           loss=loss,
#           metrics=tf.keras.metrics.SparseCategoricalAccuracy())

# Before Training
test = tokenizer(['Leben Sie allein, sind unter 2. keine weiteren Angaben erforderlich. Bitte weiter bei Abschnitt 3.'],
                 max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')

out= easy_model(test)
soft = tf.math.softmax(out)
print(f'Output of the model: {soft}')
arg_max = tf.argmax(soft, axis=2)
print(arg_max)
print(tokenizer.decode(arg_max[0]))

# Checkpoint Manger and Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'model_checkpoint_gru_1024_v5_100Epochs')
# path_to_saved_easy_model = os.path.join(os.curdir, 'saved_easy_model_gru_1024_v3')
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')
ckpt = tf.train.Checkpoint(easy_model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

#   TODO: Make a model whitouth the input restrcitions and check if the accuracy changes and the
#       guesses are better
easy_model.summary()


if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')

st = 1
easy_model.fit(dataset, epochs=200,
               callbacks=[#tf.keras.callbacks.LearningRateScheduler(set_learn_rate),
                          tf.keras.callbacks.LambdaCallback(on_epoch_end=
                                                            lambda epoch, logs: print(f" How sure is the model: {tf.exp(logs['loss'])}")
                                                            )

                          # CustomCallback(),
                          # tf.keras.callbacks.LambdaCallback(on_batch_begin=lambda epoch, logs: print(tf.exp(easy_model.loss.pop())))
                          ])
easy_model.evaluate(eval_dataset)


# d_w2 = dense_layer.weights
# print(d_w2)
# After Training
test = tokenizer(['Leben Sie allein, sind unter 2. keine weiteren Angaben erforderlich. Bitte weiter bei Abschnitt 3.'],
                 max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
test_label = tokenizer(['Wenn Sie allein leben, müssen Sie hier (Nummer 2) nichts mehr schreiben. Schreiben Sie bitte bei Nummer 3 weiter.'],
                       max_length=cfg['max_sentence'], padding='max_length', return_tensors='tf')
print(tokenizer.tokenize('Wenn Sie allein leben, müssen Sie hier (Nummer 2) nichts mehr schreiben. Schreiben Sie bitte bei Nummer 3 weiter.'))
out = easy_model(test)
print(out)
arg_max = tf.argmax(out, axis=2)
print(arg_max)
print(test_label['input_ids'])
print(tokenizer.decode(arg_max[0]))

ckpt_manager.save()
# model.save(path_to_saved_model)
# easy_model.save(path_to_saved_easy_model)

