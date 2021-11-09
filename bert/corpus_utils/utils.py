import math
import os
import random
from abc import ABC

from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
# import gluon


def _read_text(dir):
    file_name = os.path.join(dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()

        paragraphs = [line.strip().lower().split(' . ')
                      for line in lines if len(line.split(' . ')) >= 2]

        random.shuffle(paragraphs)
    return paragraphs


'''
    These are preparation methods for the NSP training data
'''


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() > 0.5:
        is_next = True
    else:
        next_sentence = random.choice(paragraphs)
        if type(next_sentence[0] == list):
            next_sentence = next_sentence[0]
        is_next = False
    return sentence, next_sentence, is_next


def _get_tokens_and_segments(sentence_a, sentence_b):
    tokens = ['<cls>'] + sentence_a + ['<sep>']
    segments = [0] * (len(sentence_a) + 2)
    if sentence_b is not None:
        tokens += sentence_b + ['sep']
        segments += [1] * (len(sentence_b) + 1)

    return tokens, segments


def get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        sentence_a, sentence_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        if len(sentence_a) + len(sentence_b) + 3 > max_len:
            continue
        token, segment = _get_tokens_and_segments(sentence_a, sentence_b)
        nsp_data_from_paragraph.append((token, segment, is_next))
    return nsp_data_from_paragraph


''' 
    These are preperation methos for the MLM training data
'''


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []

    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # change the token for a totally different one
            if random.random() > 0.5:
                masked_token = vocab[random.randint(0, len(vocab)- 1)]
            else:
                masked_token = mlm_input_tokens[mlm_pred_position]

        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []

    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue

        candidate_pred_positions.append(i)

    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = \
        _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)

    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # print(token_ids[:5])
        padding_tokens = [vocab['<pad>']] * (max_len - len(token_ids))
        result = token_ids + padding_tokens
        all_token_ids.append(
            np.array(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype='int32'))

        all_segments.append(np.array(segments + [0] * (max_len - len(token_ids)), dtype='int32'))
        valid_lens.append(np.array(len(token_ids), dtype='int32'))
        all_pred_positions.append(
            np.array(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) +
                     [0.0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='float32'))
        all_mlm_labels.append(
            np.array(
                mlm_pred_label_ids + [0] *
                (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))


    return (all_token_ids, all_segments, valid_lens,
            all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


def _tokenize(paragraphs, word='word'):
    return [d2l.tokenize(paragraph, token=word) for paragraph in paragraphs]


def _create_examples(par, min_freq, max_len):
    examples = []
    paragraphs = _tokenize(par, 'word')

    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    vocab = d2l.Vocab(sentences,
                      min_freq=min_freq, reserved_tokens=['<sep>', '<cls>', '<pad>', '<mask>'])

    for paragraph in paragraphs:
        examples.extend(get_nsp_data_from_paragraph(paragraph,
                                                    paragraphs, max_len))

    examples = [(get_mlm_data_from_tokens(tokens, vocab) + (segments, is_next))
                for tokens, segments, is_next in examples]

    (all_token_ids, all_segments, valid_lens, all_pred_positions,
     all_mlm_weights, all_mlm_labels, nsp_labels) = pad_bert_inputs(examples, max_len, vocab)

    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

def download_data_wiki(batch_size, max_len, min_freq, path):
    paragraphs = _read_text(path)
    # print(paragraphs)
    # train_examples = _create_examples(paragraphs, min_freq, max_len)
    # print(train_examples)
    # print(batch_size)
    train_set = Wiki2Corpus(paragraphs, max_len, batch_size)

    train_examples = tf.data.Dataset\
        .from_generator(train_set.gen_batches,
                        (tf.int32, tf.int32, tf.int32,
                         tf.int32, tf.float32, tf.int32, tf.bool),
                        # all_token_ids,        # segments
                         ((batch_size, max_len), (batch_size, max_len),
                        # valid_lens           all_pred_positions
                         (batch_size), (batch_size, math.ceil(0.15*max_len)),
                        # all mlm_weights, all_mlm_labels, nsp_labels
                         (batch_size, math.ceil(0.15*max_len)), (batch_size, math.ceil(0.15*max_len)), (batch_size)))

    # train_set = tf.data.Dataset.from_tensor_slices(train_examples)
    # train_iter = tf.data.Dataset.from_tensor_slices(train_set) \
    #     .batch(batch_size).shuffle(len(train_set.all_token_ids))
    return train_examples, train_set.vocab


class Wiki2Corpus(tf.keras.utils.Sequence):
    def __init__(self, paragraphs, max_len, batch_size):
        self.batch_size = batch_size
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]

        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, reserved_tokens=['<pad>', '<cls>', '<sep>', '<mask>'])
        examples = []

        for paragraph in paragraphs:
            examples.extend(get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))

        examples = [(get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions,
         self.all_mlm_weights, self.all_mlm_labels, self.nsp_labels) = pad_bert_inputs(examples,
                                                                                       max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx],
                self.all_segments[idx],
                self.valid_lens[idx],
                self.all_pred_positions[idx],
                self.all_mlm_weights[idx],
                self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return math.ceil(len(self.all_token_ids)/ self.batch_size)

    def gen_batches(self):
        # print(len(self.all_token_ids))
        while True:
            indexes = np.arange(len(self.all_token_ids))
            np.random.shuffle(indexes)
            for seq in range(len(self) - 1):
                token_ids, segments = [], []
                valid_lens, all_pred_positions = [], []
                mlm_weights, mlm_labels = [], []
                nsp_labels = []
                # print(seq)
                # print(seq+1)
                for i in indexes[np.arange(seq*self.batch_size, (seq+1)*self.batch_size)]:
                    token_ids.append(self.all_token_ids[i])
                    segments.append(self.all_segments[i])
                    valid_lens.append(self.valid_lens[i])
                    all_pred_positions.append(self.all_pred_positions[i])
                    mlm_weights.append(self.all_mlm_weights[i])
                    mlm_labels.append(self.all_mlm_labels[i])
                    nsp_labels.append(self.nsp_labels[i])

                    # print(token_ids)
                    # print(segments)

                yield (token_ids, segments, valid_lens, all_pred_positions, \
                      mlm_weights, mlm_labels, nsp_labels)

            break

