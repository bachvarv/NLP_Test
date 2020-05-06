import re

import numpy as np


def read_text(path, window):
    temp = []
    ts = []
    d = dict()
    file = open(path)
    words = file.read().lower()
    words = re.sub(r'[^\w\s]', '', words)
    words = words.replace('_', '')
    words = words.split()
    hot_index = 0
    for i in words:
        if i not in temp:
            temp.append(i)

    for i in temp:
        d[i] = np.zeros(shape=(len(temp),)).astype(float)
        d[i][hot_index] = float(1)
        hot_index += 1

    size = len(words)
    keys = list(words)
    for i in range(size):
        t = list()
        t.append([keys[i]])
        labels = list()
        for b in range(i - window, i):
            if b < 0 or b == i:
                continue
            labels.append(keys[b])
        for f in range(i, i + (window + 1)):
            if f >= size or f == i:
                continue
            labels.append(keys[f])

        t.append(labels)
        ts.append(t)

    return d, ts


def read_from_file(path):
    file = open(path, mode='r')

    for line in file.readlines():
        print(line)


def cosine_sim(A, B):
    dot_prod = np.dot(A, B)
    mag_A = np.sqrt(np.dot(A, A))
    mag_B = np.sqrt(np.dot(B, B))
    mag_mul = mag_A * mag_B
    cos_AB = dot_prod / mag_mul
    return cos_AB


def fill_dict(arr_lab, arr_val):
    d = {}
    i = 0
    for s in arr_lab:
        d[s] = arr_val[i]
        i += 1
    return d


def create_training_arrays(training_set, dictionary):
    train_x = []
    train_y = []
    for item in training_set:
        print(item)
        train_x.append([dictionary[x] for x in item[0]])
        train_y.append([dictionary[y] for y in item[1]])

    return np.array(train_x), np.array(train_y)
    # print(train_x, train_y)
