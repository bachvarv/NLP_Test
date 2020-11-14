import collections
import operator

import numpy as np
import csv

from data.utils_functions import cosine_sim


class Dictionary:

    def __init__(self, dic = None):
        self.d = {}
        if dic is not None:
            self.d = dic

    def get(self, key):
        return self.d[key]

    def get_v(self, key):
        return self.get_similar(self.d.get(key), 1)

    def get_sim_by_key(self, key, top):
        return self.get_similar(self.d.get(key), top)

    def get_similar(self, value, top):
        results = {}
        for k, v in self.d.items():
            sim = cosine_sim(v, value)
            results[k] = sim

        sorted_results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        d = collections.OrderedDict(sorted_results)
        return list(d)[0:top]

    def sum_of_values(self, key1, key2, top=1):
        # print(self.d[key1])
        return self.get_similar(self.d.get(key1) + self.d.get(key2), top)

    def sub_of_values(self, key1, key2):
        # print(self.d[key1])
        return self.get_similar(self.d.get(key1) - self.d.get(key2), 1)

    def set_dictionary(self, dic):
        self.d = dic

    def save(self, filename):
        with open(filename+'.csv', 'w+') as file:
            f = csv.writer(file, delimiter=',')
            f.writerow(['key', 'value'])
            # np.save(filename+'.npy', self.d)
            for key, value in self.d.items():
                f.writerow([key, value])

    def open_file(self, path):
        my_dict = {}
        with open(path, mode='r') as infile:
            next(infile)
            reader = csv.reader(infile)
            for key, value in reader:
                v = value.replace('\n', '')
                v = v.replace('[', '')
                v = v.replace(']', '')
                # v = v.replace(' ', '\\x0')
                my_dict[key] = np.fromstring(v, sep=' ').astype('float64')

        self.d = my_dict