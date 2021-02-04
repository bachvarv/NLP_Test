import collections
import operator

import numpy as np
import csv

from data.utils_functions import cosine_sim
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import pandas as pd


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

    def display_pca_scatterplot_2D(self):

        words = list(self.d.keys())
        # word_vectors = np.array([self.d[w] for w in words])
        # print(words[:3], word_vectors[:3])


        # print(two_dim[:3])

        data = []
        count = 0

        # for i in range(len(words)):
        for i in range(15,30):
            word_labels = self.get_sim_by_key(words[i], 5)
            print("{}: {}".format(words[i], word_labels))
            two_dim = PCA(random_state=0).fit_transform(np.array([self.d[w] for w in word_labels]))[:, :2]
            print("{}: {}".format(word_labels, two_dim))
            trace = go.Scatter(
                x=two_dim[:, 0],
                y=two_dim[:, 1],
                text=word_labels,
                name=words[i],
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': 2
                }
            )

            # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
            # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

            data.append(trace)
            # count = count + 5

        # trace_input = go.Scatter(
        #     x=two_dim[count:, 0],
        #     y=two_dim[count:, 1],
        #     text=words[count:],
        #     name='input words',
        #     textposition="top center",
        #     textfont_size=20,
        #     mode='markers+text',
        #     marker={
        #         'size': 10,
        #         'opacity': 1,
        #         'color': 'black'
        #     }
        # )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        # data.append(trace_input)

        # Configure the layout

        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            showlegend=True,
            legend=dict(
                x=1,
                y=0.5,
                font=dict(
                    family="Courier New",
                    size=25,
                    color="black"
                )),
            font=dict(
                family=" Courier New ",
                size=15),
            autosize=False,
            width=1000,
            height=1000
        )

        plot_figure = go.Figure(data=data, layout=layout)
        plot_figure.show()


    def plotWords(self):
        words = list(self.d.keys())

        # Create our Data Frame
        df = pd.DataFrame(self.d)
        vec = np.array(list(self.d.values()))
        vec = vec.transpose()
        print(vec.shape)
        print(df.shape)
        # print(df.head())

        # Compute the Corrolation Matrix
        xCorr = df.corr()
        # print(xCorr.head())

        # Compute eigenvalues and eigen vectors
        values, vectors = np.linalg.eig(xCorr)

        #Sorting the eigen vectors coresponding to eigen values in descending order
        args = (-values).argsort()
        values = values[args]
        vectors = vectors[:, args]

        # Taking the first two components
        new_vectors = vectors[:, :2]

        # Projecting it onto new dimension with 2 axis
        neww_X = np.dot(xCorr, new_vectors)

        # print(neww_X.shape)

        plt.figure(figsize=(30, 30))
        plt.scatter(neww_X[:, 0], neww_X[:, 1])#, linewidths=10, color='blue')

        # print(vectors[-1, 0])
        plt.xlabel("PC1", size=15)
        plt.ylabel("PC2", size=15)
        plt.title("Word Embedding Space", size=20)
        
        for i, word in enumerate(words):
            plt.annotate(word, xy=(neww_X[i, 0], neww_X[i, 1]))

        plt.show()
        plt.savefig()

