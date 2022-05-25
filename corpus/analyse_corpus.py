import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import transformers

path_to_models = os.path.join(os.curdir, 'bleu_scores_unseen_3.csv')
print(path_to_models)
#path_to_models = os.path.join(os.curdir, 'bleu_scores_table_3.csv')

x_label = []
y_value = []

with open(path_to_models, 'r') as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        label, value = line
        if label == 'Model Name' and value == 'BLEU-Score':
            continue

        x_label.append(label)
        y_value.append(float(value))

df = pd.DataFrame(dict(model=x_label, score=y_value))

df_sorted = df.sort_values('score', ascending=False)
print(df_sorted)


fig, ax = plt.subplots(figsize=(10, 8))
ax.bar('model', 'score', data=df_sorted)
ax.set(title='BLEU-Score für trainierten Modelle',
       xlabel='Model',
       ylabel='Score')

for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment('right')

plt.tight_layout()
# plt.setp(ax.get_xticklabels(), set_rotation=45)
plt.savefig('bleu_scores_unseen_4.png')
plt.show()

