import os

from bert.fine_tuning.data.BERTFineTuned import BERTFineTuned
import tensorflow as tf

checkpoint_dir = 'training\\checkpoints'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
bert = BERTFineTuned(latest_checkpoint)

print(bert(['people .']))
