import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CustomScheduler(LearningRateSchedule):

    def get_config(self):
        pass

    def __init__(self, d_model, warmup_step=4000):
        self.warmup_step = warmup_step
        self.d_model = d_model
        self.d_model = tf.cast(d_model, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step*(self.warmup_step ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
