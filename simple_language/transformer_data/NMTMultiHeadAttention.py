import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class NMTMultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(NMTMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attentions = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model))

        output = self.dense(concat_attentions)

        return output

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product(self, q, k, v, mask):
        """Calculate the attention weights.
          q, k, v must have matching leading dimensions.
          k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
          The mask has different shapes depending on its type(padding or look ahead)
          but it must be broadcastable for addition.

          Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable
                  to (..., seq_len_q, seq_len_k). Defaults to None.

          Returns:
            output, attention_weights
          """

        matmul_qk = tf.matmul(q, k, transpose_b=True)   # (..., seq_len_q, seq_len_k)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk/ tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)    # (..., seq_len_q, depth_v)

        return output, attention_weights