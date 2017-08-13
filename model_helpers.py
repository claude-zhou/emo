import tensorflow as tf

class Embedding(object):
    def __init__(self, vocab_size, embed_size):
        # TODO: init from embedding
        self.coder = tf.Variable(
            tf.random_normal([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size),
            name='word_embedding',
            dtype=tf.float32)

    def __call__(self, texts):
        return tf.nn.embedding_lookup(self.coder, texts)