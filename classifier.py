import tensorflow as tf

class EmojiClassifier(object):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 emoji_num,
                 embed_size,
                 num_unit,
                 num_gpu,
                 lr=0.001,
                 dropout=0.2,
                 cell_type=tf.nn.rnn_cell.GRUCell
                 ):
        self.dropout = dropout
        self.num_gpu = num_gpu
        self.cell_type = cell_type

        self.text = tf.placeholder(tf.int32, shape=[None, batch_size], name="text")
        self.len = tf.placeholder(tf.int32, shape=[batch_size], name="text_length")
        self.emoji = tf.placeholder(tf.int32, shape=[batch_size], name="emoji_label")

        xavier = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("embeddings"):
            embed_coder = tf.Variable(
                tf.random_normal([vocab_size, embed_size], -0.5/embed_size, 0.5/embed_size),
                name='word_embedding',
                dtype=tf.float32)
            # [max_time, batch_size, embed_size]
            text_emb = tf.nn.embedding_lookup(embed_coder, self.text)  # [batch_size, embedding_size]

        with tf.variable_scope("bi_rnn_1"):  # difference between var scope and name scope?
            # tuple#2: [max_time, batch_size, num_unit]
            outputs_1, _ = self.build_bidirectional_rnn(num_unit, text_emb, self.len, dtype=tf.float32)

        with tf.variable_scope("bi_rnn_2"):
            rnn2_input = tf.concat([outputs_1[0], outputs_1[1]], axis=2)
            outputs_2, _ = self.build_bidirectional_rnn(num_unit, rnn2_input, self.len, dtype=tf.float32)

        with tf.variable_scope("attention"):
            word_states = tf.concat(
                [outputs_1[0], outputs_1[1], outputs_2[0], outputs_2[1], text_emb], axis=2)  # [max_t, b_sz, h_dim]

            weights = tf.layers.dense(word_states, 1, kernel_initializer=xavier)
            weights = tf.exp(weights)   # [max_len, batch_size, 1]

            # mask superfluous dimensions
            max_time = tf.shape(self.text)[0]
            target_mask = tf.sequence_mask(self.len, max_time, dtype=tf.float32)
            target_mask = tf.expand_dims(
                tf.transpose(target_mask), axis=-1)  # transpose for time_major & expand to be broadcast-able
            weights = weights * target_mask

            # weight regularization
            sums = tf.expand_dims(tf.reduce_sum(weights, axis=0), 0)  # [1, batch_size, 1]
            weights = weights / sums

            weights = tf.transpose(weights, [1, 0, 2])  # [batch_size, max_len, 1]
            word_states = tf.transpose(word_states, [1, 2, 0])  # [batch_size, hdim, max_len]
            text_vec = tf.squeeze(tf.matmul(word_states, weights), axis=2)  # [batch_size, hdim]

        with tf.variable_scope("loss"):
            logits = tf.layers.dense(text_vec, emoji_num, kernel_initializer=xavier)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.emoji, logits=logits))

        with tf.variable_scope("accuracy"):
            top_5_accuracy = tf.nn.in_top_k(logits, self.emoji, k=5)
            self.top_5_accuracy = tf.reduce_mean(tf.cast(top_5_accuracy, tf.float32))

            accuracy = tf.nn.in_top_k(logits, self.emoji, k=1)
            self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        with tf.variable_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(lr)
            self.update_step = optimizer.minimize(self.loss)

    def build_bidirectional_rnn(self, num_unit, inputs, length, dtype, base_gpu=0):
        fw_cell = self.create_rnn_cell(num_unit, base_gpu)
        bw_cell = self.create_rnn_cell(num_unit, (base_gpu + 1))
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=length,
            time_major=True)

        return bi_outputs, bi_state

    def create_rnn_cell(self, num_dim, base_gpu, drop=True):
        # dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        device_str = "/gpu:%d" % (base_gpu % self.num_gpu)
        print(device_str)
        single_cell = self.cell_type(num_dim)
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        if drop:
            single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - self.dropout))
        return single_cell

if __name__ == '__main__':
    c = EmojiClassifier(4, 7, 64, 5, 9, 1)
