import tensorflow as tf
from model_helpers import Embedding, build_bidirectional_rnn, xavier
import os
from os import makedirs
from os.path import join, dirname
from time import strftime, gmtime
from helpers import build_vocab, build_dis_data, generate_dis_batches, print_out
import numpy as np
import json

class TweetDiscriminator(object):

    def __init__(self, num_unit, batch_size, vocab_size, embed_size,
                 cell_type=tf.nn.rnn_cell.BasicLSTMCell,
                 num_gpu=2,
                 lr=0.001):

        self.label = tf.placeholder(tf.int32, shape=[batch_size], name="label")
        self.text = tf.placeholder(tf.int32, shape=[None, batch_size], name="embed-tweet")  # [max_len, batch_size]
        self.len = tf.placeholder(tf.int32, shape=[batch_size], name="tweet_length")

        with tf.variable_scope("embeddings"):
            embedding = Embedding(vocab_size, embed_size)
            text_embed = embedding(self.text)

        with tf.variable_scope("text-encoder"):
            _, encoder_state = build_bidirectional_rnn(
                num_unit, text_embed, self.len, cell_type, num_gpu)
            text_vec = tf.concat([encoder_state[0], encoder_state[1]], axis=1)

        with tf.variable_scope("turing-result"):
            logits = tf.layers.dense(text_vec, 2, activation=None, kernel_initializer=xavier)
            self.prob = tf.nn.softmax(logits)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits))

        with tf.variable_scope("accuracy"):
            accuracy = tf.nn.in_top_k(logits, self.label, k=1)
            self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        with tf.variable_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(lr)
            self.update_step = optimizer.minimize(self.loss)

    def train_update(self, batch, sess):
        sess = sess or sess.get_default_session()
        text = batch[0]
        length = batch[1]
        label = batch[2]

        _, loss, accuracy = sess.run(
            [self.update_step, self.loss, self.accuracy],
            feed_dict={self.text: text, self.label: label, self.len: length})
        return loss, accuracy

    def eval(self, batches, sess):
        sess = sess or sess.get_default_session()
        loss_l = []
        accuracy_l = []

        for batch in batches:
            text = batch[0]
            length = batch[1]
            label = batch[2]

            loss, accuracy = sess.run(
                [self.loss, self.accuracy],
                feed_dict={self.text: text, self.label: label, self.len: length})

            loss_l.append(loss)
            accuracy_l.append(accuracy)
        return float(np.mean(loss_l)), float(np.mean(accuracy_l))

if __name__ == '__main__':
    from params.full import *
    num_epoch = 6
    test_step = 50

    # for machine samples
    from collections import Counter

    c = Counter()
    os.chdir("../data/full_64_input/dis_pretrain")
    output_dir = strftime("%m-%d_%H-%M-%S", gmtime())

    # build vocab
    word2index, index2word = build_vocab("vocab.ori")
    start_i, end_i = word2index['<s>'], word2index['</s>']
    vocab_size = len(word2index)

    discriminator = TweetDiscriminator(num_unit, batch_size, vocab_size, embed_size,
                                       cell_type=tf.nn.rnn_cell.GRUCell, num_gpu=2, lr=0.001)
    
    train_data = build_dis_data("human_train.txt", "machine_train.txt", word2index)
    test_data = build_dis_data("human_test.txt", "machine_test.txt", word2index)
    test_batches = generate_dis_batches(test_data, batch_size, False)

    print_out("*** DATA READY ***")

    makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
    log_f = open(join(output_dir, "log.log"), "w")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        best_f = join(output_dir, "best_accuracy.txt")

        global_step = best_step = 1
        start_epoch = best_epoch = 1
        best_loss = 1000.
        sess.run(tf.global_variables_initializer())

        for epoch in range(start_epoch, num_epoch + 1):
            train_batches = generate_dis_batches(train_data, batch_size, True)

            loss_l = []
            accuracy_l = []
            
            for batch in train_batches:
                loss, accuracy = discriminator.train_update(batch, sess)
                loss_l.append(loss)
                accuracy_l.append(accuracy)

                if global_step % test_step == 0:
                    time_now = strftime("%m-%d %H:%M:%S", gmtime())
                    print_out('epoch:\t%d\tstep:\t%d\tbatch-loss/accuracy:\t%.3f\t%.1f\t\t%s' %
                              (epoch, global_step,
                               np.mean(loss_l), np.mean(accuracy_l) * 100, time_now),
                              f=log_f)
                if global_step % (test_step * 10) == 0:
                    loss, accuracy = discriminator.eval(test_batches, sess)
                    print_out('EPOCH-\t%d\tSTEP-\t%d\tTEST-loss/accuracy/accuracy5-\t%.3f\t%.1f' %
                              (epoch, global_step,
                               loss, accuracy * 100),
                              f=log_f)

                    if best_loss >= loss:
                        best_loss = loss

                        best_epoch = epoch
                        best_step = global_step

                        # save breakpoint
                        path = join(output_dir, "breakpoints/best_test_loss.ckpt")
                        save_path = saver.save(sess, path)

                        # save best epoch/step
                        best_dict = {
                            "loss": best_loss, "epoch": best_epoch, "step": best_step, "accuracy": accuracy}
                        with open(path, "w") as f:
                            f.write(json.dumps(best_dict, indent=2))
                global_step += 1

            loss, accuracy = discriminator.eval(train_batches, sess)
            print_out('EPOCH!\t%d\tTRAIN!\t%d\tTRAIN-loss/accuracy-\t%.3f\t%.1f' %
                      (epoch, global_step,
                       np.mean(loss_l), np.mean(accuracy_l) * 100),
                      f=log_f)

    log_f.close()
