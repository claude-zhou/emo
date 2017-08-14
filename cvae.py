import tensorflow as tf
import numpy as np

from helpers import safe_exp
from bleu import compute_bleu
from tensorflow.python.layers import core as layers_core
from model_helpers import Embedding, build_bidirectional_rnn, create_rnn_cell
from yellowfin import YFOptimizer

import tensorflow.contrib.seq2seq as seq2seq

class CVAE(object):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_unit,
                 latent_dim,
                 emoji_dim,
                 batch_size,
                 kl_ceiling,
                 bow_ceiling,
                 decoder_layer=1,
                 start_i=1,
                 end_i=2,
                 beam_width=0,
                 maximum_iterations=50,
                 max_gradient_norm=5,
                 lr=1e-3,
                 dropout=0.2,
                 num_gpu=2,
                 cell_type=tf.nn.rnn_cell.GRUCell,
                 is_seq2seq=False):
        self.end_i = end_i
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.num_unit = num_unit
        self.dropout = dropout
        self.beam_width = beam_width
        self.cell_type = cell_type

        self.emoji = tf.placeholder(tf.int32, shape=[batch_size], name="emoji")
        self.ori = tf.placeholder(tf.int32, shape=[None, batch_size], name="original_tweet")  # [len, batch_size]
        self.ori_len = tf.placeholder(tf.int32, shape=[batch_size], name="original_tweet_length")
        self.rep = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_tweet")
        self.rep_len = tf.placeholder(tf.int32, shape=[batch_size], name="response_tweet_length")
        self.rep_input = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_start_tag")
        self.rep_output = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_end_tag")

        self.kl_weight = tf.placeholder(tf.float32, shape=(), name="kl_weight")
        # self.inferring = tf.placeholder_with_default(False, shape=(), name='inferring')

        self.placeholders = [
            self.emoji,
            self.ori, self.ori_len,
            self.rep, self.rep_len, self.rep_input, self.rep_output
        ]

        with tf.variable_scope("embeddings"):
            embedding = Embedding(vocab_size, embed_size)

            ori_emb = embedding(self.ori)  # [max_len, batch_size, embedding_size]
            rep_emb = embedding(self.rep)
            rep_input_emb = embedding(self.rep_input)
            emoji_emb = embedding(self.emoji)  # [batch_size, embedding_size]

        with tf.variable_scope("original_tweet_encoder"):
            ori_encoder_output, ori_encoder_state = build_bidirectional_rnn(
                num_unit, ori_emb, self.ori_len, cell_type, num_gpu)
            ori_encoder_state_flat = tf.concat(
                [ori_encoder_state[0], ori_encoder_state[1]], axis=1)

        emoji_vec = tf.layers.dense(emoji_emb, emoji_dim, activation=tf.nn.tanh)
        # emoji_vec = tf.ones([batch_size, emoji_dim], tf.float32)
        condition_flat = tf.concat([ori_encoder_state_flat, emoji_vec], axis=1)

        with tf.variable_scope("response_tweet_encoder"):
            _, rep_encoder_state = build_bidirectional_rnn(
                num_unit, rep_emb, self.rep_len, cell_type, num_gpu)
            rep_encoder_state_flat = tf.concat(
                [rep_encoder_state[0], rep_encoder_state[1]], axis=1)

        with tf.variable_scope("representation_network"):
            rn_input = tf.concat([rep_encoder_state_flat, condition_flat], axis=1)
            # simpler representation network
            # r_hidden = rn_input
            r_hidden = tf.layers.dense(
                rn_input, latent_dim, activation=tf.nn.relu, name="r_net_hidden")  # int(1.6 * latent_dim)
            r_hidden_mu = tf.layers.dense(
                r_hidden, latent_dim, activation=tf.nn.relu)  # int(1.3 * latent_dim)
            r_hidden_var = tf.layers.dense(
                r_hidden, latent_dim, activation=tf.nn.relu)
            self.mu = tf.layers.dense(
                r_hidden_mu, latent_dim, activation=tf.nn.tanh, name="q_mean")
            self.log_var = tf.layers.dense(
                r_hidden_var, latent_dim, activation=tf.nn.tanh, name="q_log_var")

        with tf.variable_scope("prior_network"):
            # simpler prior network
            # p_hidden = condition_flat
            p_hidden = tf.layers.dense(
                condition_flat, int(0.62 * latent_dim), activation=tf.nn.relu, name="r_net_hidden")
            p_hidden_mu = tf.layers.dense(
                p_hidden, int(0.77 * latent_dim), activation=tf.nn.relu)
            p_hidden_var = tf.layers.dense(
                p_hidden, int(0.77 * latent_dim), activation=tf.nn.relu)
            self.p_mu = tf.layers.dense(
                p_hidden_mu, latent_dim, activation=tf.nn.tanh, name="p_mean")
            self.p_log_var = tf.layers.dense(
                p_hidden_var, latent_dim, activation=tf.nn.tanh, name="p_log_var")

        with tf.variable_scope("reparameterization"):
            self.z_sample = self.mu + tf.exp(self.log_var / 2.) * tf.random_normal(shape=tf.shape(self.mu))
            self.q_z_sample = self.p_mu + tf.exp(self.p_log_var / 2.) * tf.random_normal(shape=tf.shape(self.p_mu))

        if is_seq2seq:  # vanilla seq2seq
            self.z_sample = self.z_sample - self.z_sample
            self.q_z_sample = self.q_z_sample - self.q_z_sample

        with tf.variable_scope("attention"):
            attention_state = tf.concat([ori_encoder_output[0], ori_encoder_output[1]], axis=2)
            attention_state = tf.transpose(attention_state, [1, 0, 2])

            attention_mechanism = seq2seq.BahdanauAttention(
                num_unit, attention_state, memory_sequence_length=self.ori_len)

        with tf.variable_scope("decoder_train") as decoder_scope:
            if decoder_layer == 2:
                train_decoder_init_state = (
                    tf.concat([self.z_sample, ori_encoder_state[0], emoji_vec], axis=1),
                    tf.concat([self.z_sample, ori_encoder_state[1], emoji_vec], axis=1)
                )
                dim = latent_dim + num_unit + emoji_dim
                decoder_cell_no_drop = tf.nn.rnn_cell.MultiRNNCell(
                    [create_rnn_cell(dim, 0, cell_type, num_gpu),
                     create_rnn_cell(dim, 1, cell_type, num_gpu)])
            else:
                train_decoder_init_state = tf.concat([self.z_sample, ori_encoder_state_flat, emoji_vec], axis=1)
                dim = latent_dim + 2 * num_unit + emoji_dim
                decoder_cell_no_drop = create_rnn_cell(dim, 0, cell_type, num_gpu)

            decoder_cell_no_drop = seq2seq.AttentionWrapper(
                decoder_cell_no_drop,
                attention_mechanism,
                attention_layer_size=None)

            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                cell=decoder_cell_no_drop, input_keep_prob=(1.0 - self.dropout))

            helper = seq2seq.TrainingHelper(
                rep_input_emb, self.rep_len + 1, time_major=True)
            projection_layer = layers_core.Dense(
                vocab_size, use_bias=False, name="output_projection")
            decoder = seq2seq.BasicDecoder(
                decoder_cell, helper,
                decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=train_decoder_init_state),
                output_layer=projection_layer)
            train_outputs, _, _ = seq2seq.dynamic_decode(
                decoder,
                output_time_major=True,
                swap_memory=True,
                scope=decoder_scope
            )
            self.logits = train_outputs.rnn_output

        with tf.variable_scope("decoder_infer") as decoder_scope:
            # normal_sample = tf.random_normal(shape=(batch_size, latent_dim))

            if decoder_layer == 2:
                infer_decoder_init_state = (
                    tf.concat([self.q_z_sample, ori_encoder_state[0], emoji_vec], axis=1),
                    tf.concat([self.q_z_sample, ori_encoder_state[1], emoji_vec], axis=1)
                )
            else:
                infer_decoder_init_state = tf.concat([self.q_z_sample, ori_encoder_state_flat, emoji_vec], axis=1)

            start_tokens = tf.fill([batch_size], start_i)
            end_token = end_i

            if beam_width > 0:
                # Replicate encoder info beam_width times
                infer_decoder_init_state = seq2seq.tile_batch(
                    infer_decoder_init_state, multiplier=beam_width)
                decoder = seq2seq.BeamSearchDecoder(
                    cell=decoder_cell_no_drop,
                    embedding=embedding.coder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_cell_no_drop.zero_state(
                        batch_size, tf.float32).clone(cell_state=infer_decoder_init_state),
                    beam_width=beam_width,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding.coder, start_tokens, end_token)
                decoder = seq2seq.BasicDecoder(
                    decoder_cell_no_drop,
                    helper,
                    decoder_cell_no_drop.zero_state(batch_size, tf.float32).clone(cell_state=infer_decoder_init_state),
                    output_layer=projection_layer  # applied per timestep
                )

            # Dynamic decoding
            infer_outputs, _, _ = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=maximum_iterations,
                output_time_major=True,
                swap_memory=True,
                scope=decoder_scope
            )
            if beam_width > 0:
                self.result = infer_outputs.predicted_ids
            else:
                self.result = infer_outputs.sample_id

        with tf.variable_scope("loss"):
            max_time = tf.shape(self.rep_output)[0]
            with tf.variable_scope("reconstruction"):
                # TODO: use inference decoder's logits to compute recon_loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # ce = [len, batch_size]
                    labels=self.rep_output, logits=self.logits)
                # rep: [len, batch_size]; logits: [len, batch_size, vocab_size]
                target_mask = tf.sequence_mask(
                    self.rep_len + 1, max_time, dtype=self.logits.dtype)
                # time_major
                target_mask_t = tf.transpose(target_mask)
                self.recon_loss = tf.reduce_sum(cross_entropy * target_mask_t) / batch_size

            with tf.variable_scope("latent"):
                # without prior network
                # self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.log_var) + self.mu ** 2 - 1. - self.log_var, 0)
                self.kl_loss = 0.5 * tf.reduce_sum(
                    tf.exp(self.log_var - self.p_log_var) +
                    (self.mu - self.p_mu) ** 2 / tf.exp(self.p_log_var) - 1. - self.log_var + self.p_log_var,
                    axis=0)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

            with tf.variable_scope("bow"):
                # self.bow_loss = self.kl_weight * 0
                mlp_b = layers_core.Dense(
                    vocab_size, use_bias=False, name="MLP_b")
                # is it a mistake that we only model on latent variable?
                latent_logits = mlp_b(tf.concat(
                    [self.z_sample, ori_encoder_state_flat, emoji_vec], axis=1))  # [batch_size, vocab_size]
                latent_logits = tf.expand_dims(latent_logits, 0)  # [1, batch_size, vocab_size]
                latent_logits = tf.tile(latent_logits, [max_time, 1, 1])  # [max_time, batch_size, vocab_size]

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # ce = [len, batch_size]
                    labels=self.rep_output, logits=latent_logits)
                self.bow_loss = tf.reduce_sum(cross_entropy * target_mask_t) / batch_size

            if is_seq2seq:
                self.kl_loss = self.kl_loss - self.kl_loss
                self.bow_loss = self.bow_loss - self.bow_loss

            self.loss = tf.reduce_mean(
                self.recon_loss + self.kl_loss * self.kl_weight * kl_ceiling + self.bow_loss * bow_ceiling)

        # Calculate and clip gradients
        with tf.variable_scope("optimization"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)

            # Optimization
            optimizer = tf.train.AdamOptimizer(lr)
            self.update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))

    def infer_and_eval(self, batches, sess):
        sess = sess or sess.get_default_session()

        # inference
        reference_corpus = []
        generation_corpus = []

        recon_loss_l = []
        kl_loss_l = []
        bow_loss_l = []
        word_count = 0

        for batch in batches:
            feed_dict = dict(zip(self.placeholders, batch))
            feed_dict[self.kl_weight] = 1.
            # feed_dict[self.inferring] = True

            gen_digits, recon_loss, kl_loss, bow_loss = sess.run(
                [self.result, self.recon_loss, self.kl_loss, self.bow_loss],
                feed_dict=feed_dict)
            recon_loss_l.append(recon_loss)
            kl_loss_l.append(kl_loss)
            bow_loss_l.append(bow_loss)

            rep_m = batch[3]
            rep_len = batch[4]
            for i, leng in enumerate(rep_len):
                word_count += leng
                ref = list(rep_m[0:leng, i])
                reference_corpus.append([ref])

                out = []
                if self.beam_width > 0:
                    for digit in gen_digits[:, i, 0]:
                        if digit == self.end_i:
                            break
                        out.append(digit)
                else:
                    for digit in gen_digits[:, i]:
                        if digit == self.end_i:
                            break
                        out.append(digit)
                generation_corpus.append(out)

        total_recon_loss = np.mean(recon_loss_l)
        total_kl_loss = np.mean(kl_loss_l)
        total_bow_loss_l = np.mean(bow_loss_l)
        perplexity = safe_exp(total_recon_loss * self.batch_size * len(recon_loss_l) / word_count)

        bleu_score, precisions, bp, ratio, translation_length, reference_length = compute_bleu(
            reference_corpus, generation_corpus)
        for i in range(len(precisions)):
            precisions[i] *= 100

        return (total_recon_loss, total_kl_loss, total_bow_loss_l,
                perplexity, bleu_score * 100, precisions,
                generation_corpus)

    def train_update(self, batch, sess, weight):
        sess = sess or sess.get_default_session()
        feed_dict = dict(zip(self.placeholders, batch))
        feed_dict[self.kl_weight] = weight

        _, recon_loss, kl_loss, bow_loss = sess.run(
            [self.update_step, self.recon_loss, self.kl_loss, self.bow_loss], feed_dict=feed_dict)
        return recon_loss, kl_loss, bow_loss
