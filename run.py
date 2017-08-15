# TODO-1: check emoji unknown situation
# TODO-2: print a random generation every epoch
# TODO-3: add tricks proposed by Google MT

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

from helpers import build_data, batch_generator, print_out, build_vocab
import json

from time import gmtime, strftime

from os import makedirs
from os.path import join, dirname

from math import tanh

import argparse
from cvae import CVAE

def put_eval(recon_loss, kl_loss, bow_loss, ppl, bleu_score, precisions_list, name, f):
    print_out("%s: " % name, new_line=False, f=f)
    format_string = '\trecon/kl/bow-loss/ppl:\t%.3f\t%.3f\t%.3f\t%.3f\tBLEU:' + '\t%.1f' * 5
    format_tuple = (recon_loss, kl_loss, bow_loss, ppl, bleu_score) + tuple(precisions_list)
    print_out(format_string % format_tuple, f=f)

def write_out(file, corpus):
    with open(file, 'w', encoding="utf-8") as f:
        for seq in corpus:
            to_write = ''
            for index in seq:
                to_write += index2word[index] + ' '
            f.write(to_write + '\n')

def save_best(file, best_bleu, best_epoch, best_step):
    best_dict = {"bleu": best_bleu, "epoch": best_epoch, "step": best_step}
    with open(file, "w") as f:
        f.write(json.dumps(best_dict, indent=2))

def restore_best(file):
    with open(file) as f:
        best_dict = json.load(f)
        best_bleu, best_epoch, best_step = best_dict["bleu"], best_dict["epoch"], best_dict["step"]
    return best_bleu, best_epoch, best_step

def get_kl_weight(global_step, total_step, ratio):
    # python3!
    progress = global_step / total_step
    if progress >= ratio:
        return 1.
    else:
        return tanh(6 * progress / ratio - 3) + 1

if __name__ == '__main__':

    cvae_parser = argparse.ArgumentParser()
    cvae_parser.add_argument("--anneal_ratio", type=float, required=True, help="""\
        hyper param for KL annealing:
        weight of kl_loss rises to 1 after *anneal_ratio* of global steps""")
    cvae_parser.add_argument("--kl_ceiling", type=float, default=1., help="""\
        param that limit kl_loss proportion in the total loss""")
    cvae_parser.add_argument("--bow_ceiling", type=float, default=1., help="""\
        param that limit bow_loss proportion in the total loss""")
    cvae_parser.add_argument("--init_from_dir", type=str, default="")

    # hyper params for running the graph
    cvae_parser.add_argument("--num_epoch", type=int, required=True, )
    cvae_parser.add_argument("--test_step", type=int, required=True, help="""\
        output batch eval every *test_step*
        output test eval every 10 x *test_step*
        output train eval every epoch""")

    cvae_parser.add_argument("--input_dir", type=str, required=True, )
    cvae_parser.add_argument("--param_set", type=str, required=True, help="""\
            tiny/medium/full""")
    cvae_parser.add_argument("--log_fname", type=str, default="log")
    cvae_parser.add_argument("--is_seq2seq", action="store_true", help="""\
            CVAE model or vanilla seq2seq with similar settings""")

    FLAGS, _ = cvae_parser.parse_known_args()

    if FLAGS.param_set == "tiny":
        from params.tiny import *
    elif FLAGS.param_set == "full":
        from params.full import *
    elif FLAGS.param_set == "medium":
        from params.medium import *

    output_dir_name = strftime("%m-%d_%H-%M-%S", gmtime())

    """directories"""
    input_dir = FLAGS.input_dir
    output_dir = join(input_dir, output_dir_name)
    train_out_f = join(output_dir, "train.out")
    test_out_f = join(output_dir, "test.out")
    vocab_f = "vocab.ori"
    train_ori_f = join(input_dir, "train.ori")
    train_rep_f = join(input_dir, "train.rep")
    test_ori_f = join(input_dir, "test.ori")
    test_rep_f = join(input_dir, "test.rep")

    makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
    with open(join(output_dir, "hparams.txt"), "w") as f:
        with open("params/%s.py" % FLAGS.param_set) as ff:
            content = ff.read()
        f.write(content + "\n")
        for key, var in vars(FLAGS).items():
            s = str(key) + " = " + str(var) + "\n"
            f.write(s)

    log_f = open(join(output_dir, "%s.log" % FLAGS.log_fname), "w")

    # build vocab
    word2index, index2word = build_vocab(join(input_dir, vocab_f))
    start_i, end_i = word2index['<s>'], word2index['</s>']
    vocab_size = len(word2index)

    # building graph
    cvae = CVAE(vocab_size, embed_size, num_unit, latent_dim, emoji_dim, batch_size,
                FLAGS.kl_ceiling, FLAGS.bow_ceiling, decoder_layer,
                start_i, end_i, beam_width, maximum_iterations, max_gradient_norm, lr, dropout, num_gpu, cell_type,
                FLAGS.is_seq2seq)

    # building data
    train_data = build_data(train_ori_f, train_rep_f, word2index)

    test_data = build_data(test_ori_f, test_rep_f, word2index)
    test_batches = batch_generator(
        test_data, start_i, end_i, batch_size, permutate=False)

    print_out("*** DATA READY ***")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        total_step = (FLAGS.num_epoch * len(train_data[0]) / batch_size)

        best_f = join(output_dir, "best_bleu.txt")
        global_step = best_step = 1
        start_epoch = best_epoch = 1
        best_bleu = 0.

        if FLAGS.init_from_dir == "":
            sess.run(tf.global_variables_initializer())
        else:
            recover_dir = join(input_dir, FLAGS.init_from_dir)
            best_dir = join(recover_dir, "breakpoints/best_test_bleu.ckpt")
            saver.restore(sess, best_dir)

        # generate_graph()
        for epoch in range(start_epoch, FLAGS.num_epoch + 1):
            train_batches = batch_generator(
                train_data, start_i, end_i, batch_size)

            recon_l = []
            kl_l = []
            bow_l = []
            for batch in train_batches:
                """ TRAIN """
                kl_weight = get_kl_weight(global_step, total_step, FLAGS.anneal_ratio)
                recon_loss, kl_loss, bow_loss = cvae.train_update(batch, sess, kl_weight)
                recon_l.append(recon_loss)
                kl_l.append(kl_loss)
                bow_l.append(bow_loss)

                if global_step % FLAGS.test_step == 0:
                    time_now = strftime("%m-%d %H:%M:%S", gmtime())
                    print_out('epoch:\t%d\tstep:\t%d\tbatch-recon/kl/bow-loss:\t%.3f\t%.3f\t%.3f\t\t%s' %
                              (epoch, global_step, np.mean(recon_l), np.mean(kl_l), np.mean(bow_l), time_now), f=log_f)
                    recon_l = []
                    kl_l = []
                    bow_l = []
                if global_step % (FLAGS.test_step * 10) == 0:
                    """ EVAL and INFER """

                    # TEST
                    (test_recon_loss, test_kl_loss, test_bow_loss,
                     perplexity, test_bleu_score, precisions, _) = cvae.infer_and_eval(test_batches, sess)
                    print_out("EPOCH:\t%d\tSTEP:\t%d\t" % (epoch, global_step), new_line=False, f=log_f)
                    put_eval(
                        test_recon_loss, test_kl_loss, test_bow_loss,
                        perplexity, test_bleu_score, precisions, "TEST", log_f)

                    # get down best
                    if test_bleu_score >= best_bleu and kl_weight == 1.:
                        best_bleu = test_bleu_score
                    # if train_bleu_score >= best_bleu:  # TODO: train or test?
                    #     best_bleu = train_bleu_score
                        best_epoch = epoch
                        best_step = global_step

                        path = join(output_dir, "breakpoints/best_test_bleu.ckpt")
                        save_path = saver.save(sess, path)
                        # save best epoch/step
                        save_best(best_f, best_bleu, best_epoch, best_step)
                global_step += 1

            # TRAIN
            (train_recon_loss, train_kl_loss, train_bow_loss,
             perplexity, train_bleu_score, precisions, _) = cvae.infer_and_eval(train_batches, sess)
            print_out("EPOCH:\t%d\tSTEP:\t%d\t" % (epoch, global_step), new_line=False, f=log_f)
            put_eval(
                train_recon_loss, train_kl_loss, train_bow_loss,
                perplexity, train_bleu_score, precisions, "TRAIN", log_f)

        """RESTORE BEST MODEL"""
        path = join(output_dir, "breakpoints/best_test_bleu.ckpt")
        saver.restore(sess, path)

        """GENERATE"""
        # TRAIN SET
        train_batches = batch_generator(
            train_data, start_i, end_i, batch_size, permutate=False)
        (train_recon_loss, train_kl_loss, train_bow_loss,
         perplexity, train_bleu_score, precisions, generation_corpus) = cvae.infer_and_eval(train_batches, sess)
        write_out(train_out_f, generation_corpus)
        print_out("BEST TRAIN BLEU: %.1f" % train_bleu_score, f=log_f)

        # TEST SET
        generation_corpus = cvae.infer_and_eval(test_batches, sess)[-1]
        write_out(test_out_f, generation_corpus)

    log_f.close()
