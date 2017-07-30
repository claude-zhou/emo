# TODO-1: check emoji unknown situation
# TODO-2: print a random generation every epoch
# TODO-3: add BoW trick
# TODO-4: add tricks proposed by Google MT

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

from helpers import build_data, batch_generator
from param_tiny import *
import json

from os import makedirs
from os.path import join, dirname, isfile

from math import tanh

def put_eval(recon_loss, kl_loss, ppl, bleu_score, precisions_list, name):
    print("%s: " % name, end="")
    format_string = '\tloss/ppl: recon-%.3f,kl-%.3f / %.3f BLEU:' + ' %.1f' * 5
    format_tuple = (recon_loss, kl_loss, ppl, bleu_score) + tuple(precisions_list)
    print(format_string % format_tuple)

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
        return tanh(6*progress/ratio-3)+1

if __name__ == '__main__':
    run_from_scratch = True

    train_data = build_data(train_ori_f, train_rep_f, word2index)

    test_data = build_data(test_ori_f, test_rep_f, word2index)
    test_batches = batch_generator(
        test_data, start_i, end_i, batch_size, permutate=False)

    print("*** DATA READY ***")
    makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
    best_f = join(output_dir, "best_bleu.txt")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        total_step = (num_epoch * len(train_data) / batch_size)
        if run_from_scratch or (not isfile(best_f)):
            global_step = best_step = 0
            start_epoch = best_epoch = 0
            best_bleu = 0.
            sess.run(tf.global_variables_initializer())
        else:
            best_bleu, best_epoch, best_step = restore_best(best_f)
            global_step = best_step
            start_epoch = best_epoch
            path = join(output_dir, "breakpoints/epoch_%d_step_%d.ckpt" % (best_epoch, best_step))
            saver.restore(sess, path)

        # generate_graph()
        for epoch in range(start_epoch, num_epoch):
            train_batches = batch_generator(
                train_data, start_i, end_i, batch_size)

            for batch in train_batches:
                """ TRAIN """
                kl_weight = get_kl_weight(global_step, total_step, anneal_ratio)
                recon_loss, kl_loss = cvae.train_update(batch, sess, kl_weight)

                if global_step % test_step == 0:
                    print(
                        'Epoch: %d Step: %d Batch-loss: recon-%.3f,kl-%.3f' % (epoch, global_step, recon_loss, kl_loss))

                    """ BATCH EVAL and INFER """
                    # TRAIN
                    train_recon_loss, train_kl_loss, perplexity, train_bleu_score, precisions, _ = cvae.infer_and_eval(
                        train_batches, batch_size, sess)
                    put_eval(train_recon_loss, train_kl_loss, perplexity, train_bleu_score, precisions, "TRAIN")

                    # TEST
                    test_recon_loss, test_kl_loss, perplexity, test_bleu_score, precisions, _ = cvae.infer_and_eval(
                        test_batches, batch_size, sess)
                    put_eval(test_recon_loss, test_kl_loss, perplexity, test_bleu_score, precisions, "TEST")

                    # get down best
                    # if test_bleu_score >= best_bleu:
                    #     best_bleu = test_bleu_score
                    if train_bleu_score >= best_bleu:  # TODO: train or test?
                        best_bleu = train_bleu_score

                        best_epoch = epoch
                        best_step = global_step

                        path = join(output_dir, "breakpoints/epoch_%d_step_%d.ckpt" % (epoch, global_step))
                        save_path = saver.save(sess, path)
                        # save best epoch/step
                        save_best(best_f, best_bleu, best_epoch, best_step)
                global_step += 1

        """RESTORE BEST MODEL"""
        path = join(output_dir, "breakpoints/epoch_%d_step_%d.ckpt" % (best_epoch, best_step))
        saver.restore(sess, path)

        """GENERATE"""
        # TRAIN
        train_batches = batch_generator(
            train_data, start_i, end_i, batch_size, permutate=False)
        generation_corpus = cvae.infer_and_eval(train_batches, batch_size, sess)[-1]
        write_out(train_out_f, generation_corpus)

        # TEST
        generation_corpus = cvae.infer_and_eval(test_batches, batch_size, sess)[-1]
        write_out(test_out_f, generation_corpus)
