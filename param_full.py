import tensorflow as tf
from helpers import build_vocab
import os
from os.path import join
from model import CVAE

"""directories"""
os.chdir("../data/")

data_name = "full"      #
input_dir = data_name + "_input"
output_dir = join(input_dir, data_name + "_output")
train_out_f = join(output_dir, "train.out")
test_out_f = join(output_dir, "test.out")
# default
vocab_f = "vocab.ori"
train_ori_f = join(input_dir, "train.ori")
train_rep_f = join(input_dir, "train.rep")
test_ori_f = join(input_dir, "test.ori")
test_rep_f = join(input_dir, "test.rep")

# build vocab
word2index, index2word = build_vocab(join(input_dir, vocab_f))

"""hyper params for building the graph"""
start_i, end_i = word2index['<s>'], word2index['</s>']
vocab_size = len(word2index)
batch_size = 128   #
num_unit = 128      # num_unit should be equal to embed_size
embed_size = 128    #
latent_dim = 256    #
num_gpu = 2

# default
lr = 1e-3
max_gradient_norm = 5
maximum_iterations = 50
beam_width = 0
num_layer = 1
dropout = 0.2

# GRUCell won't have multiple kinds of state. Wouldn't have to flatten its state before concatenation
cell_type = tf.nn.rnn_cell.GRUCell

"""hyper params for running the graph"""
num_epoch = 12      #
test_step = 100      #
anneal_ratio = 0.2  #

cvae = CVAE(vocab_size, embed_size, num_unit, latent_dim, batch_size, start_i, end_i, num_gpu=num_gpu)
