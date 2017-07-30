import numpy as np
import random
import tensorflow as tf
import math

"""build data"""
def build_vocab(vocab_path):
    vocab_file = open(vocab_path, encoding="utf-8")
    vocab_data = vocab_file.readlines()
    vocab_file.close()

    index2word = dict()
    word2index = dict()
    for index, line in enumerate(vocab_data):
        word = line.rstrip()
        index2word[index] = word
        word2index[word] = index
    return word2index, index2word

def build_data(ori_path, rep_path, word2index):
    unk_i = word2index['<unk>']

    ori_file = open(ori_path, encoding="utf-8")
    ori_tweets = ori_file.readlines()
    ori_file.close()

    rep_file = open(rep_path, encoding="utf-8")
    rep_tweets = rep_file.readlines()
    rep_file.close()

    assert(len(ori_tweets) == len(rep_tweets))
    emojis = []
    ori_seqs = []
    rep_seqs = []
    for line in ori_tweets:
        words = line.split()
        tweet = [word2index.get(word, unk_i) for word in words[1:]]
        ori_seqs.append(tweet)
        emojis.append(word2index.get(words[0], unk_i))
    for line in rep_tweets:
        tweet = [word2index.get(word, unk_i) for word in line.split()]
        rep_seqs.append(tweet)
    return [
        emojis,
        ori_seqs,
        rep_seqs
    ]

def generate_one_batch(data_l, start_i, end_i, s, e):
    emojis = data_l[0]
    ori_seqs = data_l[1]
    rep_seqs = data_l[2]

    if e is None:
        e = len(emojis)

    emoji_vec = np.array(emojis[s:e], dtype=np.int32)

    ori_lengths = np.array([len(seq) for seq in ori_seqs[s:e]])
    max_ori_len = np.max(ori_lengths)
    ori_matrix = np.zeros([max_ori_len, e - s], dtype=np.int32)

    for i, seq in enumerate(ori_seqs[s:e]):
        for j, elem in enumerate(seq):
            ori_matrix[j, i] = elem

    rep_lengths = np.array([len(seq) for seq in rep_seqs[s:e]])
    max_rep_len = np.max(rep_lengths)
    rep_matrix = np.zeros([max_rep_len, e - s], dtype=np.int32)
    rep_input_matrix = np.zeros([max_rep_len + 1, e - s], dtype=np.int32)
    rep_output_matrix = np.zeros([max_rep_len + 1, e - s], dtype=np.int32)

    rep_input_matrix[0, :] = start_i
    for i, seq in enumerate(rep_seqs[s:e]):
        for j, elem in enumerate(seq):
            rep_matrix[j, i] = elem
            rep_input_matrix[j + 1, i] = elem
            rep_output_matrix[j, i] = elem
        rep_output_matrix[len(seq), i] = end_i

    return [
            emoji_vec,
            ori_matrix,
            ori_lengths,
            rep_matrix,
            rep_lengths,
            rep_input_matrix,
            rep_output_matrix
    ]

def batch_generator(data_l, start_i, end_i, batch_size, permutate=True):
    # shuffle
    emojis = data_l[0]
    ori_seqs = data_l[1]
    rep_seqs = data_l[2]

    if permutate:
        all_input = list(zip(emojis, ori_seqs, rep_seqs))

        random.shuffle(all_input)
        new_all = list(zip(*all_input))
    else:
        new_all = [emojis, ori_seqs, rep_seqs]

    data_size = len(emojis)
    num_batches = int((data_size - 1.) / batch_size) + 1

    rtn = []
    for batch_num in range(num_batches):
        e = min((batch_num + 1) * batch_size, data_size)
        s = e - batch_size
        assert(s >= 0)
        rtn.append(generate_one_batch(new_all, start_i, end_i, s, e))
    return rtn

"""utils"""
def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def generate_graph():
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open('miscellanies/graphpb.txt', 'w') as f:
        f.write(graphpb_txt)
    exit(0)
