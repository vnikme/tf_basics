#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import numpy as np
import random, sys


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


def is_letter(ch):
    return ch in all_syms


# read lines, yield symbols
def iterate_symbols(path):
    c = 0
    for line in open(path):
        c += 1
        if c % 1000000 == 0:
            print c / 1000000
            sys.stdout.flush()
            if c / 1000000 >= 40:
                break
        line = to_wide_lower(line)
        for ch in line:
            if ch not in all_syms:
                continue
            yield all_syms.index(ch)


def choose_random(distr):
    #print " ".join(map(lambda t: "%.2f" % t, distr))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    return int(np.searchsorted(cs, np.random.rand(1) * s))


def make_sample(sess, x, state_x, op, state_op, cur_state, n):
    cur_sym = all_syms.index(' '.decode("utf-8"))
    cur_state = sess.run(cur_state)
    result = "".decode("utf-8")
    for k in xrange(n):
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[cur_sym]], state_x: cur_state})
        res = res[0][0]
        cur_sym = choose_random(res)
        result += all_syms[cur_sym]
    return result.encode("utf-8")


# do all stuff
def main():
    # define params
    max_time, batch_size, embedding_size, state_size, learning_rate = 128, 10000, 50, 10, 0.001
    path = "data/lib_ru"
    vocabulary_size = len(all_syms)
    # read and convert data
    data = np.asarray(list(iterate_symbols(path)))
    # create variables and graph
    x = tf.placeholder(tf.int32, [None, max_time])
    apply_x = tf.placeholder(tf.int32, [1, 1])                  # we will apply it sym-by-sym
    ew = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -0.01, 0.01))
    embed_tensor = tf.nn.embedding_lookup(ew, x)
    apply_embed_tensor = tf.nn.embedding_lookup(ew, apply_x)
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, vocabulary_size]))
    b = tf.Variable(tf.random_normal([vocabulary_size]))
    # create learning graph
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, embed_tensor, dtype = tf.float32)
    output = tf.reshape(output, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    output = tf.nn.softmax(output)
    y = tf.placeholder(tf.int32, [None, max_time])
    # define loss and optimizer
    ohy = tf.one_hot(y, vocabulary_size, on_value = 1.0)
    loss = -tf.reduce_sum(tf.mul(output, ohy))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    # create output tensors for applying
    state_x = tf.placeholder(tf.float32, [1, state_size])
    with tf.variable_scope('train', reuse = True):
        output, state = tf.nn.dynamic_rnn(gru, apply_embed_tensor, initial_state = state_x, dtype = tf.float32)
    output = tf.reshape(output, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, 1, vocabulary_size])
    output = tf.nn.softmax(output)
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    while True:
        l, c, i = 0.0, 0, 0
        batch_x, batch_y = [], []
        while (i + 1) * max_time < len(data):
            batch_x.append(data[i * max_time : (i + 1) * max_time])
            batch_y.append(data[i * max_time + 1 : (i + 1) * max_time + 1])
            i += 1
            if len(batch_x) == batch_size:
                _, _l = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
                batch_x, batch_y = [], []
                _l /= (batch_size * max_time)
                l += _l
                c += 1
                print "Progress: %.1f%%, loss: %f" % (i * 100.0 * max_time / len(data), -_l)
                sys.stdout.flush()
        print make_sample(sess, apply_x, state_x, output, state, gru.zero_state(1, tf.float32), 100)
        print "Loss: %.5f\n" % (-l / c)
        sys.stdout.flush()
        cnt += 1


# entry point
if __name__ == "__main__":
    main()

