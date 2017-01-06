#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import numpy as np
import random, sys


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")
#all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")


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
        #line = to_wide_lower(line)
        line = line.decode("utf-8")
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
    max_time, batch_size, state_size, learning_rate = 16, 50000, 64, 0.01
    path = "data/lib_ru"
    vocabulary_size = len(all_syms)
    # read and convert data
    data = np.asarray(list(iterate_symbols(path)))
    # create variables and graph
    x = tf.placeholder(tf.int32, [batch_size, max_time])
    apply_x = tf.placeholder(tf.int32, [1, 1])                  # we will apply it sym-by-sym
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, vocabulary_size]))
    b = tf.Variable(tf.random_normal([vocabulary_size]))
    # create learning graph
    state_x = tf.placeholder(tf.float32, [batch_size, state_size])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, tf.one_hot(x, vocabulary_size, on_value = 1.0), initial_state = state_x, dtype = tf.float32)
    output = tf.reshape(output, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    output = tf.nn.softmax(output)
    y = tf.placeholder(tf.int32, [batch_size, max_time])
    # define loss and optimizer
    ohy = tf.one_hot(y, vocabulary_size, on_value = 1.0)
    loss = -tf.reduce_sum(tf.mul(output, ohy))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    # create output tensors for applying
    apply_state_x = tf.placeholder(tf.float32, [1, state_size])
    with tf.variable_scope('train', reuse = True):
        output, apply_state = tf.nn.dynamic_rnn(gru, tf.one_hot(apply_x, vocabulary_size, on_value = 1.0), initial_state = apply_state_x, dtype = tf.float32)
    output = tf.reshape(output, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, 1, vocabulary_size])
    output = tf.nn.softmax(output)
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    steps = (len(data) - 1) / (batch_size * max_time)
    while True:
        l, c, i = 0.0, 0, 0
        cur_state = sess.run(gru.zero_state(batch_size, tf.float32))
        for i in xrange(steps):
            #print (i + 1) * max_time + steps * max_time * (batch_size - 1) + 1, len(data), steps
            batch_x, batch_y = [], []
            for j in xrange(batch_size):
                batch_x.append(data[i * max_time + j * steps * max_time : (i + 1) * max_time + j * steps * max_time])
                batch_y.append(data[i * max_time + j * steps * max_time + 1 : (i + 1) * max_time + j * steps * max_time + 1])
            if cnt < 10:
                cur_state = sess.run(gru.zero_state(batch_size, tf.float32))
            _, cur_state, _l = sess.run([optimizer, state, loss], feed_dict = {x: batch_x, y: batch_y, state_x: cur_state})
            _l /= (batch_size * max_time)
            l += _l
            c += 1
            print "Progress: %.1f%%, loss: %f" % (i * 100.0 / steps, -_l)
            sys.stdout.flush()
        print make_sample(sess, apply_x, apply_state_x, output, apply_state, sess.run(gru.zero_state(1, tf.float32)), 1000)
        print "Loss: %.5f\n" % (-l / c)
        sys.stdout.flush()
        cnt += 1


# entry point
if __name__ == "__main__":
    main()

