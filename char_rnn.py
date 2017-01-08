#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import numpy as np
import random, sys


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")
#all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")
#all_syms = "Helo".decode("utf-8")


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
        if c % 1000 == 0:
            print c / 1000000
            sys.stdout.flush()
            if c / 1000 >= 1:
                break
        #line = to_wide_lower(line)
        line = line.decode("utf-8")
        for ch in line:
            if ch not in all_syms:
                continue
            yield all_syms.index(ch)


# input shape: batch*time*state
# output shape: batch*time*vocabulary
# multiplies last dimention by `w` and adds `b`
def make_projection(x, state_size, max_time, vocabulary_size, w, b):
    output = tf.reshape(x, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    output = tf.nn.softmax(output)
    return output


def choose_random(distr):
    #print " ".join(map(lambda t: "%.2f" % t, distr))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    return int(np.searchsorted(cs, np.random.rand(1) * s))


def make_sample(sess, x, state_x, op, state_op, cur_state, n, seed):
    seed = [all_syms.index(ch) for ch in seed.decode("utf-8")]
    for ch in seed:
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[ch]], state_x: cur_state})
    res = res[0][0]
    #print res
    cur_sym = choose_random(res)
    result = all_syms[cur_sym]
    for k in xrange(n):
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[cur_sym]], state_x: cur_state})
        res = res[0][0]
        cur_sym = choose_random(res)
        result += all_syms[cur_sym]
    return result.encode("utf-8")


def make_sample1(sess, x, state_x, op, state_op, l, cur_state, n, seed, max_time):
    result = seed.decode("utf-8")
    seed = [all_syms.index(ch) for ch in seed.decode("utf-8")]
    for ch in seed:
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[ch] * max_time], state_x: cur_state, l: [1]})
    #print cur_state.shape
    res = res[0][0]
    #print "\t".join(map(lambda u: "%.2f" % u, res))
    cur_sym = choose_random(res)
    result += all_syms[cur_sym]
    for k in xrange(n):
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[cur_sym] * max_time], state_x: cur_state, l: [1]})
        res = res[0][0]
        #print "\t".join(map(lambda u: "%.2f" % u, res))
        cur_sym = choose_random(res)
        result += all_syms[cur_sym]
    return result.encode("utf-8")


def print_matr(matr):
    for vec in matr:
        print "\t".join(map(lambda u: "%.5f" % u, vec))


def print_matrs(matrs):
    for matr in matrs:
        print_matr(matr)
        print


def test_projection():
    max_time, state_size, vocabulary_size = 2, 3, 2
    x = tf.placeholder(tf.float32, [None, max_time, state_size])
    w = tf.Variable([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
    b = tf.Variable([1.0, 1.0])
    y = make_projection(x, state_size, max_time, vocabulary_size, w, b)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print_matrs(sess.run(y, feed_dict = {x: [[[1, 2, 3], [1, 1, 1]]]}))


def test_onehot():
    y = tf.placeholder(tf.int32, [None, 3])
    ohy = tf.one_hot(y, 5, on_value = 1.0)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print_matrs(sess.run(ohy, feed_dict = {y: [[1, 2, 3], [0, 1, 2], [2, 3, 4]]}))


def test():
    test_projection()
    test_onehot()

 
# do all stuff
def main():
    # define params
    max_time, batch_size, state_size, learning_rate = 25, 1000, 200, 0.001
    path = "data/all"
    vocabulary_size = len(all_syms)

    # read and convert data
    data = np.asarray(list(iterate_symbols(path)))

    # create variables and graph
    x = tf.placeholder(tf.int32, [None, max_time])
    lengths = tf.placeholder(tf.int32, [None])
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, vocabulary_size]))
    b = tf.Variable(tf.random_normal([vocabulary_size]))

    # create learning graph
    state_x = tf.placeholder(tf.float32, [None, state_size])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, tf.one_hot(x, vocabulary_size, on_value = 1.0), initial_state = state_x, sequence_length = lengths, dtype = tf.float32)
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
    apply_x = tf.placeholder(tf.int32, [None, max_time])                    # we will apply it sym-by-sym
    apply_state_x = tf.placeholder(tf.float32, [None, state_size])
    with tf.variable_scope('train', reuse = True):
        apply_output, apply_state = tf.nn.dynamic_rnn(gru, tf.one_hot(apply_x, vocabulary_size, on_value = 1.0), initial_state = apply_state_x, sequence_length = lengths, dtype = tf.float32)
    apply_output = tf.reshape(apply_output, [-1, state_size])
    apply_output = tf.add(tf.matmul(apply_output, w), b)
    apply_output = tf.reshape(apply_output, [-1, max_time, vocabulary_size])
    apply_output = tf.nn.softmax(apply_output)

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
            #print
            #print batch_x
            #print batch_y
            #print
            if cnt < 10:
                cur_state = sess.run(gru.zero_state(batch_size, tf.float32))
            _, cur_state, _l, dump = sess.run([optimizer, state, loss, tf.mul(output, ohy)], feed_dict = {x: batch_x, y: batch_y, state_x: cur_state, lengths: [max_time] * batch_size})
            _l /= (batch_size * max_time)
            l += _l
            c += 1
            #print_matrs(dump)
            print "Progress: %.1f%%, loss: %f" % (i * 100.0 / steps, -_l)
            sys.stdout.flush()
        while True:
            seed = raw_input("Enter begining of string: ") if (-l / c >= 0.3) else "Посадил дед ре"
            if seed == "exit":
                break
            print make_sample1(sess, apply_x, apply_state_x, apply_output, apply_state, lengths, sess.run(gru.zero_state(1, tf.float32)), 100, seed, max_time)
            #print make_sample1(sess, x, state_x, output, state, lengths, sess.run(gru.zero_state(1, tf.float32)), 1000, seed, max_time)
            if -l / c < 0.3:
                break
        print "Loss: %.5f\n" % (-l / c)
        sys.stdout.flush()
        cnt += 1


# entry point
if __name__ == "__main__":
    #test()
    main()

