#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random, sys


def eratosphen(n):
    res, mask = [], [False, False] + [True] * (n - 2)
    for i in xrange(2, n):
        if not mask[i]:
            continue
        res.append(i)
        for j in xrange(2 * i, n, i):
            mask[j] = False
    return res, mask


def generate_data(bits):
    x, y = [], []
    _, mask = eratosphen(2 ** bits)
    for k in xrange(2, 2 ** bits):
        if k % 2 == 0:
            continue
        n = k
        x.append([])
        for i in xrange(bits):
            x[-1].append([n % 2])
            n /= 2
        y.append([1.0 if mask[k] else 0.0])
    return np.asarray(x), np.asarray(y)


def split_learn_test(x, y):
    n = len(x)
    idx = range(n)
    random.shuffle(idx)
    k = n #int(n * 0.9)
    learn_x = [x[i] for i in idx[:k]]
    learn_y = [y[i] for i in idx[:k]]
    test_x = [x[i] for i in idx[k:]]
    test_y = [y[i] for i in idx[k:]]
    return np.asarray(learn_x), np.asarray(learn_y), np.asarray(test_x), np.asarray(test_y)


def get_sample(x, y, count):
    idx = range(len(x))
    random.shuffle(idx)
    idx = idx[:count]
    return [x[i] for i in idx], [y[i] for i in idx]


def calc_precicion_with_threshold(op, sess, x, test_x, test_y, minibatch, threshold):
    if len(test_x) == 0:
        return 0.0
    correct, total = 0.0, 1e-38
    for k in xrange(0, len(test_x), minibatch):
        tx, ty = test_x[k : k + minibatch], test_y[k : k + minibatch]
        res = sess.run(op, feed_dict = {x: tx})
        for i in xrange(len(tx)):
            if res[i] < threshold and ty[i] == 0:
                correct += 1
            elif res[i] > threshold and ty[i] == 1:
                correct += 1
            total += 1
    return correct / total


def calc_precicion(op, sess, x, test_x, test_y, minibatch, subsample):
    if len(test_x) == 0:
        return 0.0
    tx, ty = get_sample(test_x, test_y, subsample)
    a, b = -10.0, 10.0
    n = 10
    res = -1.0
    while b - a > 1e-3:
        best_t, best_val = -100, -1.0
        for i in xrange(n + 1):
            t = a + (b - a) * i / n
            val = calc_precicion_with_threshold(op, sess, x, tx, ty, minibatch, t)
            if val > best_val:
                best_t = t
                best_val = val
        margin = (b - a) / n
        a = best_t - margin
        b = best_t + margin
        if res < best_val:
            res = best_val
    return calc_precicion_with_threshold(op, sess, x, test_x, test_y, minibatch, a)


# do all stuff
def main():
    # define params
    max_time, state_size, hidden_size, eps, minibatch, subsample, print_freq = 20, 50, 300, 0.01, 50, 1000, 1
    #learning_rate = tf.Variable(0.0001, trainable = False)
    learning_rate, noise = 0.0001, 0.1
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    #w0 = tf.Variable(tf.random_normal([max_time, max_time * state_size], 0.0, 0.1))
    #b0 = tf.Variable(tf.random_normal([max_time * state_size], 0.0, 0.1))
    w1 = tf.Variable(tf.random_uniform([max_time * state_size, hidden_size], -0.1, 0.1))
    #w1 = tf.Variable(tf.random_normal([state_size, hidden_size1], -0.01, 0.01))
    b1 = tf.Variable(tf.random_uniform([hidden_size], -0.1, 0.1))
    w2 = tf.Variable(tf.random_uniform([hidden_size, 1], -0.1, 0.1))
    b2 = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
    # create learning graph
    x = tf.placeholder(tf.float32, [None, max_time, 1])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, x, dtype = tf.float32)
    #output = tf.reshape(x, [-1, max_time])
    #output = tf.sigmoid(tf.add(tf.matmul(output, w0), b0))
    y = tf.placeholder(tf.float32, [None, 1])
    #output = tf.sigmoid(output);
    output = tf.reshape(output, [-1, max_time * state_size])
    #output = output[:, max_time - 1, :]
    output = tf.sigmoid(tf.add(tf.matmul(output, w1), b1))
    output = tf.add(tf.matmul(output, w2), b2)
    # define loss and optimizer
    #loss = tf.nn.l2_loss(tf.subtract(output, y)) / ((2 ** max_time) * max_time)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
    #optimizer = tf.contrib.layers.optimize_loss(loss, global_step = None, learning_rate = learning_rate, optimizer = "Adam", gradient_noise_scale = noise)
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    data_x, data_y = generate_data(max_time)
    data_x, data_y, test_x, test_y = split_learn_test(data_x, data_y)
    prev_loss = None
    print "Learn baseline: %.4f" % (calc_precicion_with_threshold(output, sess, x, data_x, data_y, minibatch, 10.0))
    print "Test baseline: %.4f" % (calc_precicion_with_threshold(output, sess, x, test_x, test_y, minibatch, 10.0))
    print
    sys.stdout.flush()
    epoch = 0
    while True:
        l, c = 0.0, 0
        for i in xrange(0, len(data_x), minibatch):
            res, _l = sess.run([optimizer, loss], feed_dict = {x: data_x[i : i + minibatch], y: data_y[i : i + minibatch]})
            l += _l
            c += 1
            #print _l
        l /= c
        #print "%.6f" % l
        if False and prev_loss is not None and l > prev_loss:
            learning_rate *= 0.5
            print "Learning rate changed: %.4f %.4f %.6f" % (prev_loss, l, sess.run(learning_rate))
        prev_loss = l
        epoch += 1
        if epoch % print_freq == 0:
            print "Learn loss: %.4f, learn precision: %.4f" % (l, calc_precicion(output, sess, x, data_x, data_y, minibatch, subsample))
            tl, tc = 0.0, 1e-38
            for i in xrange(0, len(test_x), minibatch):
                _l = sess.run(loss, feed_dict = {x: test_x[i : i + minibatch], y: test_y[i : i + minibatch]})
                tl += _l
                tc += 1
            print "Test loss: %.4f, test precision: %.4f" % (tl / tc, calc_precicion(output, sess, x, test_x, test_y, minibatch, subsample))
            print
            sys.stdout.flush()
        if l <= eps:
            break


# entry point
if __name__ == "__main__":
    main()

