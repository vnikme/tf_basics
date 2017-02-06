#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random


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
    x, y = np.ndarray([2 ** bits - 2, bits, 1], np.float32), np.ndarray([2 ** bits - 2, 1], np.float32)
    _, mask = eratosphen(2 ** bits)
    print mask
    exit(1)
    for k in xrange(2, 2 ** bits):
        n = k
        for i in xrange(bits):
            x[k - 2][i][0] = n % 2
            n /= 2
        y[k - 2][0] = 1.0 if mask[k - 2] else 0.0
    return x, y


def split_learn_test(x, y):
    n = len(x)
    idx = range(n)
    random.shuffle(idx)
    k = int(n * 0.9)
    learn_x = [x[i] for i in idx[:k]]
    learn_y = [y[i] for i in idx[:k]]
    test_x = [x[i] for i in idx[k:]]
    test_y = [y[i] for i in idx[k:]]
    return np.asarray(learn_x), np.asarray(learn_y), np.asarray(test_x), np.asarray(test_y)


# do all stuff
def main():
    # define params
    max_time, state_size, eps = 8, 16, 0.001
    learning_rate = tf.Variable(0.001, trainable = False)
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, 1], -0.01, 0.01))
    b = tf.Variable(tf.random_normal([1], -0.01, 0.01))
    # create learning graph
    x = tf.placeholder(tf.float32, [None, max_time, 1])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, x, dtype = tf.float32)
    y = tf.placeholder(tf.float32, [None, 1])
    output = output[:, max_time - 1, :]
    #output = tf.sigmoid(tf.add(tf.matmul(output, w), b))
    output = tf.add(tf.matmul(output, w), b)
    # define loss and optimizer
    #loss = tf.nn.l2_loss(tf.subtract(output, y)) / ((2 ** max_time) * max_time)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    data_x, data_y = generate_data(max_time)
    data_x, data_y, test_x, test_y = split_learn_test(data_x, data_y)
    prev_loss = None
    while True:
        res, l = sess.run([optimizer, loss], feed_dict = {x: data_x, y: data_y})
        print l
        if False and prev_loss is not None and l > prev_loss:
            learning_rate *= 0.5
            print "Learning rate changed: %.4f %.4f %.6f" % (prev_loss, l, sess.run(learning_rate))
        prev_loss = l
        cnt += 1
        if cnt % 10 == 0:
            print "Test loss: %f" % sess.run(loss, feed_dict = {x: test_x, y: test_y})
        if l <= eps:
            break


# entry point
if __name__ == "__main__":
    main()

