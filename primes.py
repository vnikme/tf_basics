#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random


def eratosphen(n):
    res, mask = [], [True] * n
    for i in xrange(2, n):
        if not mask[i]:
            continue
        res.append(i)
        for j in xrange(2 * i, n, i):
            mask[j] = False
    return res, mask


def generate_data(bits):
    x, y = np.ndarray([2 ** bits, bits, 1], np.float32), np.ndarray([2 ** bits, 1], np.float32)
    _, mask = eratosphen(2 ** bits)
    for k in xrange(2 ** bits):
        n = k
        for i in xrange(bits):
            x[k][i][0] = n % 2
            n /= 2
        y[k][0] = 1.0 if mask[k] else 0.0
    return x, y


# do all stuff
def main():
    # define params
    max_time, state_size, eps, learning_rate = 16, 32, 0.01, 0.001
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, 1]))
    b = tf.Variable(tf.random_normal([1]))
    # create learning graph
    x = tf.placeholder(tf.float32, [None, max_time, 1])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, x, dtype = tf.float32)
    y = tf.placeholder(tf.float32, [None, 1])
    output = output[:, max_time - 1, :]
    output = tf.sigmoid(tf.add(tf.matmul(output, w), b))
    #output = tf.add(tf.matmul(output, w), b)
    # define loss and optimizer
    loss = tf.nn.l2_loss(tf.subtract(output, y))
    #loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(output, y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    data_x, data_y = generate_data(max_time)
    while True:
        res, l = sess.run([optimizer, loss], feed_dict = {x: data_x, y: data_y})
        print l
        cnt += 1
        if l <= eps:
            break


# entry point
if __name__ == "__main__":
    main()

