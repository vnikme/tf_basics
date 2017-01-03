#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random


def generate_batch(batch_size, bits):
    x, y = np.ndarray([batch_size, bits, 2], np.float32), np.ndarray([batch_size, bits, 1], np.float32)
    for k in xrange(batch_size):
        c = 0
        for i in xrange(bits - 1):
            a = 1 if random.random() > 0.5 else 0
            b = 1 if random.random() > 0.5 else 0
            x[k][i][0] = a
            x[k][i][1] = b
            y[k][i][0] = (a + b + c) % 2
            c = (a + b + c) / 2
        x[k][bits - 1][0] = 0
        x[k][bits - 1][1] = 0
        y[k][bits - 1][0] = c
    return x, y


# do all stuff
def main():
    # define params
    max_time, max_valid_time, batch_size, valid_batch_size, input_size, output_size, state_size, eps = 100, 100000, 1000, 1000, 2, 1, 5, 0.001
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, output_size]))
    b = tf.Variable(tf.random_normal([output_size]))
    # create learning graph
    with tf.variable_scope('train'):
        x = tf.placeholder(tf.float32, [None, max_time, input_size])
        output, state = tf.nn.dynamic_rnn(gru, x, dtype = tf.float32)
        y = tf.placeholder(tf.float32, [None, max_time, output_size])
        output = tf.reshape(output, [-1, state_size])
        output = tf.sigmoid(tf.add(tf.matmul(output, w), b))
        output = tf.reshape(output, [-1, max_time, output_size])
        # define loss and optimizer
        loss = tf.nn.l2_loss(tf.subtract(output, y))
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    # define validation and test data and operations
    with tf.variable_scope('train', reuse = True):
        valid_x = tf.placeholder(tf.float32, [None, max_valid_time, input_size])
        output, state = tf.nn.dynamic_rnn(gru, valid_x, dtype = tf.float32)
        valid_y = tf.placeholder(tf.float32, [None, max_valid_time, output_size])
        output = tf.reshape(output, [-1, state_size])
        output = tf.sigmoid(tf.add(tf.matmul(output, w), b))
        output = tf.reshape(output, [-1, max_valid_time, output_size])
        valid_loss = tf.nn.l2_loss(tf.subtract(output, valid_y))
    # begin training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    valid_batch_x, valid_batch_y = generate_batch(valid_batch_size, max_valid_time)
    while True:
        batch_x, batch_y = generate_batch(batch_size, max_time)
        res, l = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
        l /= (batch_size * max_time)
        if cnt % 10 == 0:
            vl = sess.run(valid_loss, feed_dict = {valid_x: valid_batch_x, valid_y: valid_batch_y})
            vl /= (valid_batch_size * max_valid_time)
            print l, vl
        else:
            print l
        if vl < eps:
            break
        cnt += 1
    test_x, test_y = generate_batch(valid_batch_size, max_valid_time)
    tl = sess.run(valid_loss, feed_dict = {valid_x: test_x, valid_y: test_y})
    tl /= (valid_batch_size * max_valid_time)
    print tl


# entry point
if __name__ == "__main__":
    main()

