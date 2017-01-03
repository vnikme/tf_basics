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
    max_time, input_size, output_size, state_size = 16, 2, 1, 5
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    x = tf.placeholder(tf.float32, [None, max_time, input_size])
    output, state = tf.nn.dynamic_rnn(gru, x, dtype = tf.float32)
    y = tf.placeholder(tf.float32, [None, max_time, output_size])
    w = tf.Variable(tf.random_normal([state_size, output_size]))
    b = tf.Variable(tf.random_normal([output_size]))
    output = tf.reshape(output, [-1, state_size])
    output = tf.sigmoid(tf.add(tf.matmul(output, w), b))
    output = tf.reshape(output, [-1, max_time, output_size])
    loss = tf.nn.l2_loss(tf.subtract(output, y))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    while True:
        batch_x, batch_y = generate_batch(10000, max_time)
        res, l = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
        print l
    return


# entry point
if __name__ == "__main__":
    main()

