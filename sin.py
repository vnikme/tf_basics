#!/usr/bin/python
# coding: utf-8


import math, random, sys
import tensorflow as tf
import numpy as np


FEATURES_COUNT = 10


def generate_features(x):
    t = 1.0
    res = []
    i = 1
    while len(res) < FEATURES_COUNT:
        t *= x
        t /= i
        if i % 2 == 1:
            res.append(t)
        i += 1
    return res


def shuffle(x, y):
    n = len(x)
    idx = range(n)
    random.shuffle(idx)
    x = [x[i] for i in idx]
    y = [y[i] for i in idx]
    return x, y


def generate_pool(count, step):
    learn_x, learn_y = [], []
    test_x, test_y = [], []
    for i in xrange(-count, count + 1):
        is_test = True if random.random() < 0.1 else False
        t = i * step
        if is_test:
            test_x.append(generate_features(t))
            test_y.append(math.sin(t))
            #test_y.append(2 * t)
        else:
            learn_x.append(generate_features(t))
            learn_y.append(math.sin(t))
            #learn_y.append(2 * t)
    learn_x, learn_y = shuffle(learn_x, learn_y)
    test_x, test_y = shuffle(test_x, test_y)
    return np.asarray(learn_x), np.asarray(learn_y), np.asarray(test_x), np.asarray(test_y)


def main():
    # constants
    minibatch_size, print_freq = 1000, 1000

    # generate data
    learn_x, learn_y, test_x, test_y = generate_pool(10000, 0.0001 * math.pi * 4)
    
    # create variables and placeholders
    a = tf.Variable(tf.random_normal([FEATURES_COUNT], -0.001, 0.001))
    x = tf.placeholder(tf.float32, [None, FEATURES_COUNT])
    y = tf.placeholder(tf.float32, [None])
    
    # create loss function
    loss = tf.mul(x, a)            # [None, FEATURES_COUNT] * [FEATURES_COUNT] -> [None, FEATURES_COUNT]
    loss = tf.reduce_sum(loss, 1)  # [None, FEATURES_COUNT] -> [None]
    loss = tf.subtract(loss, y)
    loss = tf.mul(loss, loss)
    loss = tf.reduce_mean(loss, 0) # [None] -> []
    
    # create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.012017).minimize(loss)

    # run
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 0
    while True:
        cost, cnt = 0.0, 0
        for i in xrange(0, len(learn_x), minibatch_size):
            c, _ = sess.run([loss, optimizer], feed_dict = {x: learn_x[i : i + minibatch_size], y: learn_y[i : i + minibatch_size]})
            cost += c
            cnt += 1
        cost /= cnt
        #print "Epoch: %d, loss: %f" % (epoch, cost)
        if epoch % print_freq == 0:
            print ["%.4f" % f for f in sess.run(a)]
            print "Error on test:", sess.run(loss, feed_dict = {x: test_x, y: test_y})
            print
            sys.stdout.flush()
        epoch += 1
        if cost < 0.001:
            break
    print ["%.4f" % f for f in sess.run(a)]
    print "Error on test:", sess.run(loss, feed_dict = {x: test_x, y: test_y})


if __name__ == "__main__":
    main()

