#!/usr/bin/python
# coding: utf-8


import math, random
import tensorflow as tf


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

        
def generate_pool(count, step):
    learn_x, learn_y = [], []
    test_x, test_y = [], []
    for i in xrange(count):
        is_test = True if random.random() < 0.1 else False
        t = i * step
        if is_test:
            test_x.append(generate_features(t))
            test_y.append(math.sin(t))
        else:
            learn_x.append(generate_features(t))
            learn_y.append(math.sin(t))
    return learn_x, learn_y, test_x, test_y


def main():
    # generate data
    learn_x, learn_y, test_x, test_y = generate_pool(100000, 0.00001 * math.pi * 2)
    
    # create variables and placeholders
    a = tf.Variable(tf.random_normal([FEATURES_COUNT], -0.01, 0.01))
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
        cost, _ = sess.run([loss, optimizer], feed_dict = {x: learn_x, y: learn_y})
        print "Epoch: %d, loss: %f" % (epoch, cost)
        if epoch % 100 == 0:
            print sess.run(a)
            print "Error on test:", sess.run(loss, feed_dict = {x: test_x, y: test_y})
        epoch += 1
        if cost < 0.001:
            break
    print sess.run(a)
    print "Error on test:", sess.run(loss, feed_dict = {x: test_x, y: test_y})


if __name__ == "__main__":
    main()

