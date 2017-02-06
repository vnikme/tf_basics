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
    x, y = [], []
    for i in xrange(count):
        t = i * step
        x.append(generate_features(t))
        y.append(math.sin(t))
    return x, y


def main():
    # generate data
    learn_x, learn_y = generate_pool(100000, 0.00001 * math.pi * 2)
    
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
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

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
        epoch += 1
        if cost < 0.001:
            break
    print sess.run(a)


if __name__ == "__main__":
    main()

