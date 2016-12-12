#!/usr/bin/python

import tensorflow as tf
import random


def bits2num(bits):
    res = 0
    for i in xrange(len(bits) - 1, -1, -1):
        res *= 2
        res += bits[i]
    return res


def num2bits(n, k):
    res = []
    while n != 0:
        res.append(n % 2)
        n /= 2
    while len(res) < k:
        res.append(0)
    return res


# initialize layer matrixes with random numbers
def create_layers(sizes):
    weights, biases = [], []
    for i in xrange(1, len(sizes)):
        weights.append(tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]])))
        biases.append(tf.Variable(tf.random_normal([sizes[i]])))
    return weights, biases


# create perceptron tensor
# return perceptron and placeholder for input data
def create_perceptron(weights, biases):
    x = tf.placeholder(tf.float32, [None, weights[0].get_shape()[0]])
    result = x
    for i in xrange(len(weights)):
        result = tf.add(tf.matmul(result, weights[i]), biases[i])
        if i != len(weights) - 1:
            result = tf.nn.sigmoid(result)
    return result, x


# create objective functional
# return objective and placeholder for target (correct answers)
def create_cost(model, out_size):
    y = tf.placeholder(tf.float32, [None, out_size])
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(model, y))
    return cost, y


# generate `count` samples
# resurn samples and correct answers
def generate_batch(count, n):
    x, y = [], []
    limit = 2 ** n
    for sample in xrange(count):
        a = int(random.random() * limit)
        b = int(random.random() * limit)
        c = a + b
        x.append(num2bits(a, n) + num2bits(b, n))
        y.append(num2bits(c, n + 1))
    return x, y


# print predictions for batch input parameters
def print_predictions(sess, model, x, batch_x, batch_y):
    n = len(batch_x[0]) / 2
    pred = sess.run(model, feed_dict = {x: batch_x})
    errors = 0
    for i in xrange(len(pred)):
        a, b = bits2num(batch_x[i][:n]), bits2num(batch_x[i][n:])
        c = map(lambda t: 1 if t > 0.5 else 0, pred[i])
        c = bits2num(c)
        if a + b != c:
            print "%d\t%d\t%d\t%d" % (a, b, a + b, c)
            errors += 1
    print errors


# read file, split words, return
def read_data(path):
    return open(path).read().split()


# do all stuff
def main():
    # nn topology, first is input, last in output
    n = 10
    sizes = [2 * n, n, n, n, n + 1]
    # step size
    learning_rate = 0.001
    # number of epochs
    eps = 1e-5
    # number of samples in each epoch (because we have the same data all the time we can set it to 1)
    batch_size = 1000
    # create matrixes
    weights, biases = create_layers(sizes)
    # create model based on matrixes
    model, x = create_perceptron(weights, biases)
    # create objective
    cost, y = create_cost(model, sizes[-1])
    # create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # main work
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # run initialization (needed for tensorflow)
        sess.run(init)
        # just check we have correct learning data
        print generate_batch(batch_size, n)
        # check what we see on random data
        print sess.run(model, feed_dict = {x: generate_batch(batch_size, n)[0]})
        # iterate while error > eps
        epoch = 0
        while True:
            # generate next batch
            batch_x, batch_y = generate_batch(batch_size, n)
            # run optimization
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            # debug print
            if c < eps or epoch % 1000 == 0:
                # print predictions
                batch_x, batch_y = generate_batch(batch_size, n)
                print_predictions(sess, model, x, batch_x, batch_y)
                # loss
                print c, epoch
                print
            if c < eps:
                break
            epoch += 1
        for i in xrange(len(sizes) - 1):
            print sess.run(weights[i])
            print sess.run(biases[i])


# entry point
if __name__ == "__main__":
    main()

