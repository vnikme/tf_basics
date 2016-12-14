#!/usr/bin/python

import tensorflow as tf


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
        result = tf.nn.sigmoid(result)
    return result, x


# create objective functional
# return objectives for points and batches and placeholder for target (correct answers)
def create_mse_cost(model, out_size):
    y = tf.placeholder(tf.float32, [None, out_size])
    batch_cost = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model, y), 2), 1)))
    point_cost = tf.reduce_sum(tf.pow(tf.subtract(model, y), 2))
    return point_cost, batch_cost, y


# generate `count` samples
# resurn samples and correct answers
def generate_batch(count):
    x, y = [], []
    for sample in xrange(count):
        for i in xrange(2):
            for j in xrange(2):
                x.append([i, j, 1])
                y.append([i ^ j, i & j, i | j, 1, 0])
    return x, y


# print predictions for all possible input parameters
def print_predictions(sess, model, x):
    pred = sess.run(model, feed_dict = {x: generate_batch(1)[0]})
    for line in pred:
        print "\t".join(map(lambda t: "%.2f" % t, line))


# do all stuff
def main():
    # nn topology, first is input, last is output
    sizes = [3, 5, 5]
    # step size
    learning_rate = 0.01
    # threshold to stop
    eps = 1e-3
    # number of samples in each epoch (because we have the same data all the time we can set it to 1)
    batch_size = 1
    # create matrixes
    weights, biases = create_layers(sizes)
    # create model based on matrixes
    model, x = create_perceptron(weights, biases)
    # create objective
    point_cost, batch_cost, y = create_mse_cost(model, sizes[-1])
    # create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(point_cost)
    # main work
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # run initialization (needed for tensorflow)
        sess.run(init)
        # just check we have correct learning data
        print generate_batch(1)
        # check what we see on random data
        print sess.run(model, feed_dict = {x: generate_batch(1)[0]})
        # iterate while error > eps
        epoch = 0
        while True:
            # generate next batch
            batch_x, batch_y = generate_batch(batch_size)
            # run optimization
            _, c = sess.run([optimizer, batch_cost], feed_dict = {x: batch_x, y: batch_y})
            # debug print
            if c < eps or epoch % 1000 == 0:
                # loss
                print c, epoch
                # print predictions
                print_predictions(sess, model, x)
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

