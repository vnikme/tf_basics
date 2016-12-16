#!/usr/bin/python

import tensorflow as tf
import random, sys


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
def create_layers(n, m, sizes):
    weights, biases = [], []
    prev_size = n + m
    for size in sizes:
        weights.append(tf.Variable(tf.random_normal([prev_size, size])))
        biases.append(tf.Variable(tf.random_normal([size])))
        prev_size = size
    sum_output_weights = tf.Variable(tf.random_normal([prev_size, max(n, m) + 1]))
    sum_output_biases = tf.Variable(tf.random_normal([max(n, m) + 1]))
    mul_output_weights = tf.Variable(tf.random_normal([prev_size, n + m]))
    mul_output_biases = tf.Variable(tf.random_normal([n + m]))
    return weights, biases, sum_output_weights, sum_output_biases, mul_output_weights, mul_output_biases


# create perceptron tensor
# return perceptron and placeholders for input data
def create_perceptron(n, m, weights, biases, sum_output_weights, sum_output_biases, mul_output_weights, mul_output_biases):
    a = tf.placeholder(tf.float32, [None, n])
    b = tf.placeholder(tf.float32, [None, m])
    result = tf.concat(1, [a, b])
    for i in xrange(len(weights)):
        result = tf.add(tf.matmul(result, weights[i]), biases[i])
        result = tf.nn.sigmoid(result)
    result_sum = tf.add(tf.matmul(result, sum_output_weights), sum_output_biases)
    result_mul = tf.add(tf.matmul(result, mul_output_weights), mul_output_biases)
    return result_sum, result_mul, a, b


# weighted cross-entropy
def weighted_crossentropy_cost(model, target):
    shape = target.get_shape()
    shape_idx = range(len(shape))
    model = tf.transpose(model, [shape_idx[-1]] + shape_idx[:-1])
    target = tf.transpose(target, [shape_idx[-1]] + shape_idx[:-1])
    n = shape[-1]
    res = None
    mul = 1
    for i in xrange(n):
        t = tf.nn.sigmoid_cross_entropy_with_logits(model, target)
        t = tf.scalar_mul(mul, t)
        if res == None:
            res = t
        else:
            res = tf.add(res, t)
    return res


# create objective functional
# return objective and placeholders for targets (correct answers)
def create_cost(model_sum, model_mul, n, m):
    s = tf.placeholder(tf.float32, [None, max(n, m) + 1])
    p = tf.placeholder(tf.float32, [None, n + m])
    return tf.reduce_sum(weighted_crossentropy_cost(model_sum, s)) + tf.reduce_sum(weighted_crossentropy_cost(model_mul, p)), s, p


# generate `count` samples
# resurn samples and correct answers
def generate_batch(count, n, m):
    a, b, s, p = [], [], [], []
    limit_a, limit_b = 2 ** n, 2 ** m
    for sample in xrange(count):
        u = int(random.random() * limit_a)
        v = int(random.random() * limit_b)
        c = u + v
        d = u * v
        a.append(num2bits(u, n))
        b.append(num2bits(v, m))
        s.append(num2bits(c, max(n, m) + 1))
        p.append(num2bits(d, n + m))
    return a, b, s, p


# print predictions for batch input parameters
def print_predictions(sess, model_sum, model_mul, a, b, batch_a, batch_b, n, m):
    pred_sum, pred_mul = sess.run([model_sum, model_mul], feed_dict = {a: batch_a, b: batch_b})
    errors = 0
    for i in xrange(len(batch_a)):
        a, b = bits2num(batch_a[i]), bits2num(batch_b[i])
        c = map(lambda t: 1 if t >= 0.0 else 0, pred_sum[i])
        d = map(lambda t: 1 if t >= 0.0 else 0, pred_mul[i])
        c = bits2num(c)
        d = bits2num(d)
        if a + b != c or a * b != d:
            print "%d\t%d\t%d\t%d\t%d\t%d" % (a, b, a + b, c, a * b, d)
            errors += 1
    print errors


# read file, split words, return
def read_data(path):
    return open(path).read().split()


# do all stuff
def main():
    with tf.device("/cpu:0"):
        # sizes of input numbers
        n, m = int(sys.argv[1]), int(sys.argv[2])
        # nn topology, hidden layers
        sizes = [5 * (n + m), 5 * (n + m), 5 * (n + m)]
        # step size
        learning_rate = float(sys.argv[3])
        # threshold to stop
        eps = 1e-3
        # number of samples in each epoch (because we have the same data all the time we can set it to 1)
        batch_size, print_freq = int(sys.argv[4]), int(sys.argv[5])
        # create matrixes
        weights, biases, sum_output_weights, sum_output_biases, mul_output_weights, mul_output_biases = create_layers(n, m, sizes)
        # create models based on matrixes
        model_sum, model_mul, a, b = create_perceptron(n, m, weights, biases, sum_output_weights, sum_output_biases, mul_output_weights, mul_output_biases)
        # create objective
        cost, s, p = create_cost(model_sum, model_mul, n, m)
        # create optimizer
        #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(cost)
        # main work
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # run initialization (needed for tensorflow)
            sess.run(init)
            # iterate while error > eps
            epoch = 0
            while True:
                # generate next batch
                batch_a, batch_b, batch_s, batch_p = generate_batch(batch_size, n, m)
                # run optimization
                _, c = sess.run([optimizer, cost], feed_dict = {a: batch_a, b: batch_b, s: batch_s, p: batch_p})
                # debug print
                if c < eps or epoch % print_freq == 0:
                    # print predictions
                    batch_a, batch_b, _, __ = generate_batch(batch_size, n, m)
                    print_predictions(sess, model_sum, model_mul, a, b, batch_a, batch_b, n, m)
                    # loss
                    print c / batch_size, epoch * batch_size
                    print
                if c / batch_size < eps:
                    break
                epoch += 1
            for i in xrange(len(sizes) - 1):
                print sess.run(weights[i])
                print sess.run(biases[i])


# entry point
if __name__ == "__main__":
    main()

