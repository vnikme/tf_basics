#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random, scipy.io.wavfile, sys, json


# initialize layer matrixes with random numbers
def create_layers(sizes):
    weights, biases = [], []
    for i in xrange(1, len(sizes)):
        weights.append(tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]])))
        biases.append(tf.Variable(tf.random_normal([sizes[i]])))
    return weights, biases


def save_layers(sess, weights, biases, path):
    data = {}
    data["weights"] = []
    data["biases"] = []
    for i in xrange(len(weights)):
        data["weights"].append(sess.run(weights[i]).tolist())
        data["biases"].append(sess.run(biases[i]).tolist())
    open(path, "wt").write(json.dumps(data))


def load_layers(path, weights, biases, layers_limit):
    data = json.loads(open(path, "rt").read())
    for i in xrange(layers_limit):
        w, b = data["weights"][i], data["biases"][i]
        if i < len(weights):
            weights[i] = tf.Variable(w)
            biases[i] = tf.Variable(b)
        else:
            weights.append(tf.Variable(w))
            biases.append(tf.Variable(b))
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
# return objectives for points and batches and placeholder for target (correct answers)
def create_mse_cost(model, out_size):
    y = tf.placeholder(tf.float32, [None, out_size])
    batch_cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model, y), 2), 1)))
    point_cost = tf.reduce_mean(tf.pow(tf.subtract(model, y), 2))
    return point_cost, batch_cost, y


def generate_sample(data, n_frames, win_size, batch_size):
    x, y = [], []
    for i in xrange(batch_size):
        r = random.randint(0, n_frames - win_size * 3 - 1)
        x.append(data[0][r : r + win_size * 3])
        y.append(data[0][r + win_size : r + win_size * 2])
    return np.stack(x), np.stack(y)


# print predictions for all possible input parameters
def print_predictions(sess, model, x, data, n_frames, win_size):
    data_x, data_y = generate_sample(data, n_frames, win_size, 1)
    pred = sess.run(model, feed_dict = {x: data_x})
    print "\t".join(map(lambda i: "%.3f-%.3f=%.3f" % (pred[0][i], data_y[0][i], pred[0][i] - data_y[0][i]), xrange(win_size)))


# do all stuff
def main():
    data = scipy.io.wavfile.read('data.wav')
    n_frames, n_channels = data[1].shape
    data = np.transpose(data[1]).astype('f4') / 32768
    win_size = 512
    # nn topology, first is input, last is output
    #layer_sizes = [win_size * 3, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, win_size]
    layer_sizes = []
    for i in xrange(2):
        layer_sizes.append(win_size * 3)
    for i in xrange(2):
        layer_sizes.append(win_size * 2)
    for i in xrange(2):
        layer_sizes.append(win_size)
    layer_sizes += [256, 128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256, 512, 512]
    # step size
    learning_rate = 0.001
    # threshold to stop
    eps = 1e-3
    # number of samples in each epoch (because we have the same data all the time we can set it to 1)
    batch_size = 1000
    # how many layers to leave
    layers_limit = 20
    layers_limit_to_load = 20
    with tf.device('/gpu:0'):
        # create matrixes
        layer_sizes = layer_sizes[:layers_limit] + [win_size]
        weights, biases = create_layers(layer_sizes)
        weights, biases = load_layers("dump.txt", weights, biases, layers_limit_to_load)
        # create model based on matrixes
        model, x = create_perceptron(weights, biases)
        # create objective
        point_cost, batch_cost, y = create_mse_cost(model, layer_sizes[-1])
        # create optimizer
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(point_cost)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(point_cost)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(point_cost)
    # main work
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # run initialization (needed for tensorflow)
        sess.run(init)
        # iterate while error > eps
        epoch = 1
        losses = []
        while True:
            data_x, data_y = generate_sample(data, n_frames, win_size, batch_size)
            # run optimization
            _, c = sess.run([optimizer, batch_cost], feed_dict = {x: data_x, y: data_y})
            losses.append(c)
            # debug print
            if len(losses) % 1000 == 0:
               # loss
                c = sum(losses) / len(losses)
                if epoch % 10 == 0 or c < eps:
                    save_layers(sess, weights, biases, "dump.txt") 
                print "loss=%.5f, epoch=%d" % (c, epoch)
                # print predictions
                print_predictions(sess, model, x, data, n_frames, win_size)
                print
                sys.stdout.flush()
                if c < eps:
                    break
                losses = []
                epoch += 1


# entry point
if __name__ == "__main__":
    main()

