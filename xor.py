#!/usr/bin/python

import tensorflow as tf


def create_layers(sizes):
    weights, biases = [], []
    for i in xrange(1, len(sizes)):
        weights.append(tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]])))
        biases.append(tf.Variable(tf.random_normal([sizes[i]])))
    return weights, biases

def create_perceptron(weights, biases):
    x = tf.placeholder(tf.float32, [None, weights[0].get_shape()[0]])
    result = x
    for i in xrange(len(weights)):
        result = tf.add(tf.matmul(result, weights[i]), biases[i])
        result = tf.nn.sigmoid(result)
    return result, x

def create_mse_cost(model, out_size):
    y = tf.placeholder(tf.float32, [None, out_size])
    return tf.reduce_sum(tf.pow(tf.subtract(model, y), 2)), y
    #return tf.nn.l2_loss(tf.subtract(model, y)), y

def generate_batch(count):
    x, y = [], []
    for sample in xrange(count):
        for i in xrange(2):
            for j in xrange(2):
                x.append([i, j, 1])
                y.append([i ^ j, i & j, i | j, 1, 0])
    return x, y

def print_predictions(sess, model, x):
    pred = sess.run(model, feed_dict = {x: generate_batch(1)[0]})
    for line in pred:
        print "\t".join(map(lambda t: "%.2f" % t, line))

def main():
    sizes = [3, 7, 5]
    learning_rate = 0.01
    epochs = 1000
    batch_size = 1000
    weights, biases = create_layers(sizes)
    model, x = create_perceptron(weights, biases)
    cost, y = create_mse_cost(model, sizes[-1])
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print generate_batch(1)
        print sess.run(model, feed_dict = {x: generate_batch(1)[0]})
        for epoch in xrange(epochs):
            batch_x, batch_y = generate_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            if epoch % 100 == 0:
                print c
                print_predictions(sess, model, x)
                #for i in xrange(len(sizes) - 1):
                #    print sess.run(weights[i])
                #    print sess.run(biases[i])
                print
        print_predictions(sess, model, x)

if __name__ == "__main__":
    main()

