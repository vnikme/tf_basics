#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import math, random, sys


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя".decode("utf-8")
def is_letter(ch):
    return ch in all_syms


# read file, split words, return
def read_data(path):
    data = []
    word = ""
    for ch in to_wide_lower(open(path).read() + " "):
        if not is_letter(ch):
            if len(word) != 0:
                data.append(word.encode("utf-8"))
            word = ""
        else:
            word += ch
    word2id = {}
    id2word = []
    for w in data:
        if w in word2id:
            continue
        word2id[w] = len(id2word)
        id2word.append(w)
    return data, word2id, id2word


#make next batch
def generate_batch(data, word2id, batch_size, context_width):
    inputs, labels = [], []
    n = len(data)
    for i in xrange(batch_size / context_width / 2):
        k = int(random.random() * n)
        for j in xrange(-context_width, context_width + 1):
            if j == 0:
                continue
            inputs.append(word2id[data[k]])
            labels.append([word2id[data[(k + j + n) % n]]])
    return inputs, labels


# do all stuff
def main():
    # define params
    embedding_size, batch_size, num_sampled, context_width, epoch = map(int, sys.argv[2:7])
    # read data, make indexes word <-> id
    data, word2id, id2word = read_data(sys.argv[1])
    vocabulary_size = len(word2id)
    # input and output placeholders
    inputs = tf.placeholder(tf.int32, shape = [None])
    labels = tf.placeholder(tf.int32, shape = [None, 1])
    # matrix and tensor for 'input->embedding' transform
    embed_weights = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed_tensor = tf.nn.embedding_lookup(embed_weights, inputs)
    # matrix and bias for 'embedding->target' transform
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=labels,
                               inputs=embed_tensor,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size
                              )
                         )
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for e in xrange(epoch):
        batch_inputs, batch_labels = generate_batch(data, word2id, batch_size, context_width)
        _, loss_val = sess.run([optimizer, loss], feed_dict = {inputs: batch_inputs, labels: batch_labels})
        if e % 1000 == 0 or e + 1 == epoch:
            print loss_val
            pred = sess.run([embed_tensor], feed_dict = {inputs: [word2id["говорил"], word2id["говорила"], word2id["мужчина"], word2id["женщина"]]})
            a = [float(t) for t in pred[0][0] - pred[0][1]]
            b = [float(t) for t in pred[0][2] - pred[0][3]]
            a_abs = math.sqrt(sum([t * t for t in a]))
            b_abs = math.sqrt(sum([t * t for t in b]))
            a = [t / a_abs for t in a]
            b = [t / b_abs for t in b]
            print sum([i * j for i, j in zip(a, b)])


# entry point
if __name__ == "__main__":
    main()

