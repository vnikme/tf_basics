#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import json, math, numpy, random, shutil, sys


# norm of vector
def get_vector_norm(a):
    a = tf.transpose(a)
    a = tf.mul(a, a)
    a = tf.reduce_sum(a, 0)
    a = tf.add(a, 1e-16)
    a = tf.sqrt(a)
    a = tf.transpose(a)
    return a


def create_cos_dist4(a, b, c, d):
    b = tf.sigmoid(tf.subtract(b, a))
    d = tf.sigmoid(tf.subtract(d, c))
    d = tf.subtract(d, b)
    d = tf.mul(d, d)
    d = -tf.sqrt(d)
    return tf.reduce_sum(d, 1)


# print nearest words
def print_analogy(a, b, c, inputs, dist_func, embed_tensor, id2word, sess, count):
    dist = dist_func(a, b, c, embed_tensor)
    dist, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print "   ".join(["%s (%.3f)" % (id2word[idx[i]], dist[i]) for i in xrange(len(idx))])


# class for matching word<->id and storing matrixes
class TWord2Vec:
    def __init__(self):
        self.Word2Id = {}
        self.Id2Word = []
        self.WordFreqs = []

    def Load(self, path):
        try:
            s = open(path, "rt").read()
        except:
            return False
        if len(s) == 0:
            return False
        data = json.loads(s)
        self.Word2Id = data["Word2Id"]
        self.Id2Word = data["Id2Word"]
        self.EmbeddingWeights = tf.Variable(data["EmbeddingWeights"])
        return True


def print_analogies(sess, embed_tensor, inputs, w2v, base_words, words, count_of_nearest):
    pred = sess.run([embed_tensor], feed_dict = {inputs: [w2v.Word2Id[t.decode("utf-8")] for t in (base_words + words)]})[0]
    a, b, c = [[pred[i][j] for j in xrange(len(pred[i]))] for i in xrange(len(pred))]
    for v in [a, b, c]:
        print "\t".join(map(lambda x: "%.2f" % x, v))
    print_analogy(a, b, c, inputs, create_cos_dist4, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
    print


# do all stuff
def main():
    # define params
    dump_path = sys.argv[1]
    count_of_nearest = int(sys.argv[2])
    base_words = sys.argv[3:5]
    # load data
    w2v = TWord2Vec()
    if not w2v.Load(dump_path):
        print "Failed to load from '%s'" % dump_path
        return
    # input and output placeholders
    inputs = tf.placeholder(tf.int32, shape = [None])
    # tensor for 'input->embedding' transform
    embed_tensor = tf.nn.embedding_lookup(w2v.EmbeddingWeights, inputs)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    while True:
        pair = []
        pair.append(raw_input("Enter word: "))
        print_analogies(sess, embed_tensor, inputs, w2v, base_words, pair, count_of_nearest)


# entry point
if __name__ == "__main__":
    main()

