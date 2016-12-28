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

# l2 distance between vectors
def create_l2_dist(a, b, c, d):
    b = tf.subtract(b, a)
    d = tf.subtract(d, c)
    dist = tf.subtract(b, d)
    dist = -tf.sqrt(tf.reduce_sum(tf.mul(dist, dist), 1))
    return dist
 
 
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


def predict_embeddings(sess, embed_tensor, inputs, w2v, words):
    pred = sess.run([embed_tensor], feed_dict = {inputs: words})[0]
    return [[pred[i][j] for j in xrange(len(pred[i]))] for i in xrange(len(pred))]


def print_analogies(sess, embed_tensor, inputs, w2v, a, b, c, count_of_nearest):
    for v in [a, b, c]:
        print "\t".join(map(lambda x: "%.2f" % x, v))
    print_analogy(a, b, c, inputs, create_cos_dist4, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
    print_analogy(a, b, c, inputs, create_l2_dist, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
    print


def get_analogy(sess, embed_tensor, inputs, dist_func, count_of_nearest, count_of_back_nearest, w2v, a, b, c, c_idx):
    dist = dist_func(a, b, c, embed_tensor)
    dist, idx = sess.run(tf.nn.top_k(dist, count_of_nearest), feed_dict = {inputs: range(len(w2v.Id2Word))})
    print "   ".join(["%s (%.3f)" % (w2v.Id2Word[idx[i]], dist[i]) for i in xrange(len(idx))])
    back_c, back_dist = [], []
    for k in xrange(len(dist)):
        back_c.append(predict_embeddings(sess, embed_tensor, inputs, w2v, [idx[k]])[0])
        back_dist.append(dist_func(b, a, back_c[k], embed_tensor))
    res = sess.run([tf.nn.top_k(dst, count_of_back_nearest) for dst in back_dist], feed_dict = {inputs: range(len(w2v.Id2Word))})
    for k in xrange(len(dist)):
        if c_idx not in res[k][1]:
            continue
        print w2v.Id2Word[idx[k]], dist[k]
    print


# do all stuff
def main():
    with tf.device('/cpu:0'):
        # define params
        dump_path = sys.argv[1]
        count_of_nearest, count_of_back_nearest = map(int, sys.argv[2:4])
        base_words = sys.argv[4:6]
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
        a, b = predict_embeddings(sess, embed_tensor, inputs, w2v, [w2v.Word2Id[t.decode("utf-8")] for t in base_words])
        while True:
            word = raw_input("Enter word: ")
            c_idx = w2v.Word2Id[word.decode("utf-8")]
            c = predict_embeddings(sess, embed_tensor, inputs, w2v, [c_idx])[0]
            #print_analogies(sess, embed_tensor, inputs, w2v, a, b, c, count_of_nearest)
            get_analogy(sess, embed_tensor, inputs, create_l2_dist, count_of_nearest, count_of_back_nearest, w2v, a, b, c, c_idx)
            #get_analogy(sess, embed_tensor, inputs, create_cos_dist4, count_of_nearest, count_of_back_nearest, w2v, a, b, c, c_idx)


# entry point
if __name__ == "__main__":
    main()

