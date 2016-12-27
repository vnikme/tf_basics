#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import json, math, numpy, random, shutil, sys


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


# print diffs between odd and even words
def print_diffs(sess, embed_tensor, inputs, w2v, words):
    pred = sess.run([embed_tensor], feed_dict = {inputs: [w2v.Word2Id[t.decode("utf-8")] for t in words]})[0]
    for i in xrange(0, len(words), 2):
        print words[i] + "\t" + words[i + 1] + ":\n\t\t" + "\t".join(["%.2f" % (pred[i][j] - pred[i + 1][j]) for j in xrange(len(pred[i]))])


# do all stuff
def main():
    with tf.device('/cpu:0'):
        # define params
        dump_path = sys.argv[1]
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
            words = raw_input("Enter pairs: ")
            print_diffs(sess, embed_tensor, inputs, w2v, words.split(" "))


# entry point
if __name__ == "__main__":
    main()

