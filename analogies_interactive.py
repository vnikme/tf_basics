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


class TContext:
    def __init__(self):
        self.w2v = None
        self.pairs = []
        self.embs = []
        self.w = None
        self.b = None


def LoadW2V(ctx, path):
    ctx.w2v = TWord2Vec()
    ctx.w2v.Load(path)
    inputs = tf.placeholder(tf.int32, shape = [None])
    embed_tensor = tf.nn.embedding_lookup(ctx.w2v.EmbeddingWeights, inputs)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    ctx.embs = sess.run([embed_tensor], feed_dict = {inputs: range(len(ctx.w2v.Id2Word))})[0]
    print "TWord2Vec loaded."


def AddPair(ctx, w1, w2):
    ctx.pairs.append([w1.decode("utf-8"), w2.decode("utf-8")])


def DelPair(ctx, w1, w2):
    for i in xrange(len(ctx.pairs)):
        if ctx.pairs[0] == w1 and ctx.pairs[1] == w2:
            ctx.pairs = ctx.pairs[:i] + ctx.pairs[i + 1:]
            print "Deleted"
            break


def LoadPairs(ctx, path):
    ctx.pairs = []
    for line in open("data/" + path, "rt"):
        if line[-1] == "\n":
            line = line[:-1]
        w = line.split("\t")
        ctx.pairs.append([w[0].decode("utf-8"), w[1].decode("utf-8")])
    print "Loaded"


def SavePairs(ctx, path):
    fout = open("data/" + path, "wt")
    for pair in ctx.pairs:
        fout.write(pair[0].encode("utf-8") + "\t" + pair[1].encode("utf-8") + "\n")


def LearnModel(ctx, iterations, alpha):
    n = len(ctx.embs[0])
    w = tf.Variable(tf.random_normal([n, n]))
    b = tf.Variable(tf.random_normal([n]))
    x = tf.placeholder(tf.float32, [None, n])
    op = tf.add(tf.matmul(x, w), b)
    y = tf.placeholder(tf.float32, [None, n])
    loss = tf.reduce_mean(tf.pow(tf.subtract(op, y), 2)) + alpha * (tf.reduce_mean(tf.abs(w)) + tf.reduce_mean(tf.abs(b)))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
    feed_x = [ctx.embs[ctx.w2v.Word2Id[p[0]]] for p in ctx.pairs]
    feed_y = [ctx.embs[ctx.w2v.Word2Id[p[1]]] for p in ctx.pairs]
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in xrange(iterations):
        _, err = sess.run([optimizer, loss], feed_dict = {x: feed_x, y: feed_y})
        print err
    ctx.w, ctx.b = sess.run([w, b])
    for line in ctx.w:
        print " ".join(map(lambda x: "% 5.2f" % x, line))
    print
    print " ".join(map(lambda x: "% 5.2f" % x, ctx.b))
    print "Learnt"


def Dist2(x, y):
    res = 0.0
    for i in xrange(len(x)):
        res += ((x[i] - y[i]) * (x[i] - y[i]))
    return res


def Test(ctx, w, count):
    print w
    w = ctx.w2v.Word2Id[w.decode("utf-8")]
    x = tf.placeholder(tf.float32, [None, len(ctx.b)])
    op = tf.add(tf.matmul(x, ctx.w), ctx.b)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    w = sess.run(op, feed_dict = {x: [ctx.embs[w]]})[0]
    d = []
    for y in ctx.embs:
        d.append(Dist2(w, y))
    idx = range(len(ctx.embs))
    idx.sort(key = lambda i: d[i])
    print "\t".join(map(lambda i: "%s %.2f" % (ctx.w2v.Id2Word[i].encode("utf-8"), d[i]), idx[:count]))


# do all stuff
def main():
    with tf.device('/cpu:0'):
        ctx = TContext()
        LoadW2V(ctx, "dump.txt")
        LoadPairs(ctx, "pairs.txt")
        LearnModel(ctx, 10000, 0.01)
        for word in ["кот", "король", "котик", "дед", "шагал"]:
            Test(ctx, word, 5)
        while True:
            try:
                line = raw_input("Enter command: ")
                cols = line.split()
                cmd = cols[0]
                if cmd == "w2v":
                    LoadW2V(ctx, cols[1])
                elif cmd == "add":
                    AddPair(ctx, cols[1], cols[2])
                elif cmd == "del":
                    DelPair(ctx, cols[1], cols[2])
                elif cmd == "load":
                    LoadPairs(ctx, cols[1])
                elif cmd == "save":
                    SavePairs(ctx, cols[1])
                elif cmd == "learn":
                    LearnModel(ctx, int(cols[1]), float(cols[2]))
                elif cmd == "exit":
                    break
                else:
                    Test(ctx, cols[0], int(cols[1]) if len(cols) > 1 else 5)
            except:
                pass


# entry point
if __name__ == "__main__":
    main()

