#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import json, math, random, shutil, sys


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя".decode("utf-8")
def is_letter(ch):
    return ch in all_syms


# read file, split words, return
def read_data(path, min_freq):
    data = []
    word = ""
    for ch in to_wide_lower(open(path).read() + " "):
        if not is_letter(ch):
            if len(word) != 0:
                data.append(word.encode("utf-8"))
            word = ""
        else:
            word += ch
    freqs = {}
    for w in data:
        freqs[w] = freqs.get(w, 0) + 1
    word2id = {"<unk>": 0}
    id2word = ["<unk>"]
    for w in data:
        if freqs[w] < min_freq:
            w = "<unk>"
        if w in word2id:
            continue
        word2id[w] = len(id2word)
        id2word.append(w)
    return data, word2id, id2word


# print some simple statistics
def print_data_stats(data, words, w2v):
    d = {}
    for word in data:
        d[word] = d.get(word, 0) + 1
    idx = d.keys()
    idx.sort(key = lambda x: -d[x])
    lens = {}
    #for word in idx[:100]:
    #    print word
    print len(data), len(w2v.Id2Word)
    #for word in d.keys():
    #    l = d[word]
    #    lens[l] = lens.get(l, 0) + 1
    #for k in sorted(lens.keys()):
    #    print k, lens[k]
    #exit(1)


# make next batch
def generate_batch(data, word2id, context_width, take_prob):
    inputs, labels = [], []
    n = len(data)
    for k in xrange(context_width, len(data) - context_width):
        if random.random() > take_prob:
            continue
        word = word2id[data[k]] if data[k] in word2id else word2id["<unk>"]
        for j in xrange(-context_width, context_width + 1):
            if j == 0:
                continue
            context_word = word2id[data[(k + j + n) % n]] if data[(k + j + n) % n] in word2id else word2id["<unk>"]
            inputs.append(word)
            labels.append([context_word])
    return inputs, labels


# norm of vector
def get_vector_norm(a):
    a = tf.transpose(a)
    a = tf.mul(a, a)
    a = tf.reduce_sum(a, 0)
    a = tf.add(a, 1e-16)
    a = tf.sqrt(a)
    a = tf.transpose(a)
    return a


# operation to calculate distance from certain word
def create_cos_dist(a, b, c, inputs, embed_weights):
    d = tf.nn.embedding_lookup(embed_weights, inputs)
    b = tf.subtract(b, a)
    d = tf.subtract(d, c)
    b = tf.divide(b, tf.transpose([get_vector_norm(b)]))
    d = tf.divide(d, tf.transpose([get_vector_norm(d)]))
    return tf.reduce_sum(tf.mul(b, d), 1)


# print nearest words
def print_analogy(a, b, c, inputs, embed_weights, id2word, sess, count):
    dist = create_cos_dist(a, b, c, inputs, embed_weights)
    dist, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print "   ".join(["%s (%.3f)" % (id2word[idx[i]], dist[i]) for i in xrange(len(idx))])


# l2 distance between vectors
def create_l2_dist(embed_weights, inputs, target):
    dist = tf.nn.embedding_lookup(embed_weights, inputs)
    dist = tf.add(dist, [-t for t in target])
    dist = -tf.sqrt(tf.reduce_sum(tf.mul(dist, dist), 1))
    return dist
 
 
 # print nearest words
def print_nearest(embed_weights, inputs, id2word, sess, target, count):
    dist = create_l2_dist(embed_weights, inputs, target)
    _, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print " ".join([id2word[t] for t in idx])


# class for matching word<->id and storing matrixes
class TWord2Vec:
    def __init__(self):
        self.Word2Id = {}
        self.Id2Word = []

    def Init(self, embedding_size):
        vocabulary_size = len(self.Id2Word)
        self.EmbeddingWeights = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        self.NCEWeights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        self.NCEBiases = tf.Variable(tf.zeros([vocabulary_size]))

    def CheckWordListConsistency(self, data):
        if len(self.Word2Id) != len(data["Word2Id"]) or len(self.Id2Word) != len(data["Id2Word"]):
            return False
        for a, b in zip(self.Id2Word, data["Id2Word"]):
            if a != b.encode("utf-8") or self.Word2Id[a] != data["Word2Id"][b]:
                return False
        return True

    def Load(self, path):
        try:
            s = open(path, "rt").read()
        except:
            return False
        if len(s) == 0:
            return False
        data = json.loads(s)
        if not self.CheckWordListConsistency(data):
            return False
        #self.Word2Id = data["Word2Id"]
        #self.Id2Word = data["Id2Word"]
        self.EmbeddingWeights = tf.Variable(data["EmbeddingWeights"])
        self.NCEWeights = tf.Variable(data["NCEWeights"])
        self.NCEBiases = tf.Variable(data["NCEBiases"])
        return True

    def Save(self, path, sess):
        data = {}
        data["Word2Id"] = self.Word2Id
        data["Id2Word"] = self.Id2Word
        data["EmbeddingWeights"] = sess.run(self.EmbeddingWeights).tolist()
        data["NCEWeights"] = sess.run(self.NCEWeights).tolist()
        data["NCEBiases"] = sess.run(self.NCEBiases).tolist()
        open(path, "wt").write(json.dumps(data))


# do all stuff
def main():
    #with tf.device('/gpu:0'):
        # define params
        params = sys.argv[1:]
        input_path, dump_path, params = params[:2] + [params[2:]]
        learning_rate, eps, valid_part, params = map(float, params[:3]) + [params[3:]]
        embedding_size, batch_size, min_freq, num_sampled, context_width, count_of_nearest, print_freq, save_freq, params = map(int, params[:8]) + [params[8:]]
        words, params = params[:4], params[4:]
        learning_rate /= batch_size
        # read data, make indexes word <-> id
        w2v = TWord2Vec()
        data, w2v.Word2Id, w2v.Id2Word = read_data(input_path, min_freq)
        if w2v.Load(dump_path):
            print "Loaded"
        else:
            print "Failed to load from '%s'" % dump_path
            w2v.Init(embedding_size)
        print_data_stats(data, words, w2v)
        vocabulary_size = len(w2v.Word2Id)
        # input and output placeholders
        inputs = tf.placeholder(tf.int32, shape = [None])
        labels = tf.placeholder(tf.int32, shape = [None, 1])
        # tensor for 'input->embedding' transform
        embed_tensor = tf.nn.embedding_lookup(w2v.EmbeddingWeights, inputs)
        # define loss
        loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights = w2v.NCEWeights,
                                   biases = w2v.NCEBiases,
                                   labels = labels,
                                   inputs = embed_tensor,
                                   num_sampled = num_sampled,
                                   num_classes = vocabulary_size
                                  )
                             )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        #sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
        sess = tf.Session()
        sess.run(init)
        # generate all batches
        all_inputs, all_labels = generate_batch(data, w2v.Word2Id, context_width, 1.0)
        all_batches_inputs, all_batches_labels = [], []
        for i in xrange(0, len(all_inputs), batch_size):
            all_batches_inputs.append(all_inputs[i:i+batch_size])
            all_batches_labels.append(all_labels[i:i+batch_size])
        batches_count = len(all_batches_inputs)
        learn = range(batches_count)
        random.shuffle(learn)
        valid, learn = learn[:int(batches_count * valid_part)], learn[int(batches_count * valid_part):]
        valid_inputs = sum([all_batches_inputs[t] for t in valid], [])
        valid_labels = sum([all_batches_labels[t] for t in valid], [])
        epoch = 0
        while True:
            loss_val, loss_cnt = 0.0, 1e-38
            for learn_idx in learn:
                batch_inputs, batch_labels = all_batches_inputs[learn_idx], all_batches_labels[learn_idx]
                _, l = sess.run([optimizer, loss], feed_dict = {inputs: batch_inputs, labels: batch_labels})
                loss_val += (l * len(batch_inputs))
                loss_cnt += len(batch_inputs)
                #print l, len(batch_inputs)
            loss_val /= loss_cnt
            valid_loss = sess.run([loss], feed_dict = {inputs: valid_inputs, labels: valid_labels})[0] if len(valid_inputs) > 0 else 0.0
            if epoch % print_freq == 0 or loss_val < eps:
                if epoch % save_freq == 0 or loss_val < eps:
                    try:
                        shutil.copy(dump_path, dump_path + ".bak")
                    except:
                        pass
                    w2v.Save(dump_path, sess)
                pred = sess.run([embed_tensor], feed_dict = {inputs: [w2v.Word2Id[t] for t in words]})
                a = [float(t) for t in pred[0][0] - pred[0][1]]
                b = [float(t) for t in pred[0][2] - pred[0][3]]
                c = [float(t) for t in pred[0][0] - pred[0][1] - pred[0][2] + pred[0][3]]
                d = [float(t) for t in pred[0][2] + pred[0][1] - pred[0][0]]
                e = [float(t) for t in pred[0][0]]
                a_abs = math.sqrt(sum([t * t for t in a]))
                b_abs = math.sqrt(sum([t * t for t in b]))
                c_abs = math.sqrt(sum([t * t for t in c]))
                a = [t / a_abs for t in a]
                b = [t / b_abs for t in b]
                print "%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f" % (loss_val, valid_loss, sum([i * j for i, j in zip(a, b)]), a_abs, b_abs, c_abs)
                print_analogy(pred[0][0], pred[0][1], pred[0][2], inputs, w2v.EmbeddingWeights, w2v.Id2Word, sess, count_of_nearest)
                print_nearest(w2v.EmbeddingWeights, inputs, w2v.Id2Word, sess, d, count_of_nearest)
                print_nearest(w2v.EmbeddingWeights, inputs, w2v.Id2Word, sess, e, count_of_nearest)
                print
            if loss_val < eps:
                break
            epoch += 1


# entry point
if __name__ == "__main__":
    main()

