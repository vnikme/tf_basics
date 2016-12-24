#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import json, math, numpy, random, shutil, sys


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя".decode("utf-8")
def is_letter(ch):
    return ch in all_syms


# read lines, yield symbols
def iterate_symbols(path):
    c = 0
    for line in open(path):
        c += 1
        if c % 1000000 == 0:
            print c / 1000000
            sys.stdout.flush()
            if c / 1000000 == 50:
                break
        line = to_wide_lower(line)
        for ch in line:
            yield ch
    yield " "


# read symbols, split on words, yield
def iterate_words(path):
    word = ""
    for ch in iterate_symbols(path):
        if not is_letter(ch):
            if len(word) != 0:
                yield word.encode("utf-8")
            word = ""
        else:
            word += ch


# read file, split words, return
def read_data(path, words_to_take):
    data, id2word, word_freqs = [], ["<unk>"], [0]
    word2id = {"<unk>": 0}
    c = 0
    for word in iterate_words(path):
        c += 1
        #if c == 1000:
        #    break
        if word not in word2id:
            word2id[word] = len(id2word)
            id2word.append(word)
            word_freqs.append(0)
            #print "%d\t%d\t%s" % (len(data), len(id2word), word)
        id_ = word2id[word]
        data.append(id_)
        word_freqs[id_] += 1
    print "Converting to ndarray"
    sys.stdout.flush()
    data = numpy.asarray(data)
    print "Filtering input"
    sys.stdout.flush()
    n = len(word_freqs)
    idx = range(n)
    idx.sort(key = lambda x: -word_freqs[x])
    min_freq = word_freqs[idx[min(words_to_take, len(word_freqs) - 1)]]
    idx, new_id2word, new_word_freqs = [0] + [-1 for i in xrange(1, n)], ["<unk>"], [0]
    for i in xrange(1, n):
        if word_freqs[i] >= min_freq:
            idx[i] = len(new_id2word)
            new_id2word.append(id2word[i])
            new_word_freqs.append(word_freqs[i])
        else:
            idx[i] = 0
            new_word_freqs[0] += word_freqs[i]
    id2word = new_id2word
    word_freqs = new_word_freqs
    for w in word2id.keys():
        word2id[w] = idx[word2id[w]]
    for i in xrange(len(data)):
        data[i] = idx[data[i]]
    print "Data read"
    sys.stdout.flush()
    return data, word2id, id2word, word_freqs


# print some simple statistics
def print_data_stats(data, words, w2v):
    idx = range(len(w2v.Id2Word))
    idx.sort(key = lambda x: -w2v.WordFreqs[x])
    lens = {}
    for word in idx[:100]:
        print w2v.Id2Word[word], w2v.WordFreqs[word]
    print len(data), len(w2v.Id2Word)
    sys.stdout.flush()
    #for word in d.keys():
    #    l = d[word]
    #    lens[l] = lens.get(l, 0) + 1
    #for k in sorted(lens.keys()):
    #    print k, lens[k]
    #exit(1)


# make arrays with learning data
def generate_learning_data(data, context_width):
    n = len(data)
    idx = range(n - context_width * 2)
    random.shuffle(idx)
    inputs, labels = numpy.ndarray([n - context_width * 2]), numpy.ndarray([n - context_width * 2, context_width * 2])
    for k in xrange(context_width, n - context_width):
        word = data[k]
        inputs[idx[k - context_width]] = word
        m = 0
        for j in xrange(-context_width, context_width + 1):
            if j == 0:
                continue
            context_word = data[k + j]
            labels[idx[k - context_width]][m] = context_word
            m += 1
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
def create_cos_dist(a, b, c, d):
    b = tf.subtract(b, a)
    d = tf.subtract(d, c)
    b = tf.divide(b, tf.transpose([get_vector_norm(b)]))
    d = tf.divide(d, tf.transpose([get_vector_norm(d)]))
    return tf.reduce_sum(tf.mul(b, d), 1)


# print nearest words
def print_analogy(a, b, c, inputs, embed_tensor, id2word, sess, count):
    dist = create_cos_dist(a, b, c, embed_tensor)
    dist, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print "   ".join(["%s (%.3f)" % (id2word[idx[i]], dist[i]) for i in xrange(len(idx))])


# l2 distance between vectors
def create_l2_dist(embed_tensor, target):
    dist = embed_tensor
    dist = tf.add(dist, [-t for t in target])
    dist = -tf.sqrt(tf.reduce_sum(tf.mul(dist, dist), 1))
    return dist
 
 
 # print nearest words
def print_nearest(embed_tensor, inputs, id2word, sess, target, count):
    dist = create_l2_dist(embed_tensor, target)
    _, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print " ".join([id2word[t] for t in idx])


# class for matching word<->id and storing matrixes
class TWord2Vec:
    def __init__(self):
        self.Word2Id = {}
        self.Id2Word = []
        self.WordFreqs = []

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
        learning_rate, eps, params = map(float, params[:2]) + [params[2:]]
        embedding_size, batch_size, valid_size, words_to_take, num_sampled, context_width, count_of_nearest, print_freq, save_freq, params = map(int, params[:9]) + [params[9:]]
        words, params = params[:4], params[4:]
        #learning_rate /= batch_size
        # read data, make indexes word <-> id
        w2v = TWord2Vec()
        data, w2v.Word2Id, w2v.Id2Word, w2v.WordFreqs = read_data(input_path, words_to_take)
        if w2v.Load(dump_path):
            print "Loaded"
        else:
            print "Failed to load from '%s'" % dump_path
            w2v.Init(embedding_size)
        sys.stdout.flush()
        print_data_stats(data, words, w2v)
        vocabulary_size = len(w2v.Id2Word)
        # input and output placeholders
        inputs = tf.placeholder(tf.int32, shape = [None])
        labels = tf.placeholder(tf.int32, shape = [None, context_width * 2])
        # tensor for 'input->embedding' transform
        embed_tensor = tf.nn.embedding_lookup(w2v.EmbeddingWeights, inputs)
        embed_tensor = tf.nn.sigmoid(embed_tensor)
        # define loss
        loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights = w2v.NCEWeights,
                                   biases = w2v.NCEBiases,
                                   labels = labels,
                                   inputs = embed_tensor,
                                   num_sampled = num_sampled,
                                   num_classes = vocabulary_size,
                                   num_true = context_width * 2,
                                   remove_accidental_hits = False
                                  )
                             )
        full_loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights = w2v.NCEWeights,
                                   biases = w2v.NCEBiases,
                                   labels = labels,
                                   inputs = embed_tensor,
                                   num_sampled = vocabulary_size,
                                   num_classes = vocabulary_size,
                                   num_true = context_width * 2,
                                   remove_accidental_hits = False
                                  )
                             )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        #sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
        sess = tf.Session()
        sess.run(init)
        # generate all batches
        all_inputs, all_labels = generate_learning_data(data, context_width)
        del data
        valid_inputs = all_inputs[:valid_size]
        valid_labels = all_labels[:valid_size]
        all_inputs = all_inputs[valid_size:]
        all_labels = all_labels[valid_size:]
        all_batches_inputs, all_batches_labels = [], []
        for i in xrange(0, len(all_inputs), batch_size):
            all_batches_inputs.append(all_inputs[i:i+batch_size])
            all_batches_labels.append(all_labels[i:i+batch_size])
        batches_count = len(all_batches_inputs)
        print len(all_inputs), len(valid_inputs), batches_count
        sys.stdout.flush()
        all_batches_inputs = numpy.asarray(all_batches_inputs)
        all_batches_labels = numpy.asarray(all_batches_labels)
        valid_inputs = numpy.asarray(valid_inputs)
        valid_labels = numpy.asarray(valid_labels)
        del all_inputs
        del all_labels
        print "Begin training"
        sys.stdout.flush()
        epoch = 0
        while True:
            for batch_inputs, batch_labels in zip(all_batches_inputs, all_batches_labels):
                sess.run([optimizer], feed_dict = {inputs: batch_inputs, labels: batch_labels})
            valid_loss = sess.run([full_loss], feed_dict = {inputs: valid_inputs, labels: valid_labels})[0] if len(valid_inputs) > 0 else 0.0
            if epoch % print_freq == 0:
                if epoch % save_freq == 0 or valid_loss < eps:
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
                print "%.2f\t%.4f\t%.4f\t%.4f\t%.4f" % (valid_loss, sum([i * j for i, j in zip(a, b)]), a_abs, b_abs, c_abs)
                print_analogy(pred[0][0], pred[0][1], pred[0][2], inputs, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
                print_nearest(embed_tensor, inputs, w2v.Id2Word, sess, d, count_of_nearest)
                print_nearest(embed_tensor, inputs, w2v.Id2Word, sess, e, count_of_nearest)
                print
                sys.stdout.flush()
                if valid_loss < eps:
                    break
            epoch += 1


# entry point
if __name__ == "__main__":
    main()

