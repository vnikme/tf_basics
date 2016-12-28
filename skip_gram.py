#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import json, math, numpy, random, shutil, sys


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя-".decode("utf-8")
allowed_syms = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя-".decode("utf-8")
tech_syms = "-".decode("utf-8")


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
            #if c / 10000 >= 50:
            #    break
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
                yield word
            word = ""
        else:
            word += ch


# filter words by language
def is_allowed_word(word):
    tech_only = True
    for ch in word:
        if ch not in allowed_syms:
            return False
        if ch not in tech_syms:
            tech_only = False
    return not tech_only


# read file, split words, return
def read_data(path, words_to_take):
    data, id2word, word_freqs = [], ["<unk>"], [0]
    word2id = {"<unk>": 0}
    c = 0
    for word in iterate_words(path):
        c += 1
        if not is_allowed_word(word):
            continue
        #if c == 1000:
        #    break
        word = word.encode("utf-8")
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
    fidx = range(n)
    fidx.sort(key = lambda x: -word_freqs[x])
    min_freq = word_freqs[fidx[min(words_to_take, len(word_freqs) - 1)]]
    idx, new_id2word, new_word_freqs = [0] + [-1 for i in xrange(1, n)], ["<unk>"], [0]
    for k in xrange(n):
        i = fidx[k]
        if i == 0:
            continue
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
def print_data_stats(data, w2v):
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
    non_zero, total = 0, len(data)
    for word in data[context_width:total - context_width]:
        if word != 0:
            non_zero += 1
    idx = range(non_zero)
    random.shuffle(idx)
    inputs, labels = numpy.ndarray([non_zero]), numpy.ndarray([non_zero, context_width * 2])
    current = 0
    for k in xrange(context_width, total - context_width):
        word = data[k]
        if word == 0:
            continue
        inputs[idx[current]] = word
        m = 0
        for j in xrange(-context_width, context_width + 1):
            if j == 0:
                continue
            context_word = data[k + j]
            labels[idx[current]][m] = context_word
            m += 1
        current += 1
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
def create_cos_dist1(a, b, c, d):
    a = tf.divide(a, tf.transpose([get_vector_norm(a)]))
    b = tf.divide(b, tf.transpose([get_vector_norm(b)]))
    c = tf.divide(c, tf.transpose([get_vector_norm(c)]))
    c = tf.add(c, tf.subtract(b, a))
    c = tf.divide(c, tf.transpose([get_vector_norm(c)]))
    d = tf.divide(d, tf.transpose([get_vector_norm(d)]))
    return tf.reduce_sum(tf.mul(c, d), 1)

def create_cos_dist2(a, b, c, d):
    b = tf.subtract(b, a)
    d = tf.subtract(d, c)
    b = tf.divide(b, tf.transpose([get_vector_norm(b)]))
    d = tf.divide(d, tf.transpose([get_vector_norm(d)]))
    return tf.reduce_sum(tf.mul(b, d), 1)

def create_cos_dist3(a, b, c, d):
    b = tf.sigmoid(tf.subtract(b, a))
    d = tf.sigmoid(tf.subtract(d, c))
    b = tf.divide(b, tf.transpose([get_vector_norm(b)]))
    d = tf.divide(d, tf.transpose([get_vector_norm(d)]))
    return tf.reduce_sum(tf.mul(b, d), 1)

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


# l2 distance between vectors
def create_l2_dist(embed_tensor, target):
    dist = embed_tensor
    #dist = tf.divide(dist, tf.transpose([get_vector_norm(dist)]))
    dist = tf.add(dist, [-t for t in target])
    dist = -tf.sqrt(tf.reduce_sum(tf.mul(dist, dist), 1))
    return dist
 
 
 # print nearest words
def print_nearest(embed_tensor, inputs, id2word, sess, target, count):
    dist = create_l2_dist(embed_tensor, target)
    dist, idx = sess.run(tf.nn.top_k(dist, count), feed_dict = {inputs: range(len(id2word))})
    print "   ".join(["%s (%.3f)" % (id2word[idx[i]], -dist[i]) for i in xrange(len(idx))])


# class for matching word<->id and storing matrixes
class TWord2Vec:
    def __init__(self):
        self.Word2Id = {}
        self.Id2Word = []
        self.WordFreqs = []

    def Init(self, embedding_size):
        vocabulary_size = len(self.Id2Word)
        hidden_size = embedding_size # * 4
        self.EmbeddingWeights = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -0.01, 0.01))
        #self.HiddenWeights = tf.Variable(tf.random_uniform([embedding_size, hidden_size], -0.01, 0.01))
        #self.HiddenBiases = tf.Variable(tf.zeros([hidden_size]))
        self.NCEWeights = tf.Variable(tf.random_normal([vocabulary_size, hidden_size], -0.01, 0.01))
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
        #self.HiddenWeights = tf.Variable(data["HiddenWeights"])
        #self.HiddenBiases = tf.Variable(data["HiddenBiases"])
        self.NCEWeights = tf.Variable(data["NCEWeights"])
        self.NCEBiases = tf.Variable(data["NCEBiases"])
        return True

    def Save(self, path, sess):
        data = {}
        data["Word2Id"] = self.Word2Id
        data["Id2Word"] = self.Id2Word
        data["EmbeddingWeights"] = sess.run(self.EmbeddingWeights).tolist()
        #data["HiddenWeights"] = sess.run(self.HiddenWeights).tolist()
        #data["HiddenBiases"] = sess.run(self.HiddenBiases).tolist()
        data["NCEWeights"] = sess.run(self.NCEWeights).tolist()
        data["NCEBiases"] = sess.run(self.NCEBiases).tolist()
        open(path, "wt").write(json.dumps(data))


def normalize_vector(x):
    x = [float(t) for t in x]
    x_abs = math.sqrt(sum([t * t for t in x]))
    x = [t / x_abs for t in x]
    return x


def print_analogies(sess, embed_tensor, inputs, w2v, base_words, words, count_of_nearest):
    nemb = tf.sigmoid(embed_tensor)
    pred, npred = sess.run([embed_tensor, nemb], feed_dict = {inputs: [w2v.Word2Id[t] for t in (base_words + words)]})
    print "%s\t%s\t%s\t%s" % (base_words[0], base_words[1], words[0], words[1])
    a, b, c, d = [[pred[i][j] for j in xrange(len(pred[i]))] for i in xrange(len(pred))]
    na, nb, nc, nd = [[npred[i][j] for j in xrange(len(npred[i]))] for i in xrange(len(npred))]
    e = [c[i] - a[i] + b[i] for i in xrange(len(a))]
    ne = [nc[i] - na[i] + nb[i] for i in xrange(len(na))]
    #for v in [na, nb, nc, nd, ne]:
    #    print "\t".join(map(lambda x: "%.2f" % x, v))
    print_nearest(nemb, inputs, w2v.Id2Word, sess, nd, count_of_nearest)
    print_analogy(na, nb, nc, inputs, create_cos_dist1, nemb, w2v.Id2Word, sess, count_of_nearest)
    print_analogy(na, nb, nc, inputs, create_cos_dist2,  nemb, w2v.Id2Word, sess, count_of_nearest)
    print_analogy(a, b, c, inputs, create_cos_dist3, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
    print_nearest(nemb, inputs, w2v.Id2Word, sess, ne, count_of_nearest)
    print_analogy(a, b, c, inputs, create_cos_dist4, embed_tensor, w2v.Id2Word, sess, count_of_nearest)
    print
    sys.stdout.flush()


# do all stuff
def main():
    #with tf.device('/gpu:0'):
        # define params
        params = sys.argv[1:]
        input_path, dump_path, params = params[:2] + [params[2:]]
        learning_rate, eps, params = map(float, params[:2]) + [params[2:]]
        embedding_size, batch_size, valid_size, words_to_take, num_sampled, context_width, count_of_nearest, print_freq, save_freq, params = map(int, params[:9]) + [params[9:]]
        base_words, words = params[:2], params[2:]
        words = [[words[i], words[i + 1]] for i in xrange(0, len(words), 2)]
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
        print_data_stats(data, w2v)
        vocabulary_size = len(w2v.Id2Word)
        # generate all batches
        all_inputs, all_labels = generate_learning_data(data, context_width)
        del data
        print "Learning data generated"
        sys.stdout.flush()
        valid_inputs = all_inputs[:valid_size]
        valid_labels = all_labels[:valid_size]
        valid_inputs = numpy.asarray(valid_inputs)
        valid_labels = numpy.asarray(valid_labels)
        all_inputs = all_inputs[valid_size:]
        all_labels = all_labels[valid_size:]
        all_batches_inputs, all_batches_labels = [], []
        print "Converting inputs"
        sys.stdout.flush()
        for i in xrange(0, len(all_inputs), batch_size):
            all_batches_inputs.append(all_inputs[i:i+batch_size])
        batches_count = len(all_batches_inputs)
        print len(all_inputs), len(valid_inputs), batches_count
        sys.stdout.flush()
        all_batches_inputs = numpy.asarray(all_batches_inputs)
        del all_inputs
        print "Converting labels"
        sys.stdout.flush()
        for i in xrange(0, len(all_labels), batch_size):
            all_batches_labels.append(all_labels[i:i+batch_size])
        all_batches_labels = numpy.asarray(all_batches_labels)
        del all_labels
        print "Creating tensors"
        sys.stdout.flush()
        # input and output placeholders
        inputs = tf.placeholder(tf.int32, shape = [None])
        labels = tf.placeholder(tf.int32, shape = [None, context_width * 2])
        # tensor for 'input->embedding' transform
        embed_tensor = tf.nn.embedding_lookup(w2v.EmbeddingWeights, inputs)
        #embed_tensor = tf.nn.sigmoid(embed_tensor)
        # define loss
        hidden = embed_tensor
        #hidden = tf.add(tf.matmul(embed_tensor, w2v.HiddenWeights), w2v.HiddenBiases)
        #hidden = tf.nn.sigmoid(hidden)
        loss = tf.reduce_mean(
                    #tf.nn.nce_loss(weights = w2v.NCEWeights,
                    tf.nn.sampled_softmax_loss(weights = w2v.NCEWeights,
                                   biases = w2v.NCEBiases,
                                   labels = labels,
                                   inputs = hidden,
                                   num_sampled = num_sampled,
                                   num_classes = vocabulary_size,
                                   num_true = context_width * 2,
                                   #remove_accidental_hits = True
                                  )
                             )
        full_loss = tf.reduce_mean(
                    #tf.nn.nce_loss(weights = w2v.NCEWeights,
                    tf.nn.sampled_softmax_loss(weights = w2v.NCEWeights,
                                   biases = w2v.NCEBiases,
                                   labels = labels,
                                   inputs = hidden,
                                   num_sampled = vocabulary_size,
                                   num_classes = vocabulary_size,
                                   num_true = context_width * 2,
                                   #remove_accidental_hits = False
                                  )
                             )
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        #sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
        sess = tf.Session()
        sess.run(init)
        print "Begin training"
        sys.stdout.flush()
        epoch = 0
        while True:
            for batch_inputs, batch_labels in zip(all_batches_inputs, all_batches_labels):
                sess.run([optimizer], feed_dict = {inputs: batch_inputs, labels: batch_labels})
            valid_loss = sess.run([full_loss], feed_dict = {inputs: valid_inputs, labels: valid_labels})[0] if len(valid_inputs) > 0 else 0.0
            if epoch % print_freq == 0:
                if epoch % save_freq == 0 or valid_loss < eps:
                #if epoch % save_freq == 0:
                    try:
                        shutil.copy(dump_path, dump_path + ".bak")
                    except:
                        pass
                    w2v.Save(dump_path, sess)

                print "Validation loss:\t%.2f" % (valid_loss)
                for pair in words:
                    print_analogies(sess, embed_tensor, inputs, w2v, base_words, pair, count_of_nearest)
                print
                if valid_loss < eps:
                    break
            epoch += 1


# entry point
if __name__ == "__main__":
    main()

