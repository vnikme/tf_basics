#!/usr/bin/python
# encoding: utf-8


import numpy as np
import tensorflow as tf
import base64, fnmatch, json, math, os, random, shutil, sys


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".decode("utf-8")
DIGITS = "0123456789".decode("utf-8")
SENTBRK = "?!.".decode("utf-8")
SPACES = " \n\t\r".decode("utf-8")
WORDBRK = "|^{}[]©<>&`~'\"(),*+-_=/\%/\\$#@:;".decode("utf-8") + SENTBRK + SPACES
ALL_SYMS = LETTERS + DIGITS + WORDBRK
GO = len(ALL_SYMS)
STOP = GO + 1
VOCABULARY_SIZE = STOP + 1


def iterate_files_in_dir(path, mask):
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            yield os.path.join(root, filename)


def iterate_chars(path, mask, enc):
    files = list(iterate_files_in_dir(path, mask))
    n = len(files)
    for i in xrange(n):
        data = []
        try:
            #print "%s\t%.3f" % (files[i], i * 100.0 / n)
            for ch in open(files[i], "rt").read().decode(enc):
                if ch in ALL_SYMS:
                    data.append(ch)
                else:
                    if ch in u"\u0014\u0015":
                        data.append(" ")
                        continue
                    raise 1
        except:
            #print "%s\t%.3f" % (files[i], i * 100.0 / n)
            data = []
        for ch in data:
            yield ch


def _iterate_words(char_iterator):
    is_space, word = True, ""
    for ch in char_iterator:
        if ch in WORDBRK:
            if word:
                yield word
                word = ""
            if ch in SPACES:
                ch = " "
                if not is_space:
                    yield ch
                    is_space = True
            else:
                is_space = False
                yield ch
        else:
            if ch in LETTERS + DIGITS:
                is_space = False
                word += ch
    if word:
        yield word


def iterate_words(char_iterator, word_len):
    for word in _iterate_words(char_iterator):
        for i in xrange(0, len(word), word_len):
            yield word[i : i + word_len]


def word_to_codes(word, word_len):
    word = [ALL_SYMS.index(ch) for ch in word]
    k = word_len - len(word)
    if k > 0:
        word += ([STOP] * k)
    return word


class TWord:
    def __init__(self, word):
        word = [GO] + word + [STOP]
        l = len(word)
        word = np.array(word)
        self.count = 1
        self.word = word[1 : l]             # input and target
        self.dword = word[0 : l - 1]        # input to decoder


def read_words(word_len):
    data = {}
    for src_word in iterate_words(iterate_chars("lib_ru/public_html/book", "*.txt", "koi8-r"), word_len - 1):
    #for src_word in iterate_words(iterate_chars("data", "all", "utf-8"), word_len - 1):
        if src_word not in data:
            word = word_to_codes(src_word, word_len - 1)
            data[src_word] = TWord(word)
        else:
            data[src_word].count += 1
        #if len(data) >= 1000:
        #    break
    return data


def iterate_batches(data, batch_size, words_to_take):
    words = data.keys()
    distr = []
    for w in words:
        distr.append(math.sqrt(data[w].count))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    x, dx = [], []
    for i in xrange(words_to_take):
        k = int(np.searchsorted(cs, np.random.rand(1) * s))
        k = min(max(k, 0), len(distr) - 1)
        txt = words[k]
        word = data[txt]
        x.append(word.word)
        dx.append(word.dword)
    for i in xrange(0, len(x), batch_size):
        yield x[i : i + batch_size], dx[i : i + batch_size]


# input shape: batch*time*input_state
# output shape: batch*output_state
# flatterns all states and multiplies them on projection matrix
def attention(inp, max_time, input_state_size, output_state_size, w, b):
    output = tf.reshape(inp, [-1, max_time * input_state_size])
    output = tf.add(tf.matmul(output, w), b)
    return output


# input shape: batch*time*state
# output shape: batch*time*vocabulary
# multiplies last dimention by `w` and adds `b`
def projection(inp, state_size, max_time, vocabulary_size, w, b):
    output = tf.reshape(inp, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    return output

def choose_random(distr):
    #print "\t\t".join(map(lambda i: "%s: %.7f" % (ALL_SYMS[i].encode("utf-8") if i < len(ALL_SYMS) else "<spec>", distr[i]), range(len(distr))))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    k = int(np.searchsorted(cs, np.random.rand(1) * s))
    return min(max(k, 0), len(distr) - 1)


class TWordPackager:
    def __init__(self, embedding_size, word_len, state_size, vocabulary_size):
        self.embedding_size = embedding_size
        self.word_len = word_len
        self.state_size = state_size
        self.vocabulary_size = vocabulary_size
        # create encoder variables
        with tf.variable_scope('encoder'):
            self.encoder_input = tf.placeholder(tf.int32, [None, self.word_len])
            self.embedding_weights = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -0.01, 0.01), name = "emb")
            self.encoder_w = tf.Variable(tf.random_normal([self.word_len * self.embedding_size, self.state_size], -0.01, 0.01), name = "ew")
            self.encoder_b = tf.Variable(tf.random_normal([self.state_size], -0.01, 0.01), name = "eb")
            embed_tensor = tf.nn.embedding_lookup(self.embedding_weights, self.encoder_input)
            self.encoder_output = attention(embed_tensor, self.word_len, self.embedding_size, self.state_size, self.encoder_w, self.encoder_b)
        # create decoder variables
        with tf.variable_scope('decoder'):
            self.decoder_input = tf.placeholder(tf.int32, [None, self.word_len])
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
            self.decoder_w = tf.Variable(tf.random_normal([self.state_size, self.vocabulary_size], -0.01, 0.01), name = "dw")
            self.decoder_b = tf.Variable(tf.random_normal([self.vocabulary_size], -0.01, 0.01), name = "db")
            output, state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(self.decoder_input, self.vocabulary_size, on_value = 1.0), initial_state = self.encoder_output, dtype = tf.float32)
            self.decoder_output = projection(output, self.state_size, self.word_len, self.vocabulary_size, self.decoder_w, self.decoder_b)
        # create decoder variables for applying
        with tf.variable_scope('decoder', reuse = True):
            self.apply_decoder_input_placeholder = tf.placeholder(tf.int32, [None, 1])
            self.apply_decoder_state_input_placeholder = tf.placeholder(tf.float32, [None, self.state_size])
            output, self.apply_decoder_state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(self.apply_decoder_input_placeholder, vocabulary_size, on_value = 1.0), initial_state = self.apply_decoder_state_input_placeholder, dtype = tf.float32)
            output = projection(output, self.state_size, 1, self.vocabulary_size, self.decoder_w, self.decoder_b)
            # renorm output logits for sampling
            self.apply_decoder_output = tf.nn.softmax(output)

    def to_json(self, sess):
        m = {}
        for var in tf.global_variables():
            if not var.name.startswith("encoder/") and not var.name.startswith("decoder/"):
                continue
            m[var.name] = var.eval(sess).tolist()
            #print var.name
        return json.dumps(m)

    def from_json(self, js, sess):
        m = json.loads(js)
        ops = []
        for var in tf.global_variables():
            if not var.name.startswith("encoder/") and not var.name.startswith("decoder/"):
                continue
            if var.name not in m:
                print var.name
                continue
            ops.append(tf.assign(var, m[var.name]))
        sess.run(ops)

    def encode_word(self, sess, word):
        word = TWord(word_to_codes(word, self.word_len - 1))
        return sess.run(self.encoder_output, feed_dict = {self.encoder_input: [word.word]})[0]

    def _decode_word(self, sess, word, limit_word_len, choose_func):
        cur_state = word
        cur_sym = GO
        result = ""
        while True:
            probs, cur_state = sess.run([self.apply_decoder_output, self.apply_decoder_state], feed_dict = {self.apply_decoder_input_placeholder: [[cur_sym]], self.apply_decoder_state_input_placeholder: [cur_state]})
            probs, cur_state = probs[0][0], cur_state[0]
            cur_sym = choose_func(probs)
            if cur_sym == STOP or len(result) >= limit_word_len:
                break
            if cur_sym == GO:
                continue
            result += ALL_SYMS[cur_sym]
        return result.encode("utf-8")

    def decode_word_sample(self, sess, word, limit_word_len):
        return self._decode_word(sess, word, limit_word_len, lambda probs: choose_random(probs))

    def decode_word_max(self, sess, word, limit_word_len):
        return self._decode_word(sess, word, limit_word_len, lambda probs: np.argmax(probs))


def some_fixed_text(word_len):
    text = "При этом экс-губернатор крайне разозлился тем, что в Интернете активно распространяются фотографии, где он выглядит, как бомж, прибывший на знаковое политическое событие в Соединенные Штаты. Так, на одном из снимков видно, что Саакашвили в непрезентабельной одежде стоит в зале аэропорта Нью-Йорка. Создается впечатление, что утром он был сильно пьян и поэтому надел на себя первое, что попалось ему под руку. На другом снимке Саакашвили одиноко стоит в кустах с телефоном в руке в то время, когда основная публика находится на инаугурации Трампа."
    return iterate_words(text.decode("utf-8"), word_len - 1)


def iterate_keyboard_input(word_len):
    return iterate_words(raw_input().decode("utf-8"), word_len - 1)


def sample_words(wp, sess, limit_word_len, word_iterator):
    original, predicted_max, predicted_rand = "", "", ""
    for word in word_iterator:
        original += word.encode("utf-8")
        word = wp.encode_word(sess, word)
        predicted_max += wp.decode_word_max(sess, word, limit_word_len)
        predicted_rand += wp.decode_word_sample(sess, word, limit_word_len)
    return original, predicted_max, predicted_rand


def correct_learning_rate_multiplier(losses):
    if len(losses) >= 2 and losses[-2] * 2 < losses[-1]:
        return 1.0
    return 1.0


def main():
    # define params
    batch_size, words_in_batch, max_word_len, embedding_size, state_size, limit_word_len, min_gap = 2000, 1000000, 25, 8, 64, 50, 5.0
    learning_rate = tf.Variable(0.001, trainable = False)

    wp = TWordPackager(embedding_size, max_word_len, state_size, VOCABULARY_SIZE)

    # prepare variables
    sess = tf.Session()

    # create loss and optimizer
    loss = None
    ohy = tf.one_hot(wp.encoder_input, wp.vocabulary_size)
    dv = tf.mul(wp.decoder_output, ohy)
    dv = tf.reduce_sum(dv, 2)
    for i in xrange(wp.vocabulary_size):
        dvi = wp.decoder_output[:, :, i]
        l = tf.maximum(dvi - dv + min_gap, 0.0)
        if loss == None:
            loss = l
        else:
            loss = loss + l
    loss = loss / wp.vocabulary_size
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    # initialize global variables
    init = tf.global_variables_initializer()
    sess.run(init)

    #wp.from_json(open("dump.char", "rt").read(), sess)

    print "Models created"

    if False:
        wp.from_json(open("dump.char", "rt").read(), sess)
        original, predicted_max, predicted_rand = sample_words(wp, sess, limit_word_len, some_fixed_text(max_word_len))
        print original
        print predicted_max
        print predicted_rand
        print
        sys.stdout.flush()
    elif False:
        wp.from_json(open("dump.char.25_8_64_5", "rt").read(), sess)
        while True:
            print sample_words(wp, sess, limit_word_len, iterate_keyboard_input(max_word_len))[1]

    # read all data
    data = read_words(max_word_len)
    print "Total number of words:", len(data)

    epoch = 0
    losses = []
    while True:
        if epoch % 10 == 0:
            all_batches = []
            for batch_x, batch_dx in iterate_batches(data, batch_size, words_in_batch):
                all_batches.append([batch_x, batch_dx])
        cnt, l = 1e-38, 0.0
        for batch_x, batch_dx in all_batches:
            cnt += 1
            _, _l = sess.run([optimizer, loss], feed_dict = {wp.encoder_input: batch_x, wp.decoder_input: batch_dx})
            l += _l
        losses.append(l / cnt)
        #learning_rate *= correct_learning_rate_multiplier(losses)
        print "loss: %f\tlearning rate: %.6f\tepoch: %d" % (l / cnt, sess.run(learning_rate), epoch)
        if epoch % 10 == 0:
            original, predicted_max, predicted_rand = sample_words(wp, sess, limit_word_len, some_fixed_text(max_word_len))
            #print original
            print predicted_max
            #print predicted_rand
            print
            try:
                shutil.copy("dump.char", "dump.char.bak")
            except:
                pass
            open("dump.char", "wt").write(wp.to_json(sess))
        sys.stdout.flush()
        epoch += 1


# entry point
if __name__ == "__main__":
    main()

