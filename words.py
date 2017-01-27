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


def iterate_chars(path, mask):
    files = list(iterate_files_in_dir(path, mask))
    n = len(files)
    for i in xrange(n):
        data = []
        try:
            #print "%s\t%.3f" % (files[i], i * 100.0 / n)
            for ch in open(files[i], "rt").read().decode("koi8-r"):
                if ch in ALL_SYMS:
                    data.append(ch)
                else:
                    if ch in u"\u0014\u0015":
                        data.append(" ")
                        continue
                    #raise 1
        except:
            #print "%s\t%.3f" % (files[i], i * 100.0 / n)
            data = []
        for ch in data:
            yield ch


def word_to_codes(word):
    return [ALL_SYMS.index(ch) for ch in word]


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


def iterate_words(char_iterator, max_word_len):
    for word in _iterate_words(char_iterator):
        for i in xrange(0, len(word), max_word_len):
            yield word[i : i + max_word_len]


def make_targets(word, max_word_len):
    word = word_to_codes(word)
    l = len(word)
    target = word + [STOP]
    dword = [GO] + word
    k = max_word_len - l
    if k > 0:
        word += ([STOP] * k)
        dword += ([STOP] * k)
        target += ([STOP] * k)
    return word, dword, target


class TWord:
    def __init__(self, word, dword, target):
        self.count = 1
        self.word = word
        self.dword = dword
        self.target = target


def read_words(max_word_len):
    data = {}
    for src_word in iterate_words(iterate_chars("lib_ru/public_html/book", "*.txt"), max_word_len):
    #for src_word in iterate_words(iterate_chars("data", "all"), max_word_len):
        if src_word not in data:
            word, dword, target = make_targets(src_word, max_word_len)
            data[src_word] = TWord(word, dword, target)
        else:
            data[src_word].count += 1
        #if len(data) >= 1000:
        #    break
    return data


def iterate_batches(data, batch_size):
    words = data.keys()
    random.shuffle(words)
    batch_x, batch_dx, batch_y, batch_m = [], [], [], []
    for txt in words:
        word = data[txt]
        batch_x.append(word.word)
        batch_dx.append(word.dword)
        batch_y.append(word.target)
        batch_m.append(math.log(word.count + 1.0))
        if len(batch_x) == batch_size:
            yield batch_x, batch_dx, batch_y, batch_m
            batch_x, batch_dx, batch_y, batch_m = [], [], [], []


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
    #print " ".join(map(lambda t: "%.2f" % t, distr))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    k = int(np.searchsorted(cs, np.random.rand(1) * s))
    return min(max(k, 0), len(distr) - 1)


class TWordPackager:
    def __init__(self, max_word_len, encoder_state_size, decoder_state_size, vocabulary_size):
        # create encoder variables
        with tf.variable_scope('encoder'):
            self.encoder_input_placeholder = tf.placeholder(tf.int32, [None, max_word_len])
            self.encoder_cell = tf.nn.rnn_cell.GRUCell(encoder_state_size)
            self.encoder_w = tf.Variable(tf.random_normal([max_word_len * encoder_state_size, decoder_state_size]), name = "w")
            self.encoder_b = tf.Variable(tf.random_normal([decoder_state_size]), name = "b")
            output, state = tf.nn.dynamic_rnn(self.encoder_cell, tf.one_hot(self.encoder_input_placeholder, vocabulary_size, on_value = 1.0), dtype = tf.float32)
            self.encoder_output = attention(output, max_word_len, encoder_state_size, decoder_state_size, self.encoder_w, self.encoder_b)
        # create decoder variables
        with tf.variable_scope('decoder'):
            self.decoder_input_placeholder = tf.placeholder(tf.int32, [None, max_word_len + 1])
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(decoder_state_size)
            self.decoder_w = tf.Variable(tf.random_normal([decoder_state_size, vocabulary_size]), name = "w")
            self.decoder_b = tf.Variable(tf.random_normal([vocabulary_size]), name = "b")
            output, state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(self.decoder_input_placeholder, vocabulary_size, on_value = 1.0), initial_state = self.encoder_output, dtype = tf.float32)
            self.decoder_output = projection(output, decoder_state_size, max_word_len + 1, vocabulary_size, self.decoder_w, self.decoder_b)
        # create decoder variables for applying
        with tf.variable_scope('decoder', reuse = True):
            self.apply_decoder_input_placeholder = tf.placeholder(tf.int32, [None, 1])
            self.decoder_state_input_placeholder = tf.placeholder(tf.float32, [None, decoder_state_size])
            output, self.apply_decoder_state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(self.apply_decoder_input_placeholder, vocabulary_size, on_value = 1.0), initial_state = self.decoder_state_input_placeholder, dtype = tf.float32)
            output = projection(output, decoder_state_size, 1, vocabulary_size, self.decoder_w, self.decoder_b)
            # renorm output logits for sampling
            self.apply_decoder_output = tf.nn.softmax(output)

    def to_json(self, sess):
        m = {}
        for var in tf.global_variables():
            if not var.name.startswith("encoder/") and not var.name.startswith("decoder/"):
                continue
            m[var.name] = var.eval(sess).tolist()
        return json.dumps(m)

    def from_json(self, js, sess):
        m = json.loads(js)
        for var in tf.global_variables():
            if not var.name.startswith("encoder/") and not var.name.startswith("decoder/"):
                continue
            sess.run(tf.assign(var, m[var.name]))

    def encode_word(self, sess, word, max_word_len):
        word, _, __ = make_targets(word, max_word_len)
        return sess.run(self.encoder_output, feed_dict = {self.encoder_input_placeholder: [word]})[0]

    def _decode_word(self, sess, word, max_word_len, limit_word_len, choose_func):
        cur_state = word
        cur_sym = GO
        result = ""
        while True:
            probs, cur_state = sess.run([self.apply_decoder_output, self.apply_decoder_state], feed_dict = {self.apply_decoder_input_placeholder: [[cur_sym]], self.decoder_state_input_placeholder: [cur_state]})
            probs, cur_state = probs[0][0], cur_state[0]
            cur_sym = choose_func(probs)
            if cur_sym == STOP or len(result) >= limit_word_len:
                break
            if cur_sym == GO:
                continue
            result += ALL_SYMS[cur_sym]
        return result.encode("utf-8")

    def decode_word_sample(self, sess, word, max_word_len, limit_word_len):
        return self._decode_word(sess, word, max_word_len, limit_word_len, lambda probs: choose_random(probs))

    def decode_word_max(self, sess, word, max_word_len, limit_word_len):
        return self._decode_word(sess, word, max_word_len, limit_word_len, lambda probs: np.argmax(probs))


def some_fixed_text(max_word_len):
    text = "При этом экс-губернатор крайне разозлился тем, что в Интернете активно распространяются фотографии, где он выглядит, как бомж, прибывший на знаковое политическое событие в Соединенные Штаты. Так, на одном из снимков видно, что Саакашвили в непрезентабельной одежде стоит в зале аэропорта Нью-Йорка. Создается впечатление, что утром он был сильно пьян и поэтому надел на себя первое, что попалось ему под руку. На другом снимке Саакашвили одиноко стоит в кустах с телефоном в руке в то время, когда основная публика находится на инаугурации Трампа."
    return iterate_words(text.decode("utf-8"), max_word_len)


def iterate_keyboard_input(max_word_len):
    return iterate_words(raw_input().decode("utf-8"), max_word_len)


def sample_words(wp, sess, max_word_len, word_iterator):
    original, predicted = "", ""
    for word in word_iterator:
        original += word.encode("utf-8")
        word = wp.encode_word(sess, word, max_word_len)
        predicted += wp.decode_word_max(sess, word, max_word_len, max_word_len * 3)
    return original, predicted


def main():
    # define params
    max_word_len, batch_size, encoder_state_size, decoder_state_size, learning_rate = 25, 10000, 32, 128, 0.00001

    wp = TWordPackager(max_word_len, encoder_state_size, decoder_state_size, VOCABULARY_SIZE)

    # define loss and optimizer
    mults = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.int32, [None, max_word_len + 1])
    ohy = tf.one_hot(y, VOCABULARY_SIZE, on_value = 1.0)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(wp.decoder_output, ohy)
    loss = tf.reduce_mean(loss, 2)
    loss = tf.reduce_mean(loss, 1)
    loss = tf.mul(loss, mults)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    # prepare variables
    sess = tf.Session()

    if False:
        wp.from_json(open("dump.char", "rt").read(), sess)
        print sample_words(wp, sess, max_word_len, some_fixed_text(max_word_len))[1]
        sys.stdout.flush()
    elif False:
        wp.from_json(open("dump.char", "rt").read(), sess)
        while True:
            print sample_words(wp, sess, max_word_len, iterate_keyboard_input(max_word_len))[1]
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    data = read_words(max_word_len)

    epoch = 0
    while True:
        cnt, l = 1e-38, 0.0
        for batch_x, batch_dx, batch_y, batch_m in iterate_batches(data, batch_size):
            cnt += 1
            _, _l = sess.run([optimizer, loss], feed_dict = {wp.encoder_input_placeholder: batch_x, wp.decoder_input_placeholder: batch_dx, y: batch_y, mults: batch_m})
            l += _l
        print "loss: %f\tepoch: %d" % (l / cnt, epoch)
        original, predicted = sample_words(wp, sess, max_word_len, some_fixed_text(max_word_len))
        print original
        print predicted
        print
        sys.stdout.flush()
        try:
            shutil.copy("dump.char", "dump.char.bak")
        except:
            pass
        open("dump.char", "wt").write(wp.to_json(sess))
        epoch += 1


# entry point
if __name__ == "__main__":
    main()

