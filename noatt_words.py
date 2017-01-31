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
                    raise 1
        except:
            #print "%s\t%.3f" % (files[i], i * 100.0 / n)
            data = []
        for ch in data:
            yield ch


def iterate_words(char_iterator):
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


def word_to_codes(word):
    return [ALL_SYMS.index(ch) for ch in word]


class TWord:
    def __init__(self, word):
        word = [GO] + word + [STOP]
        l = len(word)
        word = np.array(word)
        self.count = 1
        self.word = word[1 : l]             # input and target
        self.dword = word[0 : l - 1]        # input to decoder


def read_words():
    data = {}
    for src_word in iterate_words(iterate_chars("lib_ru/public_html/book", "*.txt")):
    #for src_word in iterate_words(iterate_chars("data", "all")):
        if src_word not in data:
            word = word_to_codes(src_word)
            data[src_word] = TWord(word)
        else:
            data[src_word].count += 1
        #if len(data) >= 1000:
        #    break
    return data


class TBatch:
    def __init__(self):
        self.x, self.dx, self.m = [], [], []


def iterate_batches(data, batch_size, long_word_limit, long_batch_size):
    words = data.keys()
    lens = {}
    for txt in words:
        word = data[txt]
        l = len(word.word)
        if l not in lens:
            lens[l] = TBatch()
        b = lens[l]
        b.x.append(word.word)
        b.dx.append(word.dword)
        #b.m.append(math.sqrt(word.count))
        b.m.append(math.log(word.count + 1.0))
    word_lens = lens.keys()
    random.shuffle(word_lens)
    for word_len in word_lens:
        b = lens[word_len]
        for i in xrange(0, len(b.x), batch_size if word_len <= long_word_limit else long_batch_size):
            yield word_len, b.x[i : i + batch_size], b.dx[i : i + batch_size], b.m[i : i + batch_size]


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
    def __init__(self, state_size, vocabulary_size):
        self.state_size = state_size
        self.vocabulary_size = vocabulary_size
        # create encoder variables
        with tf.variable_scope('encoder'):
            self.encoder_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
            self.encoder_input_placeholders, self.encoder_outputs = {}, {}
            self.encoder_input_placeholders[1], self.encoder_outputs[1] = self._create_encoder(1)
        # create decoder variables
        with tf.variable_scope('decoder'):
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
            self.decoder_input_placeholders, self.decoder_outputs = {}, {}
            self.decoder_w = tf.Variable(tf.random_normal([self.state_size, self.vocabulary_size]), name = "w")
            self.decoder_b = tf.Variable(tf.random_normal([self.vocabulary_size]), name = "b")
            self.decoder_input_placeholders[1], self.decoder_outputs[1] = self._create_decoder(1, self.encoder_outputs[1])
        # create decoder variables for applying
        with tf.variable_scope('decoder', reuse = True):
            self.apply_decoder_input_placeholder = tf.placeholder(tf.int32, [None, 1])
            self.apply_decoder_state_input_placeholder = tf.placeholder(tf.float32, [None, self.state_size])
            output, self.apply_decoder_state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(self.apply_decoder_input_placeholder, vocabulary_size, on_value = 1.0), initial_state = self.apply_decoder_state_input_placeholder, dtype = tf.float32)
            output = projection(output, self.state_size, 1, self.vocabulary_size, self.decoder_w, self.decoder_b)
            # renorm output logits for sampling
            self.apply_decoder_output = tf.nn.softmax(output)

    def _create_encoder(self, word_len):
        placeholder = tf.placeholder(tf.int32, [None, word_len])
        output, state = tf.nn.dynamic_rnn(self.encoder_cell, tf.one_hot(placeholder, self.vocabulary_size, on_value = 1.0), dtype = tf.float32)
        return placeholder, state

    def _create_decoder(self, word_len, encoder_output):
        placeholder = tf.placeholder(tf.int32, [None, word_len])
        output, state = tf.nn.dynamic_rnn(self.decoder_cell, tf.one_hot(placeholder, self.vocabulary_size, on_value = 1.0), initial_state = encoder_output, dtype = tf.float32)
        output = projection(output, self.state_size, word_len, self.vocabulary_size, self.decoder_w, self.decoder_b)
        return placeholder, output

    def full_encoder(self, word_len):
        if word_len not in self.encoder_input_placeholders:
            with tf.variable_scope('encoder', reuse = True):
                self.encoder_input_placeholders[word_len], self.encoder_outputs[word_len] = self._create_encoder(word_len)
        encoder_input, encoder_output = self.encoder_input_placeholders[word_len], self.encoder_outputs[word_len]
        return encoder_input, encoder_output

    def full_encoder_decoder(self, word_len):
        encoder_input, encoder_output = self.full_encoder(word_len)
        if word_len not in self.decoder_input_placeholders:
            with tf.variable_scope('encoder', reuse = True):
                self.decoder_input_placeholders[word_len], self.decoder_outputs[word_len] = self._create_decoder(word_len, encoder_output)
        decoder_input, decoder_output = self.decoder_input_placeholders[word_len], self.decoder_outputs[word_len]
        return encoder_input, decoder_input, decoder_output

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
            ops.append(tf.assign(var, m[var.name]))
        sess.run(ops)

    def encode_word(self, sess, word):
        word = TWord(word_to_codes(word))
        x, op = self.full_encoder(len(word.word))
        return sess.run(op, feed_dict = {x: [word.word]})[0]

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


def some_fixed_text():
    text = "При этом экс-губернатор крайне разозлился тем, что в Интернете активно распространяются фотографии, где он выглядит, как бомж, прибывший на знаковое политическое событие в Соединенные Штаты. Так, на одном из снимков видно, что Саакашвили в непрезентабельной одежде стоит в зале аэропорта Нью-Йорка. Создается впечатление, что утром он был сильно пьян и поэтому надел на себя первое, что попалось ему под руку. На другом снимке Саакашвили одиноко стоит в кустах с телефоном в руке в то время, когда основная публика находится на инаугурации Трампа."
    return iterate_words(text.decode("utf-8"))


def iterate_keyboard_input():
    return iterate_words(raw_input().decode("utf-8"))


def sample_words(wp, sess, limit_word_len, word_iterator):
    original, predicted_max, predicted_rand = "", "", ""
    for word in word_iterator:
        original += word.encode("utf-8")
        word = wp.encode_word(sess, word)
        predicted_max += wp.decode_word_max(sess, word, limit_word_len)
        predicted_rand += wp.decode_word_sample(sess, word, limit_word_len)
    return original, predicted_max, predicted_rand


class TOptimizerSelector:
    def __init__(self, wp, learning_rate):
        self.wp = wp
        self.learning_rate = learning_rate
        self.mults = tf.placeholder(tf.float32, [None])
        self.target_losses, self.optimizers = {}, {}

    def choose(self, word_len):
        encoder_input, decoder_input, decoder_output = self.wp.full_encoder_decoder(word_len)
        if word_len not in self.optimizers:
            ohy = tf.one_hot(encoder_input, self.wp.vocabulary_size, on_value = 1.0)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(decoder_output, ohy)
            loss = tf.reduce_mean(loss, 2)
            loss = tf.reduce_mean(loss, 1)
            loss = tf.mul(loss, self.mults)
            loss = tf.reduce_mean(loss)
            self.target_losses[word_len] = loss
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(loss)
            #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss, colocate_gradients_with_ops = True)
            #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
            self.optimizers[word_len] = optimizer
        return encoder_input, decoder_input, self.mults, self.target_losses[word_len], self.optimizers[word_len]


def correct_learning_rate_multiplier(losses):
    inc, dec = 0, 0
    losses = losses[-10:]
    for i in xrange(1, len(losses)):
        if losses[i - 1] < losses[i]:
            inc += 1
        else:
            dec += 1
    #print inc, dec
    if inc == 0 and dec > 8:
        return 1.05
    elif inc >= 2:
        return 0.95
    return 1.0


def main():
    # define params
    batch_size, long_word_limit, max_word_len, long_batch_size, state_size, limit_word_len = 5000, 20, 30, 100, 1024, 50
    learning_rate = tf.Variable(0.1, trainable=False)

    wp = TWordPackager(state_size, VOCABULARY_SIZE)
    opt = TOptimizerSelector(wp, learning_rate)

    # prepare variables
    sess = tf.Session()

    # read all data
    data = read_words()

    # retrive all optimizers
    for word_len, batch_x, batch_dx, batch_m in iterate_batches(data, len(data), long_word_limit, len(data)):
        print word_len, len(batch_x)
    #    x, dx, y, mults, loss, optimizer = opt.choose(word_len)

    # initialize global variables
    init = tf.global_variables_initializer()
    sess.run(init)

    wp.to_json(sess)

    if False:
        wp.from_json(open("dump.char", "rt").read(), sess)
        print sample_words(wp, sess, max_word_len, some_fixed_text(max_word_len))[1]
        sys.stdout.flush()
    elif False:
        wp.from_json(open("dump.char", "rt").read(), sess)
        while True:
            print sample_words(wp, sess, max_word_len, iterate_keyboard_input(max_word_len))[1]

    epoch, learning_rate_multiplier = 0, 1.0
    all_batches = []
    for word_len, batch_x, batch_dx, batch_m in iterate_batches(data, batch_size, long_word_limit, long_batch_size):
        if word_len <= max_word_len:
            all_batches.append([word_len, batch_x, batch_dx, batch_m])
    losses = []
    while True:
        cnt, l = 1e-38, 0.0
        for word_len, batch_x, batch_dx, batch_m in all_batches:
            cnt += 1
            x, dx, mults, loss, optimizer = opt.choose(word_len)
            #grad = []
            #for gv in optimizer.compute_gradients(loss):
            #    if gv[0] is not None:
            #        grad.append((gv[0] * learning_rate_multiplier, gv[1]))
            _, _l = sess.run([optimizer, loss], feed_dict = {x: batch_x, dx: batch_dx, mults: batch_m})
            l += _l
        losses.append(l / cnt)
        learning_rate_multiplier = correct_learning_rate_multiplier(losses)
        learning_rate *= learning_rate_multiplier
        print "loss: %f\tlearning rate: %.6f\tepoch: %d" % (l / cnt, sess.run(learning_rate), epoch)
        if epoch % 30 == 0:
            original, predicted_max, predicted_rand = sample_words(wp, sess, limit_word_len, some_fixed_text())
            print original
            print predicted_max
            print predicted_rand
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

