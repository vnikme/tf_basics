#!/usr/bin/python
# encoding: utf-8


import numpy as np
import tensorflow as tf
import base64, fnmatch, json, math, os, random, sys


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
        if src_word not in data:
            word, dword, target = make_targets(src_word, max_word_len)
            data[src_word] = TWord(word, dword, target)
        else:
            data[src_word].count += 1
        #if len(data) >= 100000:
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


def make_sample(sess, encoder_x, encoder_output, decoder_x, decoder_state_x, decoder_op, decoder_state_op, seed, max_word_len):
    word, _, __ = make_targets(seed, max_word_len)
    cur_state = sess.run(encoder_output, feed_dict = {encoder_x: [word]})[0]
    cur_sym = GO
    result = ""
    while True:
        probs, cur_state = sess.run([decoder_op, decoder_state_op], feed_dict = {decoder_x: [[cur_sym]], decoder_state_x: [cur_state]})
        probs, cur_state = probs[0][0], cur_state[0]
        cur_sym = choose_random(probs)
        if cur_sym == STOP or len(result) >= max_word_len:
            break
        if cur_sym == GO:
            continue
        result += ALL_SYMS[cur_sym].encode("utf-8")
    return result


def sample_text(max_word_len):
    text = "При этом экс-губернатор крайне разозлился тем, что в Интернете активно распространяются фотографии, где он выглядит, как бомж, прибывший на знаковое политическое событие в Соединенные Штаты. Так, на одном из снимков видно, что Саакашвили в непрезентабельной одежде стоит в зале аэропорта Нью-Йорка. Создается впечатление, что утром он был сильно пьян и поэтому надел на себя первое, что попалось ему под руку. На другом снимке Саакашвили одиноко стоит в кустах с телефоном в руке в то время, когда основная публика находится на инаугурации Трампа."
    return iterate_words(text.decode("utf-8"), max_word_len)


def to_json(sess):
    m = {}
    for var in tf.global_variables():
        m[var.name] = var.eval(sess).tolist()
    return json.dumps(m)


def from_json(js):
    m = json.loads(js)
    for var in tf.global_variables():
        var.assign(m[var.name])


def main():
    # define params
    max_word_len, batch_size, encoder_state_size, decoder_state_size, learning_rate = 25, 10000, 64, 256, 0.0001

    # create variables and graph
    encoder_x = tf.placeholder(tf.int32, [None, max_word_len])
    encoder_cell = tf.nn.rnn_cell.GRUCell(encoder_state_size)
    encoder_w = tf.Variable(tf.random_normal([max_word_len * encoder_state_size, decoder_state_size]))
    encoder_b = tf.Variable(tf.random_normal([decoder_state_size]))

    decoder_x = tf.placeholder(tf.int32, [None, max_word_len + 1])
    apply_decoder_x = tf.placeholder(tf.int32, [None, 1])
    decoder_cell = tf.nn.rnn_cell.GRUCell(decoder_state_size)
    decoder_w = tf.Variable(tf.random_normal([decoder_state_size, VOCABULARY_SIZE]))
    decoder_b = tf.Variable(tf.random_normal([VOCABULARY_SIZE]))

    # create learning graph
    decoder_state_placeholder = tf.placeholder(tf.float32, [None, decoder_state_size])
    with tf.variable_scope('encoder'):
        output, state = tf.nn.dynamic_rnn(encoder_cell, tf.one_hot(encoder_x, VOCABULARY_SIZE, on_value = 1.0), dtype = tf.float32)
    encoder_output = attention(output, max_word_len, encoder_state_size, decoder_state_size, encoder_w, encoder_b)
    with tf.variable_scope('decoder'):
        output, state = tf.nn.dynamic_rnn(decoder_cell, tf.one_hot(decoder_x, VOCABULARY_SIZE, on_value = 1.0), initial_state = encoder_output, dtype = tf.float32)
    decoder_output = projection(output, decoder_state_size, max_word_len + 1, VOCABULARY_SIZE, decoder_w, decoder_b)
    with tf.variable_scope('decoder', reuse = True):
        output, apply_decoder_state = tf.nn.dynamic_rnn(decoder_cell, tf.one_hot(apply_decoder_x, VOCABULARY_SIZE, on_value = 1.0), initial_state = decoder_state_placeholder, dtype = tf.float32)
    apply_decoder_output = projection(output, decoder_state_size, 1, VOCABULARY_SIZE, decoder_w, decoder_b)
    mults = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.int32, [None, max_word_len + 1])

    # define loss and optimizer
    ohy = tf.one_hot(y, VOCABULARY_SIZE, on_value = 1.0)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(decoder_output, ohy)
    loss = tf.reduce_mean(loss, 2)
    loss = tf.reduce_mean(loss, 1)
    loss = tf.mul(loss, mults)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    # renorm output logits for sampling
    apply_decoder_output = tf.nn.softmax(apply_decoder_output)

    # prepare variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    data = read_words(max_word_len)

    saver = tf.train.Saver(max_to_keep = 100)

    epoch = 0
    while True:
        cnt, l = 1e-38, 0.0
        for batch_x, batch_dx, batch_y, batch_m in iterate_batches(data, batch_size):
            cnt += 1
            _, _l = sess.run([optimizer, loss], feed_dict = {encoder_x: batch_x, decoder_x: batch_dx, y: batch_y, mults: batch_m})
            l += _l
        print "loss: %f\tepoch: %d" % (l / cnt, epoch)
        text = ""
        for word in sample_text(max_word_len):
            predicted = make_sample(sess, encoder_x, encoder_output, apply_decoder_x, decoder_state_placeholder, apply_decoder_output, apply_decoder_state, word, max_word_len)
            text += predicted
        #print text
        from_json(to_json(sess))
        print
        sys.stdout.flush()
        epoch += 1
        saver.save(sess, "words/char")



# entry point
if __name__ == "__main__":
    main()

