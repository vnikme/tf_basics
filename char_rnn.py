#!/usr/bin/python
# encoding: utf-8


import tensorflow as tf
import numpy as np
import random, sys


all_syms = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")
#all_syms = "0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя \n'\"()?!.,*+-/\%/\\$#@:;".decode("utf-8")
#all_syms = "Helo".decode("utf-8")


# make lower case
def to_wide_lower(s):
    s = s.decode("utf-8")
    s = s.lower()
    return s


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
            if c / 1000000 >= 30:
                break
        #line = to_wide_lower(line)
        line = line.decode("utf-8")
        for ch in line:
            if ch not in all_syms:
                continue
            yield all_syms.index(ch)


# input shape: batch*time*state
# output shape: batch*time*vocabulary
# multiplies last dimention by `w` and adds `b`
def make_projection(x, state_size, max_time, vocabulary_size, w, b):
    output = tf.reshape(x, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    output = tf.nn.softmax(output)
    return output


def choose_random(distr):
    #print " ".join(map(lambda t: "%.2f" % t, distr))
    cs = np.cumsum(distr)
    s = np.sum(distr)
    return int(np.searchsorted(cs, np.random.rand(1) * s))


def make_sample(sess, x, state_x, op, state_op, cur_state, n, seed):
    seed = [all_syms.index(ch) for ch in seed.decode("utf-8")]
    for ch in seed:
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[ch]], state_x: cur_state})
    res = res[0][0]
    #print res
    cur_sym = choose_random(res)
    result = all_syms[cur_sym]
    for k in xrange(n):
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[cur_sym]], state_x: cur_state})
        res = res[0][0]
        cur_sym = choose_random(res)
        result += all_syms[cur_sym]
    return result.encode("utf-8")


def make_sample1(sess, x, state_x, op, state_op, l, cur_state, n, seed, max_time):
    result = seed.decode("utf-8")
    seed = [all_syms.index(ch) for ch in seed.decode("utf-8")]
    for ch in seed:
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[ch] * max_time], state_x: cur_state, l: [1]})
    #print cur_state.shape
    res = res[0][0]
    #print "\t".join(map(lambda u: "%.2f" % u, res))
    cur_sym = choose_random(res)
    result += all_syms[cur_sym]
    for k in xrange(n):
        res, cur_state = sess.run([op, state_op], feed_dict = {x: [[cur_sym] * max_time], state_x: cur_state, l: [1]})
        res = res[0][0]
        #print "\t".join(map(lambda u: "%.2f" % u, res))
        cur_sym = choose_random(res)
        result += all_syms[cur_sym]
    return result.encode("utf-8")


def print_matr(matr):
    for vec in matr:
        print "\t".join(map(lambda u: "%.5f" % u, vec))


def print_matrs(matrs):
    for matr in matrs:
        print_matr(matr)
        print


def test_projection():
    max_time, state_size, vocabulary_size = 2, 3, 2
    x = tf.placeholder(tf.float32, [None, max_time, state_size])
    w = tf.Variable([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
    b = tf.Variable([1.0, 1.0])
    y = make_projection(x, state_size, max_time, vocabulary_size, w, b)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print_matrs(sess.run(y, feed_dict = {x: [[[1, 2, 3], [1, 1, 1]]]}))


def test_onehot():
    y = tf.placeholder(tf.int32, [None, 3])
    ohy = tf.one_hot(y, 5, on_value = 1.0)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print_matrs(sess.run(ohy, feed_dict = {y: [[1, 2, 3], [0, 1, 2], [2, 3, 4]]}))


def test():
    test_projection()
    test_onehot()

 
# do all stuff
def main():
    # define params
    max_time, batch_size, state_size, learning_rate = 16, 5000, 1024, 0.001
    #max_time, batch_size, state_size, learning_rate = 16, 1, 512, 0.001
    #learning_rate /= batch_size
    path = "data/all"
    #path = "char_rnn.py"
    vocabulary_size = len(all_syms)

    # read and convert data
    data = np.asarray(list(iterate_symbols(path)))

    # create variables and graph
    x = tf.placeholder(tf.int32, [None, max_time])
    lengths = tf.placeholder(tf.int32, [None])
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    w = tf.Variable(tf.random_normal([state_size, vocabulary_size]))
    b = tf.Variable(tf.random_normal([vocabulary_size]))

    # create learning graph
    state_x = tf.placeholder(tf.float32, [None, state_size])
    with tf.variable_scope('train'):
        output, state = tf.nn.dynamic_rnn(gru, tf.one_hot(x, vocabulary_size, on_value = 1.0), initial_state = state_x, sequence_length = lengths, dtype = tf.float32, swap_memory = True)
    output = tf.reshape(output, [-1, state_size])
    output = tf.add(tf.matmul(output, w), b)
    output = tf.reshape(output, [-1, max_time, vocabulary_size])
    #output = tf.nn.softmax(output)
    y = tf.placeholder(tf.int32, [None, max_time])

    # define loss and optimizer
    ohy = tf.one_hot(y, vocabulary_size, on_value = 1.0)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, ohy))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    # create output tensors for applying
    apply_x = tf.placeholder(tf.int32, [None, max_time])                    # we will apply it sym-by-sym
    apply_state_x = tf.placeholder(tf.float32, [None, state_size])
    with tf.variable_scope('train', reuse = True):
        apply_output, apply_state = tf.nn.dynamic_rnn(gru, tf.one_hot(apply_x, vocabulary_size, on_value = 1.0), initial_state = apply_state_x, sequence_length = lengths, dtype = tf.float32, swap_memory = True)
    apply_output = tf.reshape(apply_output, [-1, state_size])
    apply_output = tf.add(tf.matmul(apply_output, w), b)
    apply_output = tf.reshape(apply_output, [-1, max_time, vocabulary_size])
    apply_output = tf.nn.softmax(apply_output)

    # create saver
    saver = tf.train.Saver(max_to_keep = 20)

    # prepare variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cnt = 0
    steps = (len(data) - 1) / (batch_size * max_time)
    zero_state = sess.run(gru.zero_state(batch_size, tf.float32))
    apply_zero_state = sess.run(gru.zero_state(1, tf.float32))
    prev_loss = 0.0

    # apply mode
    if len(sys.argv) > 1:
        saver.restore(sess, sys.argv[1])
        while True:
            seed = raw_input("Enter phrase: ")
            print make_sample1(sess, apply_x, apply_state_x, apply_output, apply_state, lengths, apply_zero_state, int(sys.argv[2]), seed, max_time)

    # training mode
    while True:
        l, c, i = 0.0, 0, 0
        cur_state = zero_state
        for i in xrange(steps):
            #print (i + 1) * max_time + steps * max_time * (batch_size - 1) + 1, len(data), steps
            batch_x, batch_y = [], []
            for j in xrange(batch_size):
                batch_x.append(data[i * max_time + j * steps * max_time : (i + 1) * max_time + j * steps * max_time])
                batch_y.append(data[i * max_time + j * steps * max_time + 1 : (i + 1) * max_time + j * steps * max_time + 1])
            #print
            #print batch_x
            #print batch_y
            #print
            if c == 0 or l / c > 0.03:
                cur_state = zero_state
            _, cur_state, _l = sess.run([optimizer, state, loss], feed_dict = {x: batch_x, y: batch_y, state_x: cur_state, lengths: [max_time] * batch_size})
            #_l /= (batch_size * max_time)
            l += _l
            c += 1
            #print_matrs(dump)
            print "Progress: %.1f%%, loss: %f" % (i * 100.0 / steps, _l)
            sys.stdout.flush()
            #if _l < 0.3:
            #    break
        seed = "Посадил дед ре"
        #seed = "#!/"
        print make_sample1(sess, apply_x, apply_state_x, apply_output, apply_state, lengths, apply_zero_state, 1000, seed, max_time)
        print "Loss: %.5f\tdiff with prev: %.5f\tlrate: %.5f\n" % (l / c, l / c - prev_loss, learning_rate)
        sys.stdout.flush()
        saver.save(sess, "dumps/dump", global_step = cnt)
        cnt += 1
        #if -l / c - prev_loss < 0.00001:
        #    learning_rate *= 0.95
        prev_loss = l / c


# entry point
if __name__ == "__main__":
    #test()
    main()

