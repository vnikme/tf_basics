#!/usr/bin/python
# encoding: utf-8


import base64, fnmatch, json, os, random, sys


letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".decode("utf-8")
digits = "0123456789".decode("utf-8")
sentbrk = "?!.".decode("utf-8")
spaces = " \n\t\r".decode("utf-8")
wordbrk = "|^{}[]©<>&`~'\"(),*+-_=/\%/\\$#@:;".decode("utf-8") + sentbrk + spaces
all_syms = letters + digits + wordbrk


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
                if ch in all_syms:
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


def iterate_words(path, mask):
    is_space, word = True, ""
    for ch in iterate_chars(path, mask):
        if ch in wordbrk:
            if word:
                yield word
                word = ""
            if ch in spaces:
                ch = " "
                if not is_space:
                    yield ch
                    is_space = True
            else:
                is_space = False
                yield ch
        else:
            if ch in letters + digits:
                is_space = False
                word += ch
    if word:
        yield word


def process_words():
    cnt, words, lens, bad = 0, set(), dict(), set()
    for word in iterate_words("lib_ru/public_html/book", "*.txt"):
        l = len(word)
        if l >= 26:
            if word not in bad:
                print word.encode("utf-8")
                bad.add(word)
            continue
        cnt += 1
        words.add(word)
        lens[l] = lens.get(l, 0) + 1
        if cnt % 1000000 == 0:
            print cnt, len(words), len(bad)
    for l in sorted(lens.keys()):
        print l, lens[l]


def iterate_sentences(path, mask):
    sent = []
    for word in iterate_words(path, mask):
        sent.append(word)
        if word in sentbrk:
            yield sent
            sent = []
    if sent:
        yield sent


def avg(d):
    s, c = 0.0, 1e-38
    for k, v in d:
        s += (k * v)
        c += v
    return s / c


def med(d):
    d = sorted(d, key = lambda x: -x[1])
    c, s = 0, 0
    for k, v in d:
        s += v
    for k, v in d:
        c += v
        if c >= s / 2:
            return k
    return d[-1][0]


def mod(d):
    d = sorted(d, key = lambda x: x[0])
    m = 4
    for i in xrange(5, len(d)):
        if d[i][1] > d[m][1]:
            m = i
    return d[m][0]


def print_distr_analysis(m):
    d = []
    for k, v in m.iteritems():
        d.append((k, v))
    print avg(d), med(d), mod(d)


def process_sentences():
    cnt, m = 0, dict()
    for sent in iterate_sentences("lib_ru/public_html/book", "*.txt"):
        cnt += 1
        l = len(sent)
        #if l > 100:
        #    print l, "".join([w.encode("utf-8") for w in sent]) + "\n"
        m[l] = m.get(l, 0) + 1
        if cnt % 100000 == 0:
            print_distr_analysis(m)


def main():
    #process_words()
    process_sentences()


# entry point
if __name__ == "__main__":
    main()

