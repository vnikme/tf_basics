#!/usr/bin/python

import fnmatch, os, sys

def iterate_files(path, mask):
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            yield os.path.join(root, filename)


def main():
    for f in iterate_files("lib_ru/public_html/book", "*.txt"):
        try:
            print open(f, "rt").read().decode("koi8-r").encode("utf-8")
        except:
            pass


if __name__ == "__main__":
    main()

