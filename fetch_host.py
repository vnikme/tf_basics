#!/usr/bin/python

import base64, json, sys, time, urllib, urlparse
from bs4 import BeautifulSoup


def process_page(data, base_url, addr, queue, visited, path, number):
    time.sleep(5)
    doc = BeautifulSoup(data, "html.parser")
    content = ""
    for pre in doc.find_all("pre"):
        for s in pre.stripped_strings:
            content += (s + "\n")
    result = {}
    result["path"] = addr.path
    result["data"] = content.encode("utf-8")
    print result["path"], len(result["data"]), len(queue)
    open("%s/%d" % (path, number), "wt").write(base64.b64encode(json.dumps(result)))
    for a in doc.find_all("a"):
        if "href" not in a.attrs:
            continue
        url = urlparse.urlparse(a["href"])
        if url.scheme and url.scheme != addr.scheme:
            continue
        if url.netloc and url.netloc != addr.netloc:
            continue
        url = urlparse.urljoin(base_url, url.path)
        if url in visited:
            continue
        queue.append(url)
        visited.add(url)


def main():
    queue = [sys.argv[1]]
    visited = set()
    visited.add(queue[0])
    number = 0
    path = sys.argv[2]
    while len(queue) != 0:
        url, queue = queue[:1] + [queue[1:]]
        fd = urllib.urlopen(url)
        addr = urlparse.urlparse(url)
        process_page(fd.read().decode("windows-1251"), url, addr, queue, visited, path, number)
        number += 1


if __name__ == "__main__":
    main()

