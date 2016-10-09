# Helper File
# This file contains functions that help extract data from source and transform it into usable data frame

import pandas
import lxml.etree
import gzip
import numpy as np


def xml_to_pandas(xml_file):
    # TODO: add a check for root existence
    # TODO: fix review vs xml issue
    #xml_file = add_root(xml_file)
    with open(xml_file, 'r') as f:
        newlines = []
        for line in f.readlines():
            newlines.append(line)
    with open(xml_file, 'w') as f:
        for line in newlines:
            f.write(line.decode('utf-8', 'ignore').encode("utf-8"))

    review_list = [["rating", "review_text"]]
    parser = lxml.etree.XMLParser(encoding='utf-8', recover=True)
    tree = lxml.etree.parse(xml_file, parser=parser)
    root = tree.getroot()
    element_count = 0
    for x in root.iter():
        if x.tag == "review":
            for y in x.iter():
                if y.tag == "rating":
                    a = y.text.replace('\n', '')
                if y.tag == "review_text":
                    b = y.text.replace('\n', '')
                    element_count += 1
            review_list.append([a, b])
    df = pandas.DataFrame(review_list)
    return df


def processed_reviews(filename):
    output = []
    with open(filename) as f:
        for line in f:
            split_line = line.split()
            extra_words = []
            for item in range(len(split_line)-1):
                count_str = ''
                while split_line[item][-1] != ':':
                    count_str = split_line[item][-1].decode('utf-8') + count_str
                    split_line[item] = split_line[item][:-1].decode('utf-8')
                split_line[item] = split_line[item][:-1]
                for x in range(int(count_str)):
                    extra_words.append(split_line[item])
            if split_line[-1] == '#label#:negative':
                split_line[-1] = 0.0
            elif split_line[-1] == '#label#:positive':
                split_line[-1] = 1.0
            output.append([split_line[-1], split_line[:-1]+extra_words])
    return output


def add_all_data(size):
    output = []
    if size == 's':
        output = processed_reviews('data/processed_acl/books/positive.review') + \
                 processed_reviews('data/processed_acl/books/negative.review') + \
                 processed_reviews('data/processed_acl/dvd/positive.review') + \
                 processed_reviews('data/processed_acl/dvd/negative.review') + \
                 processed_reviews('data/processed_acl/electronics/positive.review') + \
                 processed_reviews('data/processed_acl/electronics/negative.review') + \
                 processed_reviews('data/processed_acl/kitchen/positive.review') + \
                 processed_reviews('data/processed_acl/kitchen/negative.review') + \
                 processed_reviews('data/processed_acl/books/unlabeled.review') + \
                 processed_reviews('data/processed_acl/dvd/unlabeled.review') + \
                 processed_reviews('data/processed_acl/electronics/unlabeled.review') + \
                 processed_reviews('data/processed_acl/kitchen/unlabeled.review')
    return output


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def json_to_pandas(json_file, limit):
    i = 0
    df = {}
    for d in parse(json_file):
        df[i] = d
        i += 1
        if i > limit:
            break
    pandas_data = pandas.DataFrame.from_dict(df, orient='index')
    return pandas_data[['overall', 'reviewText']]


def rating_to_label(item):
    item = float(item)
    if item > 3.0:
        return 1.0
    elif item < 3.0:
        return 0.0


def calc_avg(A):
    return sum(A) / float(len(A))


def add_root(xml_file):
    with open(xml_file, "r+w") as f:
        temp = f.read()
        f.seek(0, 0)
        f.write("<root>")
        f.write(temp)
        f.write("</root>")
        f.close()
    return xml_file


def calc_transfer_loss(x, y):
    transfer_error = 1 - x
    baseline_indomain_error = 1 - y
    transfer_loss = transfer_error - baseline_indomain_error
    return transfer_loss