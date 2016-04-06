# Helper File
# This file contains functions that help extract data from source and transform it into usable data frame

import pandas
import lxml.etree
import gzip


def xml_to_pandas(xml_file):
    # TODO: add a check for root existence
    # TODO: fix review vs xml issue
    # xml_file = add_root(xml_file)
    review_list = [["rating", "review_text"]]
    parser = lxml.etree.XMLParser(encoding='utf-8', recover=True)
    tree = lxml.etree.parse(xml_file, parser=parser)
    root = tree.getroot()
    elements = 0
    for x in root.iter():
        if x.tag == "review":
            for y in x.iter():
                if y.tag == "rating":
                    a = y.text.replace('\n', '')
                if y.tag == "review_text":
                    b = y.text.replace('\n', '')
            elements += 1
            review_list.append([a, b])
    df = pandas.DataFrame(review_list)
    return df, elements


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
    if item > 3.0:
        return 1.0
    elif item <= 3.0:
        return 0.0


def model_to_array():
    pass


def add_root(xml_file):
    with open(xml_file, "r+w") as f:
        temp = f.read()
        f.seek(0, 0)
        f.write("<root>")
        f.write(temp)
        f.write("</root>")
        f.close()
    return xml_file

