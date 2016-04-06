# This creates vectors from documents

import auxiliary
import os
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import Doc2Vec
import numpy
import codecs
from random import shuffle


VECTOR_DIMENSION = 500
BIG_SOURCE_COUNT = 1000
BIG_TARGET_COUNT = 1000


def prepare_data(size):
    if size == 'small':
        small_df_p, small_count_p = auxiliary.xml_to_pandas('data/sorted_data_acl/books/positive.xml')
        small_df_n, small_count_n = auxiliary.xml_to_pandas('data/sorted_data_acl/books/negative.xml')
        small_df_u, small_count_u = auxiliary.xml_to_pandas('data/sorted_data_acl/books/unlabeled.xml')
        small_target_p, small_count_target_p = auxiliary.xml_to_pandas('data/sorted_data_acl/books/positive.xml')
        small_target_n, small_count_target_n = auxiliary.xml_to_pandas('data/sorted_data_acl/books/negative.xml')
        small_target_u, small_count_target_u = auxiliary.xml_to_pandas('data/sorted_data_acl/books/unlabeled.xml')
        small_list_p = small_df_p.values.tolist()
        small_list_p.pop(0)  # popping headers
        small_list_n = small_df_n.values.tolist()
        small_list_n.pop(0)  # popping headers
        small_list_u = small_df_u.values.tolist()
        small_list_u.pop(0)  # popping headers
        small_target_list_p = small_target_p.values.tolist()
        small_target_list_p.pop(0)
        small_target_list_n = small_target_n.values.tolist()
        small_target_list_n.pop(0)
        small_target_list_u = small_target_u.values.tolist()
        small_target_list_u.pop(0)
        #small_list_u = []
        #small_count_u = 0
        #small_target_list_p = []
        #small_count_target_p = 0
        #small_target_list_n = []
        #small_count_target_n = 0
        #small_target_list_u = []
        #small_count_target_u = 0
        small_source_list = small_list_p + small_list_n + small_list_u
        small_source_count = small_count_p + small_count_n + small_count_u
        small_target_list = small_target_list_p + small_target_list_n + small_target_list_u
        small_target_count = small_count_target_p + small_count_target_n + small_count_target_u
        shuffle(small_source_list)
        shuffle(small_target_list)

        return small_source_list, small_source_count, small_target_list, small_target_count

    elif size == 'big':
        # Books
        # Electronics
        # Home_and_Kitchen
        # CDs_and_Vinyl
        big_source_data = auxiliary.json_to_pandas('data/reviews_Books.json.gz', BIG_SOURCE_COUNT)
        big_target_data = auxiliary.json_to_pandas('data/reviews_Electronics.json.gz', BIG_TARGET_COUNT)
        big_source_list = big_source_data.values.tolist()
        big_source_list.pop(0)
        big_target_list = big_target_data.values.tolist()
        big_target_list.pop(0)
        shuffle(big_source_list)
        shuffle(big_target_list)

        return big_source_list, BIG_SOURCE_COUNT, big_target_list, BIG_TARGET_COUNT


def text_to_vector(text, count):
    sentences_file = codecs.open("sources.txt", "w", encoding='utf8')
    for x in text:
        sentences_file.write(x[1] + '\n')
    sentences_file.close()
    sentences = TaggedLineDocument('sources.txt')
    #TODO: delete sources.txt
    #os.remove('sources.txt')
    model = Doc2Vec(sentences, alpha=0.03, size=VECTOR_DIMENSION, window=8, min_count=3, workers=4, max_vocab_size=500)
    for epoch in range(1, 5):
        model.train(sentences)
    train = numpy.zeros((count, VECTOR_DIMENSION))
    train_labels = numpy.zeros(count)
    for x in range(1, count):
        train[x] = model.docvecs[x]
        train_labels[x] = auxiliary.rating_to_label(float(text[x][0]))
    return train, train_labels
