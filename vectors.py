# This creates vectors from documents

import auxiliary
import os
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import codecs
from random import shuffle
import nltk
import warnings


warnings.filterwarnings("ignore")

VECTOR_DIMENSION = 2000
SOURCE_COUNT = 0
TARGET_COUNT = 0


def prepare_data(size):
    global SOURCE_COUNT
    global TARGET_COUNT
    if size == 'small':
        print 'SMALL'
        small_df_p = auxiliary.xml_to_pandas('data/sorted_data_acl/electronics/positive.xml')
        small_df_p.drop(small_df_p.index[0], inplace=True)
        small_df_p = small_df_p.iloc[numpy.random.permutation(len(small_df_p))]

        small_df_n = auxiliary.xml_to_pandas('data/sorted_data_acl/electronics/negative.xml')
        small_df_n.drop(small_df_n.index[0], inplace=True)
        small_df_n = small_df_n.iloc[numpy.random.permutation(len(small_df_n))]

        small_df_u = auxiliary.xml_to_pandas('data/sorted_data_acl/electronics/unlabeled.xml')
        small_df_u.drop(small_df_u.index[0], inplace=True)
        small_df_u = small_df_u.iloc[numpy.random.permutation(len(small_df_u))]

        small_target_p = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/positive.xml')
        small_target_p.drop(small_target_p.index[0], inplace=True)
        small_target_p = small_target_p.iloc[numpy.random.permutation(len(small_df_p))]

        small_target_n = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/negative.xml')
        small_target_n.drop(small_target_n.index[0], inplace=True)
        small_target_n = small_target_n.iloc[numpy.random.permutation(len(small_target_n))]

        small_target_u = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/unlabeled.xml')
        small_target_u.drop(small_target_u.index[0], inplace=True)
        small_target_u = small_target_u.iloc[numpy.random.permutation(len(small_target_u))]

        small_list_p = small_df_p.values.tolist()
        small_list_n = small_df_n.values.tolist()
        small_list_u = small_df_u.values.tolist()

        for x in small_list_u:
            if float(x[0]) < 3.0:
                small_list_n.append(x)
            elif float(x[0]) > 3.0:
                small_list_p.append(x)

        small_target_list_p = small_target_p.values.tolist()
        small_target_list_n = small_target_n.values.tolist()
        small_target_list_u = small_target_u.values.tolist()

        for x in small_target_list_u:
            if float(x[0]) < 3.0:
                small_target_list_n.append(x)
            elif float(x[0]) > 3.0:
                small_target_list_p.append(x)

        print 'Source Total:', len(small_list_p)+len(small_list_n), 'Positive:', len(small_list_p),\
            'Negative:', len(small_list_n)
        print 'Target Total:', len(small_target_list_p)+len(small_target_list_n), 'Positive:',\
            len(small_target_list_p), 'Negative:', len(small_target_list_n)

        SOURCE_COUNT = min(len(small_list_p), len(small_list_n))
        TARGET_COUNT = min(len(small_target_list_p), len(small_target_list_n))
        print 'Using ', SOURCE_COUNT, 'of each positive and negative samples per domain'
        small_source_list = small_list_p[:SOURCE_COUNT] + small_list_n[:SOURCE_COUNT]
        small_target_list = small_target_list_p[:TARGET_COUNT] + small_target_list_n[:TARGET_COUNT]
        shuffle(small_source_list)
        shuffle(small_target_list)

        return small_source_list, SOURCE_COUNT, small_target_list, TARGET_COUNT

    elif size == 'large':
        print 'LARGE'
        large_df_a = auxiliary.xml_to_pandas('data/sorted_data_large/dvd/all.xml')
        large_df_a.drop(large_df_a.index[0], inplace=True)
        large_df_a = large_df_a.iloc[numpy.random.permutation(len(large_df_a))]

        large_target_a = auxiliary.xml_to_pandas('data/sorted_data_large/books/all.xml')
        large_target_a.drop(large_target_a.index[0], inplace=True)
        large_target_a = large_target_a.iloc[numpy.random.permutation(len(large_target_a))]

        large_list_p = []
        large_list_n = []
        large_list_a = large_df_a.values.tolist()

        for x in large_list_a:
            if float(x[0]) < float(3.0):
                large_list_n.append(x)
            elif float(x[0]) > float(3.0):
                large_list_p.append(x)

        large_target_list_p = []
        large_target_list_n = []
        large_target_list_a = large_target_a.values.tolist()

        for x in large_target_list_a:
            if float(x[0]) < 3.0:
                large_target_list_n.append(x)
            elif float(x[0]) > 3.0:
                large_target_list_p.append(x)

        print 'Source Total:', len(large_list_a), 'Positive: ', len(large_list_p), 'Negative: ', len(large_list_n)
        print 'Target Total:', len(large_target_list_a), 'Positive: ', len(large_target_list_p), 'Negative: ', len(large_target_list_n)
        SOURCE_COUNT = min(len(large_list_p), len(large_list_n))
        TARGET_COUNT = min(len(large_target_list_p), len(large_target_list_n))
        print 'Using ', SOURCE_COUNT, 'of each positive and negative samples per domain'
        large_source_list = large_list_p[:SOURCE_COUNT] + large_list_n[:SOURCE_COUNT]
        large_target_list = large_target_list_p[:TARGET_COUNT] + large_target_list_n[:TARGET_COUNT]
        shuffle(large_source_list)
        shuffle(large_target_list)

        return large_source_list, SOURCE_COUNT, large_target_list, TARGET_COUNT
    '''
    elif size == 'huge':
        # Books
        # Electronics
        # Kitchen
        # CDs_and_Vinyl
        huge_source_data = auxiliary.json_to_pandas('data/reviews_Electronics.json.gz', SOURCE_COUNT)
        huge_target_data = auxiliary.json_to_pandas('data/reviews_Kitchen.json.gz', TARGET_COUNT)

        huge_source_list_p = []
        huge_source_list_n = []
        huge_source_list = huge_source_data.values.tolist()
        huge_source_list.pop(0)

        for x in huge_source_list:
            if float(x[0]) < float(3.0):
                huge_source_list_n.append(x)
            elif float(x[0]) > float(3.0):
                huge_source_list_p.append(x)

        huge_target_list_p = []
        huge_target_list_n = []
        huge_target_list = huge_target_data.values.tolist()
        huge_target_list.pop(0)

        for x in huge_target_list:
            if float(x[0]) < 3.0:
                huge_target_list_n.append(x)
            elif float(x[0]) > 3.0:
                huge_target_list_p.append(x)

        print 'Source Total:', len(huge_source_list), 'Positive: ', len(huge_source_list_p), 'Negative: ', len(huge_source_list_n)
        print 'Target Total:', len(huge_target_list), 'Positive: ', len(huge_target_list_p), 'Negative: ', len(
            huge_target_list_n)
        SOURCE_COUNT = min(len(), len(huge_list_n))
        HUGE_TARGET_COUNT = min(len(huge_target_list_p), len(huge_target_list_n))
        print 'Using ', HUGE_SOURCE_COUNT / 2, 'of each positive and negative samples per domain'
        huge_source_list = huge_list_p[:HUGE_SOURCE_COUNT / 2] + huge_list_n[:HUGE_SOURCE_COUNT / 2]
        huge_target_list = huge_target_list_p[:HUGE_TARGET_COUNT / 2] + huge_target_list_n[:HUGE_TARGET_COUNT / 2]

        shuffle(huge_source_list)
        shuffle(huge_target_list)

        return huge_source_list, HUGE_SOURCE_COUNT, huge_target_list, HUGE_TARGET_COUNT
    '''


def text_to_vector(text, algo_type=0):
    # Converts preprocessed text to vectors. For choosing the value of algorithm:
    # 0 = Tfid Vectorizer
    # 1 = Doc2Vec Paragraph Vectors
    labels, preprocessed_text = preprocess(text)
    if algo_type == 0:
        print 'Using Tf-idf Vectorizer'
        vectorizer = TfidfVectorizer(analyzer="word", max_features=2000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data_features = vectorizer.fit_transform(preprocessed_text)
        train_data_features = train_data_features.toarray()

        return train_data_features, labels

    elif algo_type == 1:
        print 'Using Doc2Vec'
        sentences_file = codecs.open("sources.txt", "w", encoding='utf8')
        for x in preprocessed_text:
            sentences_file.write(x + '\n')
        sentences_file.close()
        sentences = TaggedLineDocument('sources.txt')
        # TODO: delete sources.txt
        # os.remove('sources.txt')

        model = Doc2Vec(sentences, alpha=0.025, size=VECTOR_DIMENSION, window=8, min_count=3, workers=4, max_vocab_size=500)
        for epoch in range(1, 5):
            model.train(sentences)

            train_data_features = numpy.zeros((len(labels), VECTOR_DIMENSION))
        for x in range(1, len(labels)):
            train_data_features[x] = model.docvecs[x]

        return train_data_features, labels


def preprocess(doc):
    text, processed, labels = [], [], []
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-z]+")
    stemmer = nltk.stem.PorterStemmer()

    for x in range(len(doc)):
        text.append(doc[x][1])

    for x in range(len(text)):
        text[x].lower()

    for x in text:
        tokens = []
        for token in tokenizer.tokenize(x):
            if token not in stopwords:
                tokens.append(stemmer.stem(token))
        processed.append(tokens)

    clean_train_reviews = []
    for x in processed:
        clean_train_reviews.append(" ".join(x))

    fdist = nltk.FreqDist([item for sublist in processed for item in sublist])
    fdist.tabulate(10)

    for x in range(len(doc)):
        labels.append(auxiliary.rating_to_label(doc[x][0]))

    return labels, clean_train_reviews
