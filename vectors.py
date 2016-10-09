# This creates vectors from documents

import auxiliary
import os
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import Doc2Vec
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import codecs
from random import shuffle
import nltk
import warnings
from sklearn.naive_bayes import GaussianNB


warnings.filterwarnings("ignore")

VECTOR_DIMENSION = 1000
SOURCE_COUNT = 500000
TARGET_COUNT = 500000


def prepare_data(size, domain):
    global SOURCE_COUNT
    global TARGET_COUNT
    if size == 'p_s':
        # If size is small and data is processed
        # Variable abbreviations key: s, t, p, n, u = source, target, positive, negative, unlabeled
        s_p = auxiliary.processed_reviews('data/processed_acl/' + domain + '/positive.review')
        s_n = auxiliary.processed_reviews('data/processed_acl/' + domain + '/negative.review')
        output = s_p + s_n
        shuffle(output)
        shuffle(output)
        #s_u = auxiliary.processed_reviews('data/processed_acl/' + domain + '/unlabeled.review')
        return output

    if size == 'small':
        print 'SMALL'
        small_df_p = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/positive.xml')
        small_df_p.drop(small_df_p.index[0], inplace=True)
        small_df_p = small_df_p.iloc[numpy.random.permutation(len(small_df_p))]

        small_df_n = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/negative.xml')
        small_df_n.drop(small_df_n.index[0], inplace=True)
        small_df_n = small_df_n.iloc[numpy.random.permutation(len(small_df_n))]

        small_df_u = auxiliary.xml_to_pandas('data/sorted_data_acl/kitchen/unlabeled.xml')
        small_df_u.drop(small_df_u.index[0], inplace=True)
        small_df_u = small_df_u.iloc[numpy.random.permutation(len(small_df_u))]

        small_target_p = auxiliary.xml_to_pandas('data/sorted_data_acl/dvd/positive.xml')
        small_target_p.drop(small_target_p.index[0], inplace=True)
        small_target_p = small_target_p.iloc[numpy.random.permutation(len(small_df_p))]

        small_target_n = auxiliary.xml_to_pandas('data/sorted_data_acl/dvd/negative.xml')
        small_target_n.drop(small_target_n.index[0], inplace=True)
        small_target_n = small_target_n.iloc[numpy.random.permutation(len(small_target_n))]

        small_target_u = auxiliary.xml_to_pandas('data/sorted_data_acl/dvd/unlabeled.xml')
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
        large_df_a = auxiliary.xml_to_pandas('data/sorted_data_large/electronics/all.xml')
        large_df_a.drop(large_df_a.index[0], inplace=True)
        large_df_a = large_df_a.iloc[numpy.random.permutation(len(large_df_a))]

        large_target_a = auxiliary.xml_to_pandas('data/sorted_data_large/kitchen/all.xml')
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
        print 'Target Total:', len(huge_target_list), 'Positive: ', len(huge_target_list_p), 'Negative: ', len(huge_target_list_n)
        SOURCE_COUNT = min(len(huge_source_list_p), len(huge_source_list_n))
        TARGET_COUNT = min(len(huge_target_list_p), len(huge_target_list_n))
        print 'Using ', SOURCE_COUNT, 'of each positive and negative samples per domain'
        huge_source_list = huge_source_list_p[:SOURCE_COUNT] + huge_source_list_n[:SOURCE_COUNT]
        huge_target_list = huge_target_list_p[:TARGET_COUNT] + huge_target_list_n[:TARGET_COUNT]

        shuffle(huge_source_list)
        shuffle(huge_target_list)

        return huge_source_list, SOURCE_COUNT, huge_target_list, TARGET_COUNT


def text_to_vector(text=None, algo_type=0, source=None, target=None):
    # Converts preprocessed text to vectors. For choosing the value of algorithm:
    # 0 = Tfidf Vectorizer
    # 1 = Doc2Vec Paragraph Vectors
    # 2 = Doc2Vec Paragraph vectors for processed data
    global VECTOR_DIMENSION
    if algo_type == 2:
        preprocessed_text = auxiliary.add_all_data('s')
    elif algo_type == 3:
        preprocessed_text, labels = [], []
        for x in range(len(text)):
            preprocessed_text.append(text[x][1])
            labels.append(text[x][0])
    else:
        labels, preprocessed_text = preprocess(text)
    if algo_type == 0:
        # TFIDF and feature selection
        print 'Using Tf-idf Vectorizer'
        vectorizer = TfidfVectorizer(analyzer="word", max_features=VECTOR_DIMENSION)
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

        model = Doc2Vec(sentences, alpha=0.025, size=VECTOR_DIMENSION, window=8, min_count=3, workers=4,
                        max_vocab_size=20000)
        for epoch in range(1, 5):
            model.train(sentences)

        train_data_features = numpy.zeros((len(labels), VECTOR_DIMENSION))
        for x in range(1, len(labels)):
            train_data_features[x] = model.docvecs[x]

        return train_data_features, labels

    elif algo_type == 3:
        print 'Using Doc2Vec'
        sentences_file = codecs.open("sources.txt", "w", encoding='utf8')
        for x in preprocessed_text:
            sentences_file.write(' '.join(x) + '\n')
        sentences_file.close()
        sentences = TaggedLineDocument('sources.txt')
        # TODO: delete sources.txt
        # os.remove('sources.txt')

        model = Doc2Vec(sentences, alpha=0.025, size=VECTOR_DIMENSION, window=500, min_count=3, workers=16,
                        dm=0)

        for x in range(1, 2):
            model.train(sentences)

        train_data_features = numpy.zeros((len(labels), VECTOR_DIMENSION))
        for x in range(1, len(labels)):
            train_data_features[x] = model.docvecs[x]

        return train_data_features, labels

    elif algo_type == 2:
        print 'Using Doc2Vec'
        sentences_file = codecs.open("sources.txt", "w", encoding='utf8')
        for x in preprocessed_text:
            sentences_file.write(' '.join(x[1]) + '\n')
        sentences_file.close()
        sentences = TaggedLineDocument('sources.txt')
        # TODO: delete sources.txt
        # os.remove('sources.txt')

        model = Doc2Vec(sentences, alpha=0.025, size=VECTOR_DIMENSION, window=8, min_count=3, workers=4,
                        max_vocab_size=5000)
        #for epoch in range(1, 5):
        #    model.train(sentences)
        #model.save('small.txt')
        model.load('small.txt')
        train_data_features = numpy.zeros((2000, VECTOR_DIMENSION))
        test_data_features = numpy.zeros((2000, VECTOR_DIMENSION))
        train_labels, test_labels = [], []

        for x in range(2000):
            if source == 'books':
                train_data_features[x] = model.docvecs[x]
                train_labels.append(preprocessed_text[x][0])
            elif source == 'dvd':
                train_data_features[x] = model.docvecs[2000 + x]
                train_labels.append(preprocessed_text[2000 + x][0])
            elif source == 'electronics':
                train_data_features[x] = model.docvecs[4000 + x]
                train_labels.append(preprocessed_text[4000 + x][0])
            elif source == 'kitchen':
                train_data_features[x] = model.docvecs[6000 + x]
                train_labels.append(preprocessed_text[6000 + x][0])

        for x in range(2000):
            if target == 'books':
                test_data_features[x] = model.docvecs[x]
                test_labels.append(preprocessed_text[x][0])
            elif target == 'dvd':
                test_data_features[x] = model.docvecs[2000 + x]
                test_labels.append(preprocessed_text[2000 + x][0])
            elif target == 'electronics':
                test_data_features[x] = model.docvecs[4000 + x]
                test_labels.append(preprocessed_text[4000 + x][0])
            elif target == 'kitchen':
                test_data_features[x] = model.docvecs[6000 + x]
                test_labels.append(preprocessed_text[6000 + x][0])

        return train_data_features, train_labels, test_data_features, test_labels


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

    # Frequency Distribution
    fdist = nltk.FreqDist([item for sublist in processed for item in sublist])
    # Returns the x most frequent words with their count
    fdist.tabulate(10)

    for x in range(len(doc)):
        labels.append(auxiliary.rating_to_label(doc[x][0]))

    return labels, clean_train_reviews
