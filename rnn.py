#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import numpy as np
from sklearn import metrics
import auxiliary
import algorithms
import vectors
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow

BIG_SOURCE_COUNT = 1000
BIG_TARGET_COUNT = 500
'''
train = auxiliary.json_to_pandas('data/reviews_Books.json.gz', BIG_SOURCE_COUNT)
train.overall = train.overall.replace(3, 1)
train.overall = train.overall.replace(2, 1)
train.overall = train.overall.replace(1, 1)
train.overall = train.overall.replace(5, 0)
train.overall = train.overall.replace(4, 0)
X_train, y_train = train['reviewText'], train['overall']
test = auxiliary.json_to_pandas('data/reviews_Electronics.json.gz', BIG_TARGET_COUNT)
test.overall = test.overall.replace(3, 1)
test.overall = test.overall.replace(2, 1)
test.overall = test.overall.replace(1, 1)
test.overall = test.overall.replace(5, 0)
test.overall = test.overall.replace(4, 0)
X_test, y_test = test['reviewText'], test['overall']
'''

X_train, y_train = vectors.text_to_vector(auxiliary.json_to_pandas('data/reviews_Books.json.gz'))
X_test, y_test = vectors.text_to_vector(auxiliary.json_to_pandas('data/reviews_Electronics.json.gz'))
### Process vocabulary

MAX_DOCUMENT_LENGTH = 5

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models
EMBEDDING_SIZE = 50


def average_model(X, y):
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    features = tf.reduce_max(word_vectors, reduction_indices=1)
    return skflow.models.logistic_regression(features, y)


def rnn_model(X, y):
    """Recurrent neural network model to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    return skflow.models.logistic_regression(encoding, y)


def input_op_fn(X):
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    return word_list


def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=0.1, global_step=global_step,
        decay_steps=100, decay_rate=0.001)

classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=2, steps=100, optimizer='Adam',
                                        learning_rate=exp_decay, continue_training=True)
'''
classifier = skflow.TensorFlowRNNClassifier(rnn_size=EMBEDDING_SIZE,
    n_classes=2, cell_type='gru', input_op_fn=input_op_fn,
    num_layers=1, bidirectional=True, sequence_length=vectors.VECTOR_DIMENSION,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)
'''
#print X_train
#X_train = algorithms.coral(X_train, X_test)
#print X_train
while True:
    try:
        classifier.fit(X_train, y_train, logdir='/tmp/tf_examples/my_model_1/')
    except KeyboardInterrupt:
        #classifier.save(model_path)
        break

score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))