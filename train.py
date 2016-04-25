import algorithms
import visualizations
import auxiliary
import vectors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation as cv
from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer
import skflow
import numpy as np


source_train, source_count, target_test, target_count = vectors.prepare_data('small')
print 'Source: ' + str(source_count) + ', Target: ' + str(target_count)
print 'Converting text to vectors...'
# Combining source and target data together for training
raw_data = source_train + target_test
# text_to_vector returns the labels and vectors
data, labels = vectors.text_to_vector(raw_data, 0)
# source_data is source_count number of positive reviews and the same number of negative reviews
# and same for source_labels
source_data, source_labels = data[:source_count*2], labels[:source_count*2]
# target_data is all data after source_data's positive and negative vectors and same for target_labels
target_data, target_labels = data[source_count*2:], labels[source_count*2:]
# printing shapes of the source and target arrays
print 'source shape', np.array(source_data).shape
print 'target shape', np.array(target_data).shape

# Domain Adaptation step - This adjusts the source_data based on target_data
#source_data = algorithms.coral(source_data, target_data)
#X_train, X_val, y_train, y_val = cv.train_test_split(source_data, source_labels, test_size=0.2, random_state=42)

print 'Training...'
'''
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[vectors.VECTOR_DIMENSION/2,
                                                          vectors.VECTOR_DIMENSION/4,
                                                          vectors.VECTOR_DIMENSION/2],
                                            n_classes=2, steps=10000)
'''
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[500, 100, 500], n_classes=2, batch_size=source_count/10,
                                            steps=10000, learning_rate=0.05)

while True:
    try:
        classifier.fit(source_data, source_labels)
        print accuracy_score(target_labels, classifier.predict(target_data))
    except KeyboardInterrupt:
        break

accuracy = accuracy_score(target_labels, classifier.predict(target_data))
precision = precision_score(target_labels, classifier.predict(target_data))
f1_score = f1_score(target_labels, classifier.predict(target_data))
print 'Accuracy: ', accuracy, '; Precision: ', precision, '; F1 Score: ', f1_score
print 'Done!'
# Plotting confusion matrix
visualizations.plot_conf_matrix(target_labels, classifier.predict(target_data))
# Plotting roc auc curve
visualizations.draw_roc_auc(target_labels, classifier.predict(target_data))
