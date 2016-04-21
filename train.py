import algorithms
import visualizations
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
raw_data = source_train + target_test
data, labels = vectors.text_to_vector(raw_data, 0)
source_data, source_labels = data[:source_count*2], labels[:source_count*2]
target_data, target_labels = data[source_count*2:], labels[source_count*2:]
print np.array(source_data).shape
print np.array(target_data).shape
# Domain Adaptation step
#source_data = algorithms.coral(source_data, target_data)
X_train, X_val, y_train, y_val = cv.train_test_split(source_data, source_labels, test_size=0.2, random_state=42)

print 'Training...'

#classifier = RandomForestClassifier(n_estimators=1000)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[vectors.VECTOR_DIMENSION/2], n_classes=2, steps=15000)
#classifier = SVC(C=0.1)

classifier.fit(source_data, source_labels)

accuracy = accuracy_score(target_labels, classifier.predict(target_data))
precision = precision_score(target_labels, classifier.predict(target_data))
f1_score = f1_score(target_labels, classifier.predict(target_data))
crossval_score = cv.cross_val_score(classifier, source_data, source_labels, cv=5, scoring=make_scorer(accuracy_score))
print 'Accuracy: ', accuracy, '; Precision: ', precision, '; F1 Score: ', f1_score, '; CV Score: ', crossval_score
print 'Done!'
