import algorithms
import visualizations
import vectors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score
import skflow


def my_model(X, y):
    """This is DNN with 10, 20, 40, 20, 10 hidden layers, and dropout of 0.5 probability."""
    layers = skflow.ops.dnn(X, [250], keep_prob=0.5)
    return skflow.models.logistic_regression(layers, y)

source_train, source_count, target_test, target_count = vectors.prepare_data('big')
print 'Source: ' + str(source_count) + ', Target: ' + str(target_count)
print 'Converting text to vectors...'
source_data, source_labels = vectors.text_to_vector(source_train, source_count)
target_data, target_labels = vectors.text_to_vector(target_test, target_count)
# Domain Adaptation step
source_data = algorithms.coral(source_data, target_data)
print 'Training...'
classifier = RandomForestClassifier(n_estimators=1000)
#classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=2)
while True:
    classifier.fit(source_data, source_labels)
    accuracy = accuracy_score(target_labels, classifier.predict(target_data))
    precision = precision_score(target_labels, classifier.predict(target_data))
    print 'Accuracy: {0:f},'.format(accuracy), 'Precision: {0:f}'.format(precision)
print 'Done!'