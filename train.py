import algorithms
import visualizations
import auxiliary
import vectors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import skflow
import numpy as np
from random import shuffle
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


'''
source_train, source_count, target_test, target_count = vectors.prepare_data('small')
print 'Source: ' + str(source_count) + ', Target: ' + str(target_count)
print 'Converting text to vectors...'
# Combining source and target data together for training
raw_data = source_train + target_test
# text_to_vector returns the labels and vectors
data, labels = vectors.text_to_vector(raw_data, 1)
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

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[vectors.VECTOR_DIMENSION/2,
                                                          vectors.VECTOR_DIMENSION/4,
                                                          vectors.VECTOR_DIMENSION/2],
                                            n_classes=2, steps=10000)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[500, 100, 500], n_classes=2, batch_size=source_count/10,
                                            steps=10000, learning_rate=0.05)

while True:
    try:
        classifier.fit(source_data, source_labels)
        print accuracy_score(target_labels, classifier.predict(target_data))
    except KeyboardInterrupt:
        break
'''


def train_model(train_size, coral, source, target=None):
    gammas = np.logspace(-6, -1, 10)
    gammas = [0.1]
    Cs = np.logspace(-2, 2, 5)  # Keeping C values between 10^-2 and 10^2
    Cs = [1]
    if target:
        print source, '->', target
        #source_data, source_labels, target_data, target_labels = vectors.text_to_vector(None, 2, SOURCE, TARGET)
        source = vectors.prepare_data('p_s', source)
        target = vectors.prepare_data('p_s', target)
        shuffle(source)
        shuffle(target)
        data, labels = vectors.text_to_vector(text=source+target, algo_type=3)
        source_data, source_labels = data[:train_size], labels[:train_size]
        target_data, target_labels = data[train_size:], labels[train_size:]

        if coral:
            # Domain Adaptation step - This adjusts the source_data based on target_data
            source_data = algorithms.coral(source_data, target_data)

        # Splitting data for cross validation
        X_train, X_test, y_train, y_test = train_test_split(source_data,
                                                            source_labels,
                                                            test_size=0.2,
                                                            random_state=2)

        print 'Training...'
        # Linear SVM with default square hinge loss
        # Tutorial for tuning parameters used from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
        cv = ShuffleSplit(X_train.shape[0],
                          n_iter=10,
                          test_size=0.2,
                          random_state=0)
        grid = GridSearchCV(estimator=SVC(),
                            cv=cv,
                            param_grid=dict(gamma=gammas, C=Cs))
        grid.fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

        estimator = SVC(kernel='linear',
                        gamma=grid.best_estimator_.gamma,
                        C=grid.best_estimator_.C)
        '''
        print 'Plotting Learning Curve...'
        title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f, C=%.3f$)' % (
            grid.best_estimator_.gamma, grid.best_estimator_.C)
        visualizations.plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
        '''
        print 'Fitting model on complete train/source data...'
        estimator.fit(source_data, source_labels)

        print 'Computing metrics...'
        predicted_labels = estimator.predict(target_data)
        return target_labels, predicted_labels
    else:
        data = vectors.prepare_data('p_s', source)
        shuffle(data)
        data, labels = vectors.text_to_vector(text=data, algo_type=3)
        train_data, train_labels = data[:train_size], labels[:train_size]
        test_data, test_labels = data[train_size:], labels[train_size:]
        # Splitting data for cross validation
        X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                            train_labels,
                                                            test_size=0.2,
                                                            random_state=0)

        print 'Training...'
        # Linear SVM with default square hinge loss
        # Tutorial for tuning parameters used from http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
        cv = ShuffleSplit(X_train.shape[0],
                          n_iter=10,
                          test_size=0.2,
                          random_state=0)
        grid = GridSearchCV(estimator=SVC(),
                            cv=cv,
                            param_grid=dict(gamma=gammas, C=Cs))
        grid.fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

        estimator = SVC(kernel='linear',
                        gamma=grid.best_estimator_.gamma,
                        C=grid.best_estimator_.C)

        print 'Fitting model on train data...'
        estimator.fit(train_data, train_labels)

        print 'Computing metrics...'
        pred_labels = estimator.predict(test_data)
        return test_labels, pred_labels

accuracies, t_accuracies = [], []
for rounds in range(5):
    target_labels, predicted_labels = train_model(train_size=2000,
                                                  coral=True,
                                                  source='books',
                                                  target='electronics')
    t_target_labels, t_predicted_labels = train_model(train_size=1600,
                                                      coral=False,
                                                      source='electronics')

    accuracy = accuracy_score(target_labels, predicted_labels)
    t_accuracy = accuracy_score(t_target_labels, t_predicted_labels)

    print accuracy, t_accuracy
    accuracies.append(accuracy)
    t_accuracies.append(t_accuracy)

accuracy = np.mean(np.array(accuracies))
t_accuracy = np.mean(np.array(t_accuracies))
transfer_loss = auxiliary.calc_transfer_loss(accuracy, t_accuracy)

print transfer_loss, 1.0-accuracy, 1.0-t_accuracy

print 'Plotting other visualisations...'
#visualizations.plot_hyperparameter_heatmap(grid=grid, gamma_list=gammas, C_list=Cs)
# Plotting confusion matrix
#visualizations.plot_conf_matrix(target_labels, classifier.predict(target_data))
# Plotting roc auc curve
#visualizations.draw_roc_auc(target_labels, classifier.predict(target_data))
print 'Done!'
