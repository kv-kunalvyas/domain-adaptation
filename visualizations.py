#TODO: code for learning curve
#TODO: lstm epoch graph
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc
import numpy as np


def two_d(x, y):
    plt.plot(x, y)
    plt.show()


def plot_conf_matrix(true, pred, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(true, pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['0', '1']))
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def draw_roc_auc(true, pred):
    fpr, tpr, _ = roc_curve(true, pred)

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print 'ROC AUC: %0.2f' % roc_auc

    # Plot of a ROC curve for a specific class
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()