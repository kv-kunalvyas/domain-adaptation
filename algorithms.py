# the CORAL algorithm
import numpy as np


def coral(d_source, d_target):
    print 'Adapting Domains - CORAL'
    d_source = softmax(d_source)
    d_target = softmax(d_target)
    covariance_source = np.cov(d_source) + np.eye(len(d_source))
    covariance_target = np.cov(d_target) + np.eye(len(d_target))
    d_source = np.dot(np.nan_to_num(np.sqrt(np.reciprocal(covariance_source))), d_source)
    ds_out = np.dot(np.nan_to_num(softmax(np.sqrt(covariance_target))), d_source)
    return np.nan_to_num(ds_out)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
