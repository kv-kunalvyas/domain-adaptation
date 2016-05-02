# the CORAL algorithm
import numpy
import scipy


def coral(d_source, d_target):
    print 'Adapting Domains - CORAL'
    # Calculating covariances of source and target. Here, numpy.eye(d_source.shape[1]) is the identity matrix of size
    # n X n where n is the second dimension of the matrix, or the number of columns.
    # numpy.transpose(d_source) transposes the matrix so as to add it's covariance to the identity matrix
    covariance_source = numpy.add(numpy.cov(numpy.transpose(d_source)), numpy.eye(d_source.shape[1]))
    covariance_target = numpy.add(numpy.cov(numpy.transpose(d_target)), numpy.eye(d_target.shape[1]))
    # scipy.linalg.sqrtm computes the square root of the matrix (not element wise)
    # numpy.linalg.inv computes the inverse/reciprocal of the matrix
    # numpy.dot multiplies the square root and source matrix
    d_source = numpy.dot(d_source, scipy.linalg.sqrtm(numpy.linalg.inv(covariance_source)))
    # Finally, adjusted_source is the output we need
    adjusted_source = numpy.dot(d_source, scipy.linalg.sqrtm(covariance_target))
    return adjusted_source
