#! /usr/bin/python
# -*- coding:utf-8 -*-

"""compute cost - module."""

import numpy as np


def compute_cost(feature_matrix, output_colvec, theta_colvec):
    """Compute cost.

    feature_matrix = (num_examples  x num_features + 1)
                     matrix - features
                     (first column in all ones)
    output_colvec = (num_examples x 1) col vector - actual cost
    num_examples = number of training samples
    num_features = number of features
    cost_func = (1/2num_examples)*(sum(h - output_colvec)**2)

    hypothesis_colvec(i) = theta_colvec[0]*feature_matrix[i, 0] +
                           theta_colvec[1]*feature_matrix[i, 1] +
                           ...
                           0 <= i < num_examples + 1
    """
    num_examples = np.shape(feature_matrix)[0]
#    num_features = np.shape(feature_matrix)[1] - 1
#    theta_colvec = theta_colvec.transpose()
#    hypothesis_colvec = np.zeros(shape=(num_examples, 1))
#    for i in range(0, num_examples):
#        hypothesis_colvec[i, 0] = \
#            np.matmul(theta_colvec,
#                      feature_matrix[i:i+1, :].transpose())[0][0]
#    return compute_cost_given_hypothesis(hypothesis_colvec,
#                                         output_colvec, num_examples)

#   Vectorized Implementation
    cost_colvec = np.matmul(feature_matrix, theta_colvec) - output_colvec
    return (np.matmul(cost_colvec.transpose(),
                      cost_colvec)/(2*num_examples))[0, 0]


def compute_cost_given_hypothesis(hypothesis_colvec,
                                  output_colvec, num_examples):
    """Compute cost given hypothesis and the actual output.

    num_examples = number of training samples
    output_colvec = num_examples x 1 col vector - actual cost
    hypothesis_colvec = cost column vector - num_examples x 1 -
        cost associated with current values of theta
    """
    return np.sum((hypothesis_colvec - output_colvec)**2)/(2*num_examples)


def compute_cost_given_cost(cost_colvec, num_examples):
    """Compute cost given the cost difference.

    num_examples = number of training samples
    cost_colvec = cost column vector - num_examples x 1 -
        cost associated with current values of theta
    """
    return np.sum((cost_colvec)**2)/(2*num_examples)


if __name__ == '__main__':
    try:
        from .. import util
    except ImportError:
        import os
        import sys
        import inspect
        from pathlib import Path
        filename = inspect.getfile(inspect.currentframe())
        abspath = os.path.abspath(filename)
        currentdir = os.path.dirname(abspath)
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0, parentdir)
        import util

    data, mrows, ncols = util.\
        get_data_as_matrix('resources/data/ex1data1.txt', Path(__file__))

    output = data[:, ncols - 1:ncols]
    features = np.append(np.ones(shape=(mrows, 1)),
                         data[:, 0:ncols - 1], axis=1)
    cost = compute_cost(features, output, np.zeros(shape=(ncols, 1)))
    print(cost)
