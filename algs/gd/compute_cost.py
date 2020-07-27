#! /usr/bin/python
# -*- coding:utf-8 -*-

"""compute cost - module."""

import numpy as np


def compute_cost(feature_matrix, output_colvec,
                 num_examples, num_features, theta_colvec=None):
    """Compute cost.

    feature_matrix = (num_examples  x num_features)
                     dimensional matrix - features
                     (first column in all ones)
    output_colvec = num_examples x 1 col vector - actual cost
    num_examples = number of training samples
    num_features = number of features
    cost_func = (1/2num_examples)*(sum(h - output_colvec)**2)

    hypothesis_colvec(i) = theta_colvec[0]*feature_matrix[i, 0] +
                           theta_colvec[1]*feature_matrix[i, 1] +
                           ...
                           0 <= i < num_examples
    """
    if theta_colvec is None:
        theta_colvec = np.zeros(shape=(num_features, 1))

    theta_colvec = theta_colvec.transpose()

    hypothesis_colvec = np.zeros(shape=(num_examples, 1))

    for i in range(0, num_examples):
        hypothesis_colvec[i, 0] = \
            np.matmul(theta_colvec,
                      np.reshape(feature_matrix[i, :],
                                 newshape=(num_features, 1)))[0][0]

    return compute_cost_given_hypothesis(hypothesis_colvec,
                                         output_colvec, num_examples)


def compute_cost_given_hypothesis(hypothesis_colvec,
                                  output_colvec, num_examples):
    """Compute cost.

    num_examples = number of training samples
    output_colvec = m x 1 col vector - actual host
    hypothesis_colvec = cost column vector - num_examples x 1 -
        cost associated with current values of theta
    """
    return np.sum((hypothesis_colvec - output_colvec)**2)/(2*num_examples)


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

    output = np.reshape(data[:, ncols - 1], newshape=(mrows, 1))
    features = np.append(np.ones(shape=(mrows, 1)),
                         np.reshape(data[:, 0], newshape=(mrows, ncols - 1)),
                         axis=1)
    theta = compute_cost(features, output, mrows, ncols,
                         np.zeros(shape=(ncols, 1)))
    print(theta)

    theta = compute_cost(features, output, mrows, ncols)
    print(theta)
