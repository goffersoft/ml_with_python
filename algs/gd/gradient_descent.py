#! /usr/bin/python
# -*- coding:utf-8 -*-

"""gradient descent - module."""

import numpy as np

try:
    from .. import util
    from .compute_cost import compute_cost_given_hypothesis
except (ImportError, ValueError):
    import os
    import sys
    import inspect
    from pathlib import Path
    filename = inspect.getfile(inspect.currentframe())
    abspath = os.path.abspath(filename)
    currentdir = os.path.dirname(abspath)
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from compute_cost import compute_cost_given_hypothesis
    import util


def gradient_descent(feature_matrix, output_colvec,
                     num_examples, num_features,
                     alpha, num_iters, theta_colvec=None, debug=False):
    """Run Gradient Descent Algorithm."""
    if debug:
        print(f'num_examples(number of training samples)={num_examples}')
        print(f'num_features(number of features)={num_features}')
        print(f'theta_colvec-shape={np.shape(theta_colvec)}')
        print(f'alpha={alpha}')
        print(f'num_iters={num_iters}')
        print(f'feature_matrix-shape={np.shape(feature_matrix)}')
        print(f'output_colvec-shape={np.shape(output_colvec)}')
        cost_hist = np.zeros(shape=(num_iters, 1))
    else:
        cost_hist = None

    if theta_colvec is None:
        theta_colvec = np.zeros(shape=(num_features, 1))

    hypothesis_colvec = np.zeros(shape=(num_examples, 1))

    theta_colvec = theta_colvec.transpose()

    for iter_num in range(0, num_iters):
        for i in range(0, num_examples):
            hypothesis_colvec[i, 0] = \
                np.matmul(theta_colvec,
                          np.reshape(feature_matrix[i, :],
                                     newshape=(num_features, 1)))
        for i in range(0, num_features):
            theta_colvec[0, i] -= \
                alpha * \
                ((np.sum((hypothesis_colvec - output_colvec) *
                         np.reshape(feature_matrix[:, i],
                                    newshape=(num_examples, 1))))/num_examples)

        if debug:
            cost_hist[iter_num, 0] = \
                compute_cost_given_hypothesis(hypothesis_colvec,
                                              output_colvec,
                                              num_examples)

    return theta_colvec.transpose(), cost_hist


if __name__ == '__main__':
    data, mrows, ncols = util.\
        get_data_as_matrix('resources/data/ex1data1.txt', Path(__file__))

    output = np.reshape(data[:, ncols - 1], newshape=(mrows, 1))
    features = np.append(np.ones(shape=(mrows, 1)),
                         np.reshape(data[:, 0], newshape=(mrows, ncols - 1)),
                         axis=1)
    theta, cost_history = \
        gradient_descent(features, output, mrows, ncols,
                         theta_colvec=np.zeros(shape=(ncols, 1)),
                         alpha=0.01, num_iters=1500, debug=True)
    print(theta)
    print(cost_history)
    theta, cost_history = \
        gradient_descent(features, output, mrows, ncols,
                         alpha=0.01, num_iters=1500, debug=False)
    print(theta)
    print(cost_history)
