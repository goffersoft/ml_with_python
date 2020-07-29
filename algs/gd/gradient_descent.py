#! /usr/bin/python
# -*- coding:utf-8 -*-

"""gradient descent - module."""

import numpy as np

try:
    from .. import util
#    from .compute_cost import compute_cost_given_hypothesis
    from .compute_cost import compute_cost_given_cost
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
#    from compute_cost import compute_cost_given_hypothesis
    from compute_cost import compute_cost_given_cost
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
        theta_colvec = np.zeros(shape=(num_features + 1, 1))

#    hypothesis_colvec = np.zeros(shape=(num_examples, 1))

#    theta_colvec = theta_colvec.transpose()
#
    for iter_num in range(0, num_iters):
        # for i in range(0, num_examples):
        #    hypothesis_colvec[i, 0] = \
        #        np.matmul(theta_colvec,
        #                  feature_matrix[i:i+1, :].transpose())
        # for i in range(0, num_features + 1):
        #    theta_colvec[0, i] -= \
        #        alpha * \
        #            ((np.sum((hypothesis_colvec - output_colvec) *
        #             feature_matrix[:, i:i+1]))/num_examples)

        # Vectorized Implementation
        cost_colvec = np.matmul(feature_matrix, theta_colvec) - output_colvec
        theta_colvec = theta_colvec - \
            (alpha*np.matmul(feature_matrix.transpose(),
                             cost_colvec))/num_examples

        if debug:
            cost_hist[iter_num, 0] = \
                compute_cost_given_cost(cost_colvec, num_examples)

    return theta_colvec, cost_hist


if __name__ == '__main__':
    DATASET = 'resources/data/ex1data1.txt'
    print(f'Gradient Descent For Dataset : {DATASET}')
    data, mrows, ncols = util.\
        get_data_as_matrix(DATASET, Path(__file__))

    output = data[:, ncols - 1:ncols]
    features = np.append(np.ones(shape=(mrows, 1)),
                         data[:, 0:ncols - 1],
                         axis=1)
    theta, cost_history = \
        gradient_descent(features, output, mrows, ncols - 1,
                         theta_colvec=np.zeros(shape=(ncols, 1)),
                         alpha=0.01, num_iters=1500, debug=True)
    print(f'theta={theta}')
    print(f'cost_history={cost_history}')
    print(f'{"*" * 80}')

    DATASET = 'resources/data/ex1data2.txt'
    print(f'Gradient Descent For Dataset : {DATASET}')
    data, mrows, ncols = util.\
        get_data_as_matrix(DATASET, Path(__file__))

    util.normalize_data(data[:, 0:ncols - 1])

    output = data[:, ncols - 1:ncols]
    features = np.append(np.ones(shape=(mrows, 1)),
                         data[:, 0:ncols - 1],
                         axis=1)
    theta, cost_history = \
        gradient_descent(features, output, mrows, ncols - 1,
                         theta_colvec=np.zeros(shape=(ncols, 1)),
                         alpha=0.01, num_iters=1500, debug=True)
    print(f'theta={theta}')
    print(f'cost_history={cost_history}')
    print(f'{"*" * 80}')
