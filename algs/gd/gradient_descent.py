#! /usr/bin/python
# -*- coding:utf-8 -*-

"""gradient descent - module."""

import numpy as np

try:
    from .. import util
    from ..transform import identity
    from ..cost import mean_squared_error
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
    import util
    from transform import identity
    from cost import mean_squared_error


def gradient_descent(feature_matrix, output_colvec,
                     num_examples, num_features,
                     alpha, num_iters, theta_colvec=None,
                     transform=identity, cost_func=mean_squared_error,
                     debug=False):
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

    for iter_num in range(0, num_iters):
        # Vectorized Implementation
        hypothesis_colvec = transform(np.matmul(feature_matrix, theta_colvec))
        cost_colvec = hypothesis_colvec - output_colvec
        theta_colvec = theta_colvec - \
            (alpha*np.matmul(feature_matrix.transpose(),
                             cost_colvec))/num_examples

        if debug:
            cost_hist[iter_num, 0] = \
                util.compute_cost_given_hypothesis(hypothesis_colvec,
                                                   output_colvec, num_examples,
                                                   cost_func)

    return theta_colvec, cost_hist


def normal_equation(feature_matrix, output_colvec):
    """Run Gradient Descent using the Normal Equation."""
    fm_transpose_fm = \
        np.matmul(feature_matrix.transpose(), feature_matrix)
    fm_transpose_fm_pinv = np.linalg.pinv(fm_transpose_fm)
    fm_transpose_output_colvec = \
        np.matmul(feature_matrix.transpose(), output_colvec)

    return np.matmul(fm_transpose_fm_pinv, fm_transpose_output_colvec)


if __name__ == '__main__':
    DATASET = 'resources/data/city_dataset_97_2.txt'
    print(f'Gradient Descent For Dataset : {DATASET}')
    data, nrows, ncols = util.\
        get_data_as_matrix(DATASET, Path(__file__))

    output = data[:, ncols - 1:ncols]
    features = np.append(np.ones(shape=(nrows, 1)),
                         data[:, 0:ncols - 1],
                         axis=1)
    theta, cost_history = \
        gradient_descent(features, output, nrows, ncols - 1,
                         theta_colvec=np.zeros(shape=(ncols, 1)),
                         alpha=0.01, num_iters=1500, debug=True)
    print(f'theta(Gradient Descent)={theta}')
    print(f'cost_history={cost_history}')
    print(f'{"*" * 80}')

    DATASET = 'resources/data/housing_dataset_47_3.txt'
    print(f'Gradient Descent For Dataset : {DATASET}')
    data, nrows, ncols = util.\
        get_data_as_matrix(DATASET, Path(__file__))

    util.normalize_data(data[:, 0:ncols - 1])

    output = data[:, ncols - 1:ncols]
    features = np.append(np.ones(shape=(nrows, 1)),
                         data[:, 0:ncols - 1],
                         axis=1)
    theta, cost_history = \
        gradient_descent(features, output, nrows, ncols - 1,
                         theta_colvec=np.zeros(shape=(ncols, 1)),
                         alpha=0.01, num_iters=1500, debug=True)
    print(f'theta(Gradient Descent)={theta}')
    print(f'cost_history={cost_history}')
    print(f'{"*" * 80}')

    theta = \
        normal_equation(features, output)
    print(f'theta (Normal Equation)={theta}')
    print(f'{"*" * 80}')
