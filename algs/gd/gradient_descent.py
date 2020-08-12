#! /usr/bin/python
# -*- coding:utf-8 -*-

"""gradient descent - module."""

from math import isclose
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


def gradient_descent_alphas(feature_matrix, output_colvec,
                            num_examples, num_features,
                            num_iters, theta_colvec=None,
                            alpha_range=None, num_alphas=None,
                            transform=identity, cost_func=mean_squared_error,
                            regularization_param=0,
                            debug=False, debug_print=False):
    """Run Gradient Descent Algorithm.

    Iterate over alphas and returns the following:
    theta values = matrix ( (num_features + 1) X num_alphas )
    alphas_rowvec = row vector ( 1 X num_alphas )
    if debug is set to True
         cost history = matrix ( num_iters X num_alphas )
    else
         None
    """
    if debug_print:
        print(f'num_examples(number of training samples)={num_examples}')
        print(f'num_features(number of features)={num_features}')
        print(f'theta_colvec-shape={np.shape(theta_colvec)}')
        print(f'alpha_range={alpha_range}')
        print(f'num_alphas={num_alphas}')
        print(f'num_iters={num_iters}')
        print(f'debug={debug}')
        print(f'debug_print={debug_print}')
        print(f'feature_matrix-shape={np.shape(feature_matrix)}')
        print(f'output_colvec-shape={np.shape(output_colvec)}')

    orig_theta_colvec = theta_colvec

    if orig_theta_colvec is None:
        orig_theta_colvec = np.zeros(shape=(num_features + 1, 1))

    if None in (alpha_range, num_alphas):
        num_alphas = 5
        alpha_range = (x*0.3 for x in range(1, num_alphas + 1))

    if debug:
        cost_matrix = np.zeros(shape=(num_iters, num_alphas))
    else:
        cost_matrix = None

    theta_values = np.zeros(shape=(num_features + 1, num_alphas))
    alphas_rowvec = np.zeros(shape=(1, num_alphas))

    optimal_alpha = 0
    optimal_cost = np.inf
    optimal_theta_colvec = np.nan

    for index, alpha in enumerate(alpha_range):
        theta_colvec = np.copy(orig_theta_colvec)
        theta_colvec, cost_hist = \
            gradient_descent(feature_matrix, output_colvec,
                             num_examples, num_features,
                             theta_colvec=theta_colvec,
                             alpha=alpha, num_iters=num_iters,
                             transform=transform, cost_func=cost_func,
                             regularization_param=regularization_param,
                             debug=debug, debug_print=False)
        if debug:
            tmp_cost = np.min(cost_hist)
            cost_matrix[:, index:index + 1] = cost_hist
        else:
            tmp_cost = util.compute_cost(feature_matrix,
                                         output_colvec, num_examples,
                                         transform, cost_func,
                                         regularization_param)
        if not isclose(tmp_cost, optimal_cost) and tmp_cost < optimal_cost:
            optimal_cost = tmp_cost
            optimal_alpha = alpha
            optimal_theta_colvec = theta_colvec

        alphas_rowvec[0, index] = alpha
        theta_values[:, index:index + 1] = theta_colvec[:]

    return optimal_theta_colvec, optimal_alpha, optimal_cost, \
        theta_values, alphas_rowvec, cost_matrix


def gradient_descent_iterate_alphas(feature_matrix, output_colvec,
                                    num_examples, num_features,
                                    num_iters, theta_colvec=None,
                                    alpha_range=None, num_alphas=None,
                                    transform=identity,
                                    cost_func=mean_squared_error,
                                    regularization_param=0,
                                    debug=False, debug_print=False):
    """Run Gradient Descent Algorithm as a Iterator pattern.

    Iterate over alphas and returns (yields) the following:
    theta_colvec = col vector ( (num_features + 1) X 1 )
    alpha = the alpha value used to compute this theta
    if debug is set to True
         cost history = matrix ( num_iters X 1)
    else
         None
    """
    if debug_print:
        print(f'num_examples(number of training samples)={num_examples}')
        print(f'num_features(number of features)={num_features}')
        print(f'theta_colvec-shape={np.shape(theta_colvec)}')
        print(f'alpha_range={alpha_range}')
        print(f'num_alphas={num_alphas}')
        print(f'num_iters={num_iters}')
        print(f'debug={debug}')
        print(f'debug_print={debug_print}')
        print(f'feature_matrix-shape={np.shape(feature_matrix)}')
        print(f'output_colvec-shape={np.shape(output_colvec)}')

    orig_theta_colvec = theta_colvec

    if orig_theta_colvec is None:
        orig_theta_colvec = np.zeros(shape=(num_features + 1, 1))

    if None in (alpha_range, num_alphas):
        num_alphas = 5
        alpha_range = (x*0.3 for x in range(1, num_alphas + 1))

    for alpha in alpha_range:
        theta_colvec = np.copy(orig_theta_colvec)
        theta_colvec, cost_hist = \
            gradient_descent(feature_matrix, output_colvec,
                             num_examples, num_features,
                             theta_colvec=theta_colvec,
                             alpha=alpha, num_iters=num_iters,
                             transform=transform, cost_func=cost_func,
                             regularization_param=regularization_param,
                             debug=debug, debug_print=False)
        yield theta_colvec, alpha, cost_hist


def gradient_descent(feature_matrix, output_colvec,
                     num_examples, num_features,
                     alpha, num_iters, theta_colvec=None,
                     transform=identity, cost_func=mean_squared_error,
                     regularization_param=0,
                     debug=False, debug_print=False):
    """Run Gradient Descent Algorithm once for a given alpha.

    returns the following :
    theta_colvec = col vector ( (num_features + 1) X 1 )
    if debug is set to True
         cost history = matrix ( num_iters X 1)
    else
         None
    """
    if debug_print:
        print(f'num_examples(number of training samples)={num_examples}')
        print(f'num_features(number of features)={num_features}')
        print(f'theta_colvec-shape={np.shape(theta_colvec)}')
        print(f'alpha={alpha}')
        print(f'num_iters={num_iters}')
        print(f'debug={debug}')
        print(f'debug_print={debug_print}')
        print(f'feature_matrix-shape={np.shape(feature_matrix)}')
        print(f'output_colvec-shape={np.shape(output_colvec)}')

    if theta_colvec is None:
        theta_colvec = np.zeros(shape=(num_features + 1, 1))

    if debug:
        cost_hist = np.zeros(shape=(num_iters, 1))
    else:
        cost_hist = None

    for iter_num in range(0, num_iters):
        # Vectorized Implementation
        hypothesis_colvec = transform(feature_matrix @ theta_colvec)
        cost_colvec = hypothesis_colvec - output_colvec
        gradient = feature_matrix.transpose() @ cost_colvec
        adjust_theta = 1 - (alpha * regularization_param)/num_examples
        tmp_theta = 0
        if adjust_theta:
            tmp_theta = theta_colvec[0, 0]

        theta_colvec = theta_colvec * adjust_theta - \
            ((alpha*gradient)/num_examples)

        if adjust_theta:
            theta_colvec[0, 0] = tmp_theta - \
                ((alpha * gradient[0, 0])/num_examples)

        if debug:
            cost_hist[iter_num, 0] = \
                util.compute_cost_given_hypothesis(hypothesis_colvec,
                                                   output_colvec, num_examples,
                                                   cost_func,
                                                   regularization_param,
                                                   theta_colvec)

    return theta_colvec, cost_hist


def normal_equation(feature_matrix, output_colvec):
    """Run Gradient Descent using the Normal Equation."""
    fm_transpose_fm = feature_matrix.transpose() @ feature_matrix
    fm_transpose_fm_pinv = np.linalg.pinv(fm_transpose_fm)
    fm_transpose_output_colvec = feature_matrix.transpose() @ output_colvec

    return fm_transpose_fm_pinv @ fm_transpose_output_colvec


if __name__ == '__main__':
    DATASET = 'resources/data/city_dataset_97_2.txt'
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
                         alpha=0.03, num_iters=1500,
                         debug=True, debug_print=True)
    print(f'theta(Gradient Descent)={theta}')
    print(f'cost_history={cost_history}')

    print(f'{"*" * 80}')
    optimal_theta, optimal_alpha_val, optimal_cost_val, \
        thetas, alphas, cost_history = \
        gradient_descent_alphas(features, output, nrows, ncols - 1,
                                theta_colvec=np.zeros(shape=(ncols, 1)),
                                num_iters=1500,
                                debug=True, debug_print=True)
    print(f'thetas(Gradient Descent)={thetas}')
    print(f'alphas={alphas}')
    print(f'optima_alpha_val={optimal_alpha_val}')
    print(f'optima_cost_val={optimal_cost_val}')
    print(f'optima_theta={optimal_theta}')

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
                         alpha=0.01, num_iters=1500,
                         debug=True, debug_print=True)
    print(f'theta(Gradient Descent)={theta}')
    print(f'cost_history={cost_history}')
    print(f'{"*" * 80}')

    optimal_theta, optimal_alpha_val, optimal_cost_val, \
        thetas, alphas, cost_history = \
        gradient_descent_alphas(features, output, nrows, ncols - 1,
                                theta_colvec=np.zeros(shape=(ncols, 1)),
                                num_iters=1500,
                                debug=True, debug_print=True)
    print(f'thetas(Gradient Descent)={thetas}')
    print(f'alphas={alphas}')
    print(f'optima_alpha_val={optimal_alpha_val}')
    print(f'optima_cost_val={optimal_cost_val}')
    print(f'optima_theta={optimal_theta}')

    print(f'{"*" * 80}')
    theta = \
        normal_equation(features, output)
    print(f'theta (Normal Equation)={theta}')
    print(f'{"*" * 80}')
