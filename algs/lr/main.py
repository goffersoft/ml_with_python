#! /usr/bin/python
# -*- coding:utf-8 -*-
"""Main - Logistic Regression - module."""

from pathlib import Path
import numpy as np

try:
    from .. import util
    from ..transform import sigmoid
    from ..cost import cross_entropy
    from ..plot import scatter_plot
    from ..plot import line_plot
    from ..gd.gradient_descent import gradient_descent
except ImportError:
    import os
    import sys
    import inspect
    filename = inspect.getfile(inspect.currentframe())
    abspath = os.path.abspath(filename)
    currentdir = os.path.dirname(abspath)
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import util
    from transform import sigmoid
    from cost import cross_entropy
    from plot import scatter_plot
    from plot import line_plot
    from gd.gradient_descent import gradient_descent


def load_data(dataset, normalize=False, print_data=False):
    """Get Data."""
    print(f'Loading Data from file... {dataset}')
    data, nrows, ncols = util.get_data_as_matrix(dataset, Path(__file__))

    mu_rowvec = None
    sigma_rowvec = None

    if print_data:
        print('First 10 rows of the dataset:')
        print('\n'.join(f'rownum={i} : '
                        f'feature_matrix_row={j}, output_row={k}'
                        for i, j, k in
                        util.iterate_matrix(data, ((0, 10),),
                                            ((0, ncols - 1),
                                             (ncols - 2, ncols)))))

    if normalize:
        _, mu_rowvec, sigma_rowvec = \
            util.normalize_data(data[:, 0:np.shape(data)[1] - 1])

        if print_data:
            print(f'mu={mu_rowvec}')
            print(f'sigma={sigma_rowvec}')
            print('First 10 rows of the dataset (After Normalization):')
            print('\n'.join(f'rownum={i} : '
                            f'feature_matrix_row={j}, output_row={k}'
                            for i, j, k in
                            util.iterate_matrix(data, ((0, 10),),
                                                ((0, ncols - 1),
                                                 (ncols - 1, ncols)))))

    feature_matrix = np.append(np.ones(shape=(nrows, 1)),
                               data[:, 0:ncols - 1],
                               axis=1)
    output_colvec = np.reshape(data[:, ncols - 1], newshape=(nrows, 1))

    return data, feature_matrix, output_colvec, \
        nrows, ncols - 1, mu_rowvec, sigma_rowvec


def plot_dataset(feature_matrix, output_colvec, num_features,
                 dataset_title, dataset_xlabel='X', dataset_ylabel='Y',
                 label=None):
    """Plot data as a scatter diagram."""
    fig = None
    subplot = None
    if num_features == 2:
        print('Plotting Data ...')
        fig, subplot = \
            scatter_plot(feature_matrix[(output_colvec == 1).nonzero(), 1],
                         feature_matrix[(output_colvec == 1).nonzero(), 2],
                         xlabel=dataset_xlabel,
                         ylabel=dataset_ylabel,
                         title=dataset_title,
                         marker='.', color='black',
                         label=label[0],
                         linewidths=None)

        scatter_plot(feature_matrix[(output_colvec == 0).nonzero(), 1],
                     feature_matrix[(output_colvec == 0).nonzero(), 2],
                     marker='+', color='y', linewidths=None,
                     label=label[1],
                     fig=fig, subplot=subplot)

    util.pause('Program paused. Press enter to continue.\n')

    return fig, subplot


def run_gradient_descent(feature_matrix, output_colvec,
                         num_examples, num_features,
                         alpha, num_iters, fig, subplot,
                         theta_colvec=None, debug=False):
    """Run Gradient Descent/Normal Equation.

    1) num_examples - number of training samples
    2) num_features - number of features
    3) feature_matrix - num_examples x (num_features + 1)
    4) output_colvec - num_examples x 1 col vector
    5) alpha - alpha value for gradient descent
    6) num_iters - number of iterations
    7) theta_colvec - (num_features + 1) x 1 col vector
                      initial values of theta
    8) debug - print debug info
    """
    print('Running Gradient Descent ...')

    if not theta_colvec:
        theta_colvec = np.zeros(shape=(num_features + 1, 1))

    cost_hist = None
    theta_colvec, cost_hist = \
        gradient_descent(feature_matrix, output_colvec,
                         num_examples, num_features,
                         alpha, num_iters, theta_colvec,
                         transform=sigmoid,
                         cost_func=cross_entropy,
                         debug=debug)
    print(f'Theta found by gradient descent: {theta_colvec}')
    if debug:
        print(f'cost history : {cost_hist}')

    if num_features == 2:
        # With 2 features. Decision boundary is
        #            theta-0 + theta-1*x1 + theta-2*x2 >= 0
        # to draw a line we need 2 points.
        # take the min and max of the first feature
        # x2 = -1/theta-2*(theta-0 + theta-1*x1)
        xdata_colvec = np.reshape([np.min(feature_matrix[:, 1]),
                                   np.max(feature_matrix[:, 1])],
                                  newshape=(2, 1))
        ydata_colvec = -(theta_colvec[0, 0] +
                         (theta_colvec[1, 0] *
                          xdata_colvec))/theta_colvec[2, 0]
        line_plot(xdata_colvec,
                  ydata_colvec,
                  marker='x', label='Logistic regression',
                  color='r', markersize=2,
                  fig=fig, subplot=subplot)
        util.pause('Program paused. Press enter to continue.')

    return theta_colvec, cost_hist


def run_dataset(dataset_name, dataset_title,
                dataset_xlabel='X', dataset_ylabel='Y',
                normalize=False, print_data=False,
                label=None,
                predict_func=None):
    """Run Logistic Regression."""
    _, features, output, \
        sample_count, feature_count, _, _ = \
        load_data(dataset_name, normalize, print_data)

    fig, subplot = \
        plot_dataset(features, output, feature_count,
                     dataset_title, dataset_xlabel,
                     dataset_ylabel, label)

    theta_colvec, cost_hist = \
        run_gradient_descent(features, output,
                             sample_count, feature_count,
                             alpha=1.0, num_iters=1500,
                             fig=fig, subplot=subplot,
                             theta_colvec=None,
                             debug=True)


def run():
    """Run Logistic Regression against various datasets."""
    dataset = 'resources/data/exam_dataset_100_3.txt'
    run_dataset(dataset, print_data=True, normalize=True,
                dataset_title='Logistic Regression - Exam Dataset',
                dataset_xlabel='Exam1 Score',
                dataset_ylabel='Exam2 Score',
                label=['Admitted', 'Not Admitted'],
                predict_func=None)

    dataset = 'resources/data/microchip_test_dataset_118_3.txt'
    run_dataset(dataset, print_data=True, normalize=True,
                dataset_title='Logistic Regression - Microchip Test1 Dataset',
                dataset_xlabel='Test1 Results',
                dataset_ylabel='Test2 Results',
                label=['Accepted', 'Rejected'],
                predict_func=None)


if __name__ == '__main__':
    run()
