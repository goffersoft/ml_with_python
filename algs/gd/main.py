#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Main - Gradient Descent - module."""

from pathlib import Path
import numpy as np

try:
    from .. import util
    from .plotdata import line_plot
    from .plotdata import scatter_plot
    from .plotdata import contour_plot
    from .plotdata import surface_plot
    from .plotdata import close_plot
    from .gradient_descent import gradient_descent
    from .compute_cost import compute_cost
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
    from plotdata import line_plot
    from plotdata import scatter_plot
    from plotdata import contour_plot
    from plotdata import surface_plot
    from plotdata import close_plot
    from gradient_descent import gradient_descent
    from compute_cost import compute_cost


def load_data(path):
    """Part 1: Get Data."""
    print(f'Loading Data from file... {path}')
    return util.get_data_as_matrix(path, Path(__file__))


def prepare_data_for_gradient_descent(data, mrows, ncols):
    """Plot and Prepare Data For Gradient Descent."""
    print('Plotting Data ...')

    feature_matrix = np.append(np.ones(shape=(mrows, 1)),
                               np.reshape(data[:, 0],
                                          newshape=(mrows, ncols - 1)),
                               axis=1)
    output_colvec = np.reshape(data[:, ncols - 1], newshape=(mrows, 1))

    fig, subplot = scatter_plot(feature_matrix[:, 1], output_colvec,
                                xlabel='Population of City in 10,000s',
                                ylabel='Profit in $10,000s',
                                title='Gradient Descent',
                                marker='o', color='r',
                                legend_label='Training data')

    util.pause('Program paused. Press enter to continue.\n')

    return feature_matrix, output_colvec, mrows, ncols, fig, subplot


def run_gradient_descent(feature_matrix, output_colvec,
                         num_examples, num_features,
                         alpha, num_iters, fig, subplot,
                         theta_colvec=None, debug=False):
    """Part 3: Gradient Descent.

    1) num_examples - number of training samples
    2) num_features - number of features (including the all ones first row)
    3) feature_matrix - m,n col vector
    4) output_colvec - m,1 col vector
    5) alpha - alpha value for gradient descent
    6) num_iters - number of iterations
    7) theta_colvec - initial values of theta
    8) debug - print debug info
    """
    print('Running Gradient Descent ...')

    if not theta_colvec:
        theta_colvec = np.zeros(shape=(num_features, 1))

    theta_colvec, jhist = gradient_descent(feature_matrix, output_colvec,
                                           num_examples, num_features,
                                           alpha, num_iters, theta_colvec,
                                           debug)

    print('Theta found by gradient descent: '
          f'{theta_colvec[0]}, {theta_colvec[1]}')

    line_plot(feature_matrix[:, 1],
              np.matmul(feature_matrix, theta_colvec),
              marker='x', legend_label='Linear regression',
              color='b', fig=fig, subplot=subplot)

#   Predict values for population sizes of 35,000 and 70,000
    if num_features == 2:
        predict1 = np.matmul(np.reshape([1, 3.5],
                                        newshape=(1, num_features)),
                             theta_colvec)
        print('For population = 35,000, '
              f'we predict a profit of {predict1[0, 0]*10000}')

        predict2 = np.matmul(np.reshape([1, 7],
                                        newshape=(1, num_features)),
                             theta_colvec)
        print('For population = 70,000, '
              f'we predict a profit of {predict2[0, 0]*10000}')

    util.pause('Program paused. Press enter to continue.')

    return theta_colvec, jhist


def run_cost_analysis(feature_matrix, output_colvec,
                      num_examples, num_features,
                      theta_colvec):
    """Visualize Cost data using contour and sureface plots."""

    def get_z_values(theta0, theta1):
        return compute_cost(feature_matrix, output_colvec,
                            num_examples, num_features,
                            np.reshape([theta0, theta1], newshape=(2, 1)))

    if num_features > 2:
        print('Cost analysis only supported for 2 features!!')
        return None, None

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    fig, subplot = surface_plot(theta0_vals, theta1_vals, get_z_values,
                                xlabel='theta_0',
                                ylabel='theta_1')
    util.pause('Program paused. Press enter to continue.')
    close_plot(fig)
    fig, subplot = contour_plot(theta0_vals, theta1_vals,
                                get_z_values,
                                levels=np.logspace(-2, 3, 20))
    fig, subplot = line_plot(theta_colvec[0], theta_colvec[1],
                             marker='x', color='r',
                             fig=fig, subplot=subplot)
    util.pause('Program paused. Press enter to continue.')
    return fig, subplot


def run():
    """Run Various Stages."""
    data, mrows, ncols = load_data('resources/data/ex1data1.txt')

    features, output, sample_count, feature_count, fig, subplot = \
        prepare_data_for_gradient_descent(data, mrows, ncols)

    theta_colvec, _ = \
        run_gradient_descent(features, output,
                             sample_count, feature_count,
                             alpha=0.01, num_iters=1500,
                             fig=fig, subplot=subplot,
                             theta_colvec=None, debug=True)

    fig1, _ = run_cost_analysis(features, output, sample_count,
                                feature_count, theta_colvec)

    close_plot(fig)
    close_plot(fig1)


if __name__ == '__main__':
    run()
