#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Main - Gradient Descent - module."""

from pathlib import Path
import numpy as np

try:
    from .. import util
    from ..plot_util import line_plot
    from ..plot_util import scatter_plot
    from ..plot_util import contour_plot
    from ..plot_util import surface_plot
    from ..plot_util import close_plot
    from .gradient_descent import gradient_descent
    from .gradient_descent import normal_equation
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
    from plot_util import line_plot
    from plot_util import scatter_plot
    from plot_util import contour_plot
    from plot_util import surface_plot
    from plot_util import close_plot
    from gradient_descent import gradient_descent
    from gradient_descent import normal_equation
    from compute_cost import compute_cost


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
                                             (ncols - 1, ncols)))))

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
                 dataset_title, dataset_xlabel='X', dataset_ylabel='Y'):
    """Plot data as a scatter diagram."""
    fig = None
    subplot = None
    if num_features == 1:
        print('Plotting Data ...')
        fig, subplot = \
            scatter_plot(feature_matrix[:, 1], output_colvec,
                         xlabel=dataset_xlabel,
                         ylabel=dataset_ylabel,
                         title=dataset_title,
                         marker='o', color='r',
                         legend_label='Training data')

    util.pause('Program paused. Press enter to continue.\n')

    return fig, subplot


def run_gradient_descent(feature_matrix, output_colvec,
                         num_examples, num_features,
                         alpha, num_iters, fig, subplot,
                         theta_colvec=None, normal_eq=False,
                         debug=False):
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
    if normal_eq:
        theta_colvec = \
            normal_equation(feature_matrix, output_colvec)
        print(f'Theta found by normal equation : {theta_colvec}')
    else:
        theta_colvec, cost_hist = \
            gradient_descent(feature_matrix, output_colvec,
                             num_examples, num_features,
                             alpha, num_iters, theta_colvec,
                             debug)
        print(f'Theta found by gradient descent: {theta_colvec}')

    if num_features == 1:
        line_plot(feature_matrix[:, 1],
                  np.matmul(feature_matrix, theta_colvec),
                  marker='x', legend_label='Linear regression',
                  color='b', fig=fig, subplot=subplot)
        util.pause('Program paused. Press enter to continue.')

    return theta_colvec, cost_hist


def predict_dataset1(theta_colvec, num_features):
    """Predict profits based on the trained theta vals."""
    predict1 = np.matmul(np.reshape([1, 3.5],
                                    newshape=(1, num_features + 1)),
                         theta_colvec)
    print('For population = 35,000, '
          f'we predict a profit of {predict1[0, 0]*10000}')

    predict2 = np.matmul(np.reshape([1, 7],
                                    newshape=(1, num_features + 1)),
                         theta_colvec)
    print('For population = 70,000, '
          f'we predict a profit of {predict2[0, 0]*10000}')

    util.pause('Program paused. Press enter to continue.')


def predict_dataset2(theta_colvec, num_features, mu_rowvec, sigma_rowvec):
    """Predict profits based on the trained theta vals."""
    num_bedrooms = (3 - mu_rowvec[0, 1])/sigma_rowvec[0, 1]
    sq_footage = (1650 - mu_rowvec[0, 0])/sigma_rowvec[0, 0]
    predict1 = np.matmul(np.reshape([1, num_bedrooms, sq_footage],
                                    newshape=(1, num_features + 1)),
                         theta_colvec)
    print('For a 1650 sq-ft 3 br house, '
          f'we predict an estmated house price of {predict1[0, 0]}')

    util.pause('Program paused. Press enter to continue.')


def run_cost_analysis(feature_matrix, output_colvec,
                      num_features, theta_colvec, cost_hist,
                      dataset_title):
    """Visualize Cost data using contour and sureface plots."""

    def get_z_values(theta0, theta1):
        return compute_cost(feature_matrix, output_colvec,
                            np.reshape([theta0, theta1], newshape=(2, 1)))

    if cost_hist is not None:
        fig, subplot = \
            line_plot(np.reshape(
                range(1, np.size(cost_hist) + 1),
                newshape=(np.size(cost_hist), 1)),
                      cost_hist,
                      xlabel='Number Of Iterations',
                      ylabel='Cost J',
                      marker='x',
                      title=f'{dataset_title}\nConvergence Graph',
                      color='b')
        util.pause('Program paused. Press enter to continue.')
        close_plot(fig)

    if num_features > 1:
        print('Detailed Cost analysis only supported for 1 features!!')
        return None, None

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    fig, subplot = surface_plot(theta0_vals, theta1_vals, get_z_values,
                                title=dataset_title,
                                xlabel='theta_0',
                                ylabel='theta_1')
    util.pause('Program paused. Press enter to continue.')
    close_plot(fig)

    fig, subplot = contour_plot(theta0_vals, theta1_vals,
                                get_z_values,
                                title=dataset_title,
                                levels=np.logspace(-2, 3, 20))
    fig, subplot = line_plot(theta_colvec[0], theta_colvec[1],
                             marker='x', color='r',
                             title=dataset_title,
                             fig=fig, subplot=subplot)
    util.pause('Program paused. Press enter to continue.')
    return fig, subplot


def run_dataset(dataset_name, dataset_title,
                dataset_xlabel='X', dataset_ylabel='Y',
                normalize=False, print_data=False,
                predict_func=None,
                normal_eq=False):
    """Run Various Stages."""
    _, features, output, \
        sample_count, feature_count, mu_rowvec, sigma_rowvec = \
        load_data(dataset_name, normalize, print_data)

    fig, subplot = \
        plot_dataset(features, output, feature_count,
                     dataset_title, dataset_xlabel,
                     dataset_ylabel)

    theta_colvec, cost_hist = \
        run_gradient_descent(features, output,
                             sample_count, feature_count,
                             alpha=0.01, num_iters=1500,
                             fig=fig, subplot=subplot,
                             theta_colvec=None,
                             normal_eq=normal_eq,
                             debug=True)

    if predict_func:
        if normalize:
            predict_func(theta_colvec, feature_count,
                         mu_rowvec, sigma_rowvec)
        else:
            predict_func(theta_colvec, feature_count)

    fig1, _ = run_cost_analysis(features, output, feature_count,
                                theta_colvec, cost_hist, dataset_title)

    close_plot(fig)
    close_plot(fig1)


def run():
    """Run Gradient Descent against various datasets."""
    dataset = 'resources/data/city_dataset_97_2.txt'
    run_dataset(dataset, print_data=True,
                dataset_title='Gradient Descent - Population Dataset - '
                              'Single-Variable',
                dataset_xlabel='Population of City in 10,000s',
                dataset_ylabel='Profit in $10,000s',
                predict_func=predict_dataset1)

    dataset = 'resources/data/housing_dataset_47_3.txt'
    run_dataset(dataset,
                print_data=True, normalize=True,
                dataset_title='Gradient Descent - Housing Prices - '
                              'Multi-Variable',
                predict_func=predict_dataset2)

    run_dataset(dataset,
                print_data=True, normalize=True,
                dataset_title='Normal Equation - Housing Prices - '
                              'Multi-Variable',
                predict_func=predict_dataset2,
                normal_eq=True)


if __name__ == '__main__':
    run()
