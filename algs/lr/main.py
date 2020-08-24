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
    from ..plot import contour_plot
    from ..plot import image_plot
    from ..plot import close_plot
    from ..plot import get_markers
    from ..plot import get_colors
    from ..gd.gradient_descent import gradient_descent_alphas
    from .logistic_regression import training_accuracy
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
    from plot import contour_plot
    from plot import image_plot
    from plot import close_plot
    from plot import get_markers
    from plot import get_colors
    from gd.gradient_descent import gradient_descent_alphas
    from logistic_regression import training_accuracy


def load_data(dataset, dataset_type='txt', normalize=False, print_data=False):
    """Get Data."""
    print(f'Loading Data from file... {dataset}')
    data, nrows, ncols = util.get_data_as_matrix(dataset, Path(__file__),
                                                 filetype=dataset_type)

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


def plot_dataset(feature_matrix, output_colvec,
                 dataset_title, dataset_xlabel='X', dataset_ylabel='Y',
                 label=None, plot_image=False,
                 img_size=None,
                 num_images_per_row=None,
                 num_images_per_col=None):
    """Plot data as a scatter diagram."""
    fig = None
    subplot = None
    print('Plotting Data ...')
    yvals = np.unique(output_colvec)
    markers = get_markers(len(yvals))
    colors = get_colors(len(yvals))
    if plot_image:
        random_rows = np.random.randint(0, len(output_colvec),
                                        num_images_per_row *
                                        num_images_per_col)

        def get_img(index):
            return np.reshape(feature_matrix[random_rows[index], 0:-1],
                              newshape=img_size)

        fig, subplot = image_plot(num_images_per_row,
                                  num_images_per_col, get_img)
    else:
        for index, yval in enumerate(yvals):
            fig, subplot = \
                scatter_plot(feature_matrix[(output_colvec == yval).
                                            nonzero(), 1],
                             feature_matrix[
                                 (output_colvec == yval).nonzero(), 2],
                             xlabel=dataset_xlabel,
                             ylabel=dataset_ylabel,
                             title=dataset_title,
                             marker=markers[index], color=colors[index],
                             label=label[index],
                             linewidths=None,
                             fig=fig, subplot=subplot)

    util.pause('Program paused. Press enter to continue.\n')

    return fig, subplot


def run_logistic_regression(feature_matrix, output_colvec,
                            num_examples, num_features,
                            num_iters, fig, subplot,
                            theta_colvec=None, debug=False,
                            uv_vals=None, degree=None,
                            regularization_param=0):
    """Run Logistic Regression.

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

    def get_z_values(u_val, v_val):
        return (util.add_features(u_val, v_val, degree) @
                theta_colvec)[0, 0]

    print('Running Logistic Regression...')

    if not theta_colvec:
        theta_colvec = np.zeros(shape=(num_features + 1, 1))

    cost_hist = None

    theta_colvec, alpha, cost, \
        thetas, alphas, cost_hist, = \
        gradient_descent_alphas(feature_matrix, output_colvec,
                                num_examples, num_features,
                                num_iters, theta_colvec,
                                transform=sigmoid,
                                cost_func=cross_entropy,
                                regularization_param=regularization_param,
                                debug=debug, debug_print=debug)
    print(f'Theta found by gradient descent(alpha={alpha}, '
          f'cost={cost}) : {theta_colvec}')
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
                  fig=fig, subplot=subplot, linewidth=1)
        util.pause('Program paused. Press enter to continue.')
    elif num_features > 2 and degree:
        if not uv_vals:
            uv_vals = [0, 0]
            uv_vals[0] = np.linspace(-2, 2, 50)
            uv_vals[1] = np.linspace(-2, 2, 50)

        fig, subplot = contour_plot(uv_vals[0], uv_vals[1],
                                    get_z_values,
                                    levels=0,
                                    fig=fig, subplot=subplot)

    return theta_colvec, alpha, cost, thetas, alphas, cost_hist


def predict_dataset1(theta_colvec, num_features, mu_rowvec, sigma_rowvec):
    """Predict admission probability based on the trained theta vals."""
    exam1_score = (45 - mu_rowvec[0, 0])/sigma_rowvec[0, 0]
    exam2_score = (85 - mu_rowvec[0, 1])/sigma_rowvec[0, 1]
    predict1 = sigmoid(np.reshape([1, exam1_score, exam2_score],
                                  newshape=(1, num_features + 1)) @
                       theta_colvec)

    print('For a student with scores 45 and 85, we predict an admission '
          f'probability of {predict1[0, 0]}')

    util.pause('Program paused. Press enter to continue.')


def run_cost_analysis(alphas, cost_hist, dataset_title):
    """Run Cost analysis based on learnt values of theta."""
    min_cost = np.zeros(shape=(1, np.size(alphas)))
    if cost_hist is not None:
        fig = None
        subplot = None
        colors = get_colors(np.shape(alphas)[1])
        for index in range(0, np.shape(alphas)[1]):
            min_cost[0, index] = np.min(cost_hist[:, index])
            fig, subplot = \
                line_plot(np.reshape(range(1, np.shape(cost_hist)[0] + 1),
                                     newshape=(np.shape(cost_hist)[0], 1)),
                          cost_hist[:, index],
                          xlabel='Number Of Iterations',
                          ylabel='Cost J',
                          marker='x', markersize=2,
                          title=f'{dataset_title}\nConvergence Graph',
                          color=colors[index],
                          label=f'alpha={alphas[0, index]}',
                          linewidth=1,
                          fig=fig, subplot=subplot)
        util.pause('Program paused. Press enter to continue.')
        close_plot(fig)

        fig, subplot = \
            line_plot(alphas.transpose(), min_cost.transpose(),
                      xlabel='alpha',
                      ylabel='cost',
                      marker='x', markersize=2,
                      color='r',
                      title=f'{dataset_title}\n Alphas Vs Cost',
                      linewidth=2)
        util.pause('Program paused. Press enter to continue.')
        close_plot(fig)


def run_mcc_dataset(dataset_name, dataset_type, dataset_title,
                    dataset_xlabel='X', dataset_ylabel='Y',
                    label=None, normalize=False, print_data=False,
                    plot_image=False, image_size=None,
                    num_images_per_row=None, num_images_per_col=None):
    """Run Logistic Regression For multi class classification problem."""
    _, feature_matrix, output_colvec, \
        _, _, _, _ = \
        load_data(dataset_name, dataset_type, normalize, print_data)

    fig, _ = \
        plot_dataset(feature_matrix, output_colvec,
                     dataset_title, dataset_xlabel,
                     dataset_ylabel, label,
                     plot_image, image_size,
                     num_images_per_row, num_images_per_col)

    close_plot(fig)


def run_dataset(dataset_name, dataset_title, dataset_type='txt',
                dataset_xlabel='X', dataset_ylabel='Y',
                normalize=False, print_data=False,
                label=None,
                predict_func=None,
                add_features=False,
                degree=None,
                regularization_param=0,
                plot_image=False, image_size=None,
                num_images_per_row=None, num_images_per_col=None):
    """Run Logistic Regression For single class classification problem."""
    _, feature_matrix, output_colvec, \
        num_examples, num_features, mu_rowvec, sigma_rowvec = \
        load_data(dataset_name, dataset_type, normalize, print_data)

    fig, subplot = \
        plot_dataset(feature_matrix, output_colvec,
                     dataset_title, dataset_xlabel,
                     dataset_ylabel, label,
                     plot_image, image_size,
                     num_images_per_row, num_images_per_col)

    if add_features and np.shape(feature_matrix)[1] >= 2:
        if not degree:
            degree = 6
        print('Improve Training Accuracy By Adding New Features...')
        feature_matrix = util.add_features(feature_matrix[:, 1],
                                           feature_matrix[:, 2], degree)
        num_examples, num_features = np.shape(feature_matrix)
        num_features -= 1
        print('num_features={num_features}, num_examples={num_examples}')

    theta_colvec, alpha, cost, \
        _, alphas, cost_hist = \
        run_logistic_regression(feature_matrix, output_colvec,
                                num_examples, num_features,
                                num_iters=1500,
                                fig=fig, subplot=subplot,
                                theta_colvec=None,
                                debug=True,
                                degree=degree,
                                regularization_param=regularization_param)

    if predict_func:
        predict_func(theta_colvec, num_features,
                     mu_rowvec, sigma_rowvec)

    accuracy = training_accuracy(feature_matrix,
                                 output_colvec, theta_colvec)
    print(f'Train Accuracy(alpha={alpha}, cost={cost}): {accuracy}')
    util.pause('Program paused. Press enter to continue.')

    run_cost_analysis(alphas, cost_hist, dataset_title)

    close_plot(fig)


def run():
    """Run Logistic Regression against various datasets."""
    dataset = 'resources/data/exam_dataset_100_3.txt'
    run_dataset(dataset, print_data=True, normalize=True,
                dataset_title='Logistic Regression - Exam Dataset',
                dataset_xlabel='Exam1 Score',
                dataset_ylabel='Exam2 Score',
                label=['Not Admitted', 'Admitted'],
                predict_func=predict_dataset1)

    dataset = 'resources/data/microchip_test_dataset_118_3.txt'
    for reg_param in (0, 1, 10, 100):
        run_dataset(dataset, print_data=True, normalize=True,
                    dataset_title='Logistic Regression - '
                                  'Microchip Test1 Dataset\n'
                                  f'Regularization Param Value - {reg_param}',
                    dataset_xlabel='Test1 Results',
                    dataset_ylabel='Test2 Results',
                    label=['Rejected', 'Accepted'],
                    predict_func=None,
                    add_features=True,
                    regularization_param=reg_param)

    dataset = 'resources/data/handwritten_digits_5000_400.mat'
    run_mcc_dataset(dataset, dataset_type='mat',
                    print_data=False, normalize=False,
                    dataset_title='Logistic Regression (Multi Class) -'
                                  'Handwritten Digits Dataset',
                    plot_image=True, image_size=(20, 20),
                    num_images_per_row=10, num_images_per_col=10)


if __name__ == '__main__':
    run()
