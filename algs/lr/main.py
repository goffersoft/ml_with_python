#! /usr/bin/python
# -*- coding:utf-8 -*-
"""Main - Logistic Regression - module."""

from pathlib import Path
import numpy as np

try:
    from .. import util
    from ..plot import scatter_plot
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
    from plot import scatter_plot


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


def run_dataset(dataset_name, dataset_title,
                dataset_xlabel='X', dataset_ylabel='Y',
                normalize=False, print_data=False,
                label=None,
                predict_func=None):
    """Run Logistic Regression."""
    _, features, output, \
        _, feature_count, _, _ = \
        load_data(dataset_name, normalize, print_data)

    _, _ = \
        plot_dataset(features, output, feature_count,
                     dataset_title, dataset_xlabel,
                     dataset_ylabel, label)


def run():
    """Run Logistic Regression against various datasets."""
    dataset = 'resources/data/exam_dataset_100_3.txt'
    run_dataset(dataset, print_data=True,
                dataset_title='Logistic Regression - Exam Dataset',
                dataset_xlabel='Exam1 Score',
                dataset_ylabel='Exam2 Score',
                label=['Admitted', 'Not Admitted'],
                predict_func=None)

    dataset = 'resources/data/microchip_test_dataset_118_3.txt'
    run_dataset(dataset, print_data=True,
                dataset_title='Logistic Regression - Microchip Test1 Dataset',
                dataset_xlabel='Test1 Results',
                dataset_ylabel='Test2 Results',
                label=['Accepted', 'Rejected'],
                predict_func=None)


if __name__ == '__main__':
    run()
