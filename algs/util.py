#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Util module."""

from pathlib import Path
import numpy as np

try:
    from .transform import identity
    from .transform import sigmoid
    from .cost import mean_squared_error
    from .cost import cross_entropy
except ImportError:
    from transform import identity
    from transform import sigmoid
    from cost import mean_squared_error
    from cost import cross_entropy


def pause(prompt=''):
    """Pause execution."""
    input(prompt)
    print()


def get_file_path(path, caller_path=None):
    """Determine if path is within the this directory.

    if input path is valid, return input path
    else
        1) get path of this module
        2) get its parent
        3) concatenate parent with 'path'
        4) if valid path, return this new path
    """
    node = Path(path).expanduser().resolve()
    if node.exists():
        return path, None

    if caller_path:
        node = caller_path.parent.joinpath(path)
    else:
        node = Path(__file__).parent.joinpath(path)

    if node.exists():
        return str(node), None

    return None, f'Invalid path name : {path}'


def get_data_as_matrix(path, caller_path=None, delimiter=','):
    """Read data from a file and return as a numpy ndarray."""
    file_path, err = \
        get_file_path(path, caller_path)
    if err:
        raise FileNotFoundError(err)

    dataset = np.loadtxt(file_path, delimiter=delimiter)

    nrows, ncols = np.shape(dataset)

    return dataset, nrows, ncols


def iterate_matrix(data_matrix,
                   row_range_list=None,
                   col_range_list=None):
    """Get columns as a list for the specified rows in the matrix.

    rstart = first row
    rend = last row (not inlcuded)
    col_range_list = list or tuple of lists or tuples
    example : ((c1start, c1 end), ..., (cnstart, cnend))
    cnstart = first col
    cnend = last col (not included)
    """
    if not row_range_list:
        row_range_list = ((0, np.shape(data_matrix)[0]))

    if not col_range_list:
        col_range_list = ((0, np.shape(data_matrix)[1]))

    for rstart, rend in row_range_list:
        for row in range(rstart, rend):
            print_output = [row]
            print_output.extend([[data_matrix[row, col]
                                  for col in range(col_start, col_end)]
                                 for col_start, col_end in col_range_list])
            yield print_output


def normalize_data(data_matrix):
    """Normalize data.

    data_matrix -> nrows X ncols
    Normlization involves
    1) Computing the mean (m) for each column -> 1 x ncols
    2) Computing the standard devistion (s) for each column -> 1 x ncols
    3) for elemnt(i, j) -> e(i, j) -> e(i, j) = (e(i, j) - m(0, j))/s(0, j)
    """
    ncols = np.shape(data_matrix)[1]

    mu_rowvec = np.zeros(shape=(1, ncols))
    sigma_rowvec = np.zeros(shape=(1, ncols))

    for col in range(0, ncols):
        mu_rowvec[0, col] = np.mean(data_matrix[:, col])
        sigma_rowvec[0, col] = np.std(data_matrix[:, col])
        data_matrix[:, col] = \
            (data_matrix[:, col] - mu_rowvec[0, col])/sigma_rowvec[0, col]

    return data_matrix, mu_rowvec, sigma_rowvec


def compute_cost(feature_matrix, output_colvec, theta_colvec,
                 transform_func, cost_func):
    """Compute cost.

    feature_matrix = (num_examples  x num_features + 1)
                     matrix - features
                     (first column in all ones)
    output_colvec = (num_examples x 1) col vector - actual cost
    num_examples = number of training samples
    num_features = number of features
    cost_func = (1/2num_examples)*(sum(h - output_colvec)**2)

    hypothesis_colvec(i) = \
        transform_func(theta_colvec[0]*feature_matrix[i, 0]) +
        transform_func(theta_colvec[1]*feature_matrix[i, 1]) +
        ...
        0 <= i < num_examples + 1
    """
#   Vectorized Implementation
    num_examples = np.shape(feature_matrix)[0]
    hypothesis_colvec = transform_func(np.matmul(feature_matrix, theta_colvec))

    return cost_func(hypothesis_colvec, output_colvec, num_examples)


def compute_cost_given_hypothesis(hypothesis_colvec,
                                  output_colvec, num_examples,
                                  cost_func):
    """Compute cost given hypothesis and the actual output.

    num_examples = number of training samples
    output_colvec = num_examples x 1 col vector - actual cost
    hypothesis_colvec = cost column vector - num_examples x 1 -
        cost associated with current values of theta
    """
    return cost_func(hypothesis_colvec, output_colvec, num_examples)


def add_features(feature1, feature2, degree):
    """Add additional feature columns in addition tot he 2 input features.

    feature 1(f1) is a row/column vector or a 1d array
    feature 2(f2) is a row/column vector or a 1d array
    m = size(feature1)
    degree 0 = 1's (m x 1)
    degree 1 = 1's, f1, f2
    degree 2 = 1's, f1, f2, f1^2, f2^2, f1*f2
    degree 3 = 1's, f1, f2, f1^2, f2^2, f1*f2, f1^3, f1^2*f2, f1*f2^2, f2^3
    ...
    """
    out = np.ones(shape=(np.size(feature1), 1))
    for deg in range(1, degree + 1):
        for index in range(deg):
            tmp = np.reshape(feature1**(deg - index) * feature2**index,
                             newshape=(np.size(feature1), 1))
            out = np.append(out, tmp, axis=1)
    return out


if __name__ == '__main__':
    DATASET = 'gd/resources/data/city_dataset_97_2.txt'
    print(f"{'*'*20}Testing Iterate Matrix Function{'*'*20}")
    print(f'First 10 rows of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((0, 10),), ((0, 1), (1, 2)))))

    DATASET = 'gd/resources/data/housing_dataset_47_3.txt'
    print(f"{'*'*20}Testing Iterate Matrix Function{'*'*20}")
    print(f'First 10 rows of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((0, 10),), ((0, 2), (2, 3)))))

    print(f"{'*'*20}Testing Iterate Matrix Function{'*'*20}")
    print(f'Rows 5 thru 10 of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((1, 2), (5, 10)), ((0, 2), (2, 3)))))

    print(f"{'*'*20}Testing Iterate Matrix Function{'*'*20}")
    print('Rows 1 and Rows 5 thru 10 of the '
          f'dataset-using enumerate: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'index={i} : rownum={j[0]} : feature_matrix_row={j[1]} : '
                    f'output_row={j[2]}'
                    for i, j in
                    enumerate(iterate_matrix(data,
                                             ((1, 2), (5, 10)),
                                             ((0, 2), (2, 3))))))

    DATASET = 'gd/resources/data/city_dataset_97_2.txt'
    print(f"{'*'*20}Testing Comput Cost Function{'*'*20}")
    print(f'DATASET : {DATASET}')
    data, num_rows, num_cols = get_data_as_matrix(DATASET, Path(__file__))

    output = data[:, num_cols - 1:num_cols]
    features = np.append(np.ones(shape=(num_rows, 1)),
                         data[:, 0:num_cols - 1], axis=1)
    cost = compute_cost(features, output, np.zeros(shape=(num_cols, 1)),
                        transform_func=identity, cost_func=mean_squared_error)
    print(f'cost(identity, mean_squared_error)={cost}')

    print(f"{'*'*20}Testing Comput Cost Function{'*'*20}")
    DATASET = 'lr/resources/data/exam_dataset_100_3.txt'
    data, num_rows, num_cols = get_data_as_matrix(DATASET, Path(__file__))
    print(f'First 10 rows of the dataset: {DATASET}')
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((0, 10),), ((0, 2), (2, 3)))))

    output = data[0:4, num_cols - 1:num_cols]
    features = np.append(np.ones(shape=(4, 1)),
                         data[0:4, 0:num_cols - 1], axis=1)
    cost = compute_cost(features, output, np.zeros(shape=(num_cols, 1)),
                        transform_func=sigmoid, cost_func=cross_entropy)
    print(f'cost(sigmoid, cross_entropy)={cost}')
