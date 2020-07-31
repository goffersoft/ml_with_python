#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Util module."""

from pathlib import Path
import numpy as np


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
            output = [row]
            output.extend([[data_matrix[row, col]
                            for col in range(col_start, col_end)]
                           for col_start, col_end in col_range_list])
            yield output


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


if __name__ == '__main__':
    DATASET = 'gd/resources/data/city_dataset_97_2.txt'
    print(f"{'*'*80}")
    print(f'First 10 rows of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((0, 10),), ((0, 1), (1, 2)))))

    DATASET = 'gd/resources/data/housing_dataset_47_3.txt'
    print(f"{'*'*80}")
    print(f'First 10 rows of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((0, 10),), ((0, 2), (2, 3)))))

    print(f"{'*'*80}")
    print(f'Rows 5 thru 10 of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'rownum={i} : feature_matrix_row={j}, : '
                    f'output_row={k}'
                    for i, j, k in
                    iterate_matrix(data, ((1, 2), (5, 10)), ((0, 2), (2, 3)))))

    print(f"{'*'*80}")
    print('Rows 1 and Rows 5 thru 10 of the '
          f'dataset-using enumerate: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'index={i} : rownum={j[0]} : feature_matrix_row={j[1]} : '
                    f'output_row={j[2]}'
                    for i, j in
                    enumerate(iterate_matrix(data,
                                             ((1, 2), (5, 10)),
                                             ((0, 2), (2, 3))))))
