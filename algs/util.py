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

    mrows, ncols = np.shape(dataset)

    return dataset, mrows, ncols


def iterate_matrix(data_matrix, rstart, rend, col_range_list):
    """Get columns as a list for the specified rows in the matrix.

    rstart = first row
    rend = last row (not inlcuded)
    col_range_list = list or tuple of lists or tuples
    example : ((c1start, c1 end), ..., (cnstart, cnend))
    cnstart = first col
    cnend = last col (not included)
    """
    for row in range(rstart, rend):
        yield [[data_matrix[row, col] for col in range(cstart, cend)]
               for cstart, cend in col_range_list]


if __name__ == '__main__':
    DATASET = 'gd/resources/data/ex1data1.txt'
    print()
    print(f'First 10 Examples of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'X={i}, y={j}'
                    for i, j in
                    iterate_matrix(data, 0, 10, ((0, 1), (1, 2)))))

    DATASET = 'gd/resources/data/ex1data2.txt'
    print()
    print(f'First 10 Examples of the dataset: {DATASET}')
    data, _, _ = get_data_as_matrix(DATASET, Path(__file__))
    print('\n'.join(f'X={i}, y={j}'
                    for i, j in
                    iterate_matrix(data, 0, 10, ((0, 2), (2, 3)))))
