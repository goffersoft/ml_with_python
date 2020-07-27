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

    data = np.loadtxt(file_path, delimiter=delimiter)

    mrows, ncols = np.shape(data)

    return data, mrows, ncols


if __name__ == '__main__':
    pause()
    pause('Presss Any Key')
