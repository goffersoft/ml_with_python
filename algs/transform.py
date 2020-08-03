#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Transforms module."""

import numpy as np


def identity(zzz):
    """Apply identity function to data.

    zzz = scalar or a mxn dimensional array.
    returns a scala or a mxn dimensional array
    where each element is the output if the
    identity function -> g(zzz) = zzz
    """
    return zzz


def sigmoid(zzz):
    """Apply the Sigmoid Function to zzz.

    zzz = scalar or a mxn dimensional array.
    returns a scala or a mxn dimensional array
    where each element is the output if the
    sigmoid function -> g(zzz) = 1/(1 + e^(-zzz)
    """
    return 1.0/(1.0 + np.exp(-zzz))


if __name__ == '__main__':
    print(f"{'*'*20}Testing Sigmoid Function{'*'*20}")
    print(f'sigmoid(0)={sigmoid(0)}')
    print(f'sigmoid(100)={sigmoid(100)}')
    print(f'sigmoid(-100)={sigmoid(-100)}')
    print('sigmoid([-100, 0, 100]_1x3)=')
    print(f'{sigmoid(np.reshape([-100, 0, 100], newshape=(1,3)))}')
    print('sigmoid([-100, 0, 100]_3x3)=')
    print(f'{sigmoid(np.reshape([-100, 0, 100]*3, newshape=(3,3)))}')
