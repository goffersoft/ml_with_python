#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Cost Functions - module."""

import numpy as np


def mean_squared_error(hypothesis_colvec, output_colvec, num_examples):
    """Compute mean squared error cost function."""
    cost_colvec = hypothesis_colvec - output_colvec
    return (np.matmul(cost_colvec.transpose(),
                      cost_colvec)/(2*num_examples))[0, 0]


def cross_entropy(hypothesis_colvec, output_colvec, num_examples):
    """Compute the cross entropy error cost function."""
    return - ((np.matmul(output_colvec.transpose(),
                         np.log(hypothesis_colvec)) +
               np.matmul((1 - output_colvec).transpose(),
                         np.log(1 - hypothesis_colvec)))/num_examples)[0, 0]


if __name__ == '__main__':
    pass
