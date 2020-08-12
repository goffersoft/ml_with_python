#! /usr/bin/python
# -*- coding:utf-8 -*-

"""Cost Functions - module."""

import numpy as np


def get_regularization_value(num_examples, theta_colvec,
                             regularization_param):
    """Compute the regularization value.

    Donot include theta-0 in the computation.
    """
    return (regularization_param *
            (np.sum(np.square(theta_colvec[1:num_examples])))) \
        / (2 * num_examples)


def mean_squared_error(hypothesis_colvec, output_colvec, num_examples,
                       regularization_param=0,
                       theta_colvec=None):
    """Compute mean squared error cost function."""
    cost_colvec = hypothesis_colvec - output_colvec

    if regularization_param:
        reg_value = get_regularization_value(num_examples,
                                             theta_colvec,
                                             regularization_param)

    return ((cost_colvec.transpose() @
             cost_colvec)/(2*num_examples))[0, 0] + reg_value


def cross_entropy(hypothesis_colvec, output_colvec, num_examples,
                  regularization_param=0,
                  theta_colvec=None):
    """Compute the cross entropy error cost function."""
    reg_value = 0
    if regularization_param:
        reg_value = get_regularization_value(num_examples,
                                             theta_colvec,
                                             regularization_param)

    return - ((output_colvec.transpose() @ np.log(hypothesis_colvec) +
               (1 - output_colvec).transpose() @
               np.log(1 - hypothesis_colvec))/num_examples)[0, 0] + reg_value


if __name__ == '__main__':
    pass
