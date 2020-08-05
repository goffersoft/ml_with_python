#! /usr/bin/python
# -*- coding:utf-8 -*-

"""gradient descent - module."""

import numpy as np

try:
    from ..transform import sigmoid
except (ImportError, ValueError):
    import os
    import sys
    import inspect
    filename = inspect.getfile(inspect.currentframe())
    abspath = os.path.abspath(filename)
    currentdir = os.path.dirname(abspath)
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from transform import sigmoid


def training_accuracy(feature_matrix, output_colvec, theta_colvec,
                      transform=sigmoid, threshold=0.5):
    """Compute Training Accuracy."""
    predicted_output = \
        transform(np.matmul(feature_matrix, theta_colvec)) >= threshold

    return np.mean((predicted_output == output_colvec) * 100)


if __name__ == '__main__':
    pass
