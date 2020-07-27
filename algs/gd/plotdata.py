#! /usr/bin/python
# -*- coding:utf-8 -*-

"""plotdata - module."""

from matplotlib import pyplot as plt
import numpy as np


def enable_interactive_mode():
    """Enable Interactive Mode."""
    plt.ion()


def disable_interactive_mode():
    """Disable Interactive Mode."""
    plt.ioff()


def is_interactive_mode_enabled():
    """Get interactive mode status."""
    return plt.isinteractive()


def close_plot(fig=None):
    """Close plot."""
    plt.close(fig)


def set_plot_attributes(xlabel=None, ylabel=None,
                        title=None, legend_label=None,
                        projection=None,
                        fig=None, subplot=None):
    """Set attributes for the specified plot.

    If an existing figure is passed in, it is used
    Else a new figure is created

    If an existing subplot is passed in, it is used
    Else a new subplot is created

    Initially let the function create the figure and subplot
    for you.

    Subsequent calls to this function should pass in the
    figure and subplot descriptors

    To Overlay several plots on one graph, pass
    the both the figure and subplot descriptors
    to this funtion.
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    title = title for the subplot
    legend_title = label for the plot within this sub-plot
    """
    if not fig:
        enable_interactive_mode()
        fig = plt.figure()

    if not subplot:
        params_list = {}
        if projection:
            params_list['projection'] = projection
        subplot = fig.add_subplot(**params_list)

    if xlabel:
        subplot.set_xlabel(xlabel)

    if ylabel:
        subplot.set_ylabel(ylabel)

    if legend_label:
        subplot.set_label(legend_label)

    if title:
        subplot.set_title(title)

    return fig, subplot


def line_plot(xaxis_data, yaxis_data,
              xlabel=None, ylabel=None,
              title=None, legend_label=None,
              marker=None, color=None,
              fig=None, subplot=None):
    """Line Plot within the figure.

    If an existing figure is passed in, it is used
    Else a new figure is created

    If an existing subplot is passed in, it is used
    Else a new subplot is created

    Initially let the function create the figure and subplot
    for you.

    Subsequent calls to this function should pass in the
    figure and subplot descriptors

    To Overlay several plots on one graph, pass
    the both the figure and subplot descriptors
    to this funtion.
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    title = title for the subplot
    legend_title = label for the plot within this sub-plot
    xaxis_data = 1-D array
    yaxis_data = 1-D array
    color = color to use for the plot
    marker = marker to use for the data points
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       legend_label, None, fig, subplot)

    params_list = {}

    if marker:
        params_list['marker'] = marker

    if color:
        params_list['color'] = color

    subplot.plot(xaxis_data, yaxis_data, **params_list)

    return fig, subplot


def scatter_plot(xaxis_data, yaxis_data,
                 xlabel=None, ylabel=None,
                 title=None, legend_label=None,
                 marker=None, color=None,
                 fig=None, subplot=None):
    """Scatter Plot within the figure.

    If an existing figure is passed in, it is used
    Else a new figure is created

    If an existing subplot is passed in, it is used
    Else a new subplot is created

    Initially let the function create the figure and subplot
    for you.

    Subsequent calls to this function should pass in the
    figure and subplot descriptors

    To Overlay several plots on one graph, pass
    the both the figure and subplot descriptors
    to this funtion.
    xaxis_data = 1-D array
    yaxis_data = 1-D array
    title = title for the subplot
    legend_title = label for the plot within this sub-plot
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    color = color to use for the plot
    marker = marker to use for the data points
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       legend_label, None, fig, subplot)

    params_list = {}

    if marker:
        params_list['marker'] = marker

    if color:
        params_list['color'] = color

    subplot.scatter(xaxis_data, yaxis_data, **params_list)

    return fig, subplot


def contour_plot(xaxis_data, yaxis_data,
                 compute_zaxis_data_func, levels=None,
                 xlabel=None, ylabel=None,
                 title=None, legend_label=None,
                 fig=None, subplot=None):
    """Plot input data as a contour plot.

    If an existing figure is passed in, it is used
    Else a new figure is created

    If an existing subplot is passed in, it is used
    Else a new subplot is created

    Initially let the function create the figure and subplot
    for you.

    Subsequent calls to this function should pass in the
    figure and subplot descriptors

    To Overlay several plots on one graph, pass
    the both the figure and subplot descriptors
    to this funtion.
    xaxis_data = 1-D array
    yaxis_data = 1-D array
    zaxis_data = 2-D array
    levels = number and positions of the contour lines/regions
    title = title for the subplot
    legend_title = label for the plot within this sub-plot
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       legend_label, None, fig, subplot)

    nrows = np.size(xaxis_data)
    ncols = np.size(yaxis_data)

    zaxis_data = np.reshape(
        [compute_zaxis_data_func(xaxis_data[i], yaxis_data[j])
         for i in range(0, nrows) for j in range(0, ncols)],
        newshape=(nrows, ncols))

    # need to transpose cost matrix or else axis gets flipped
    subplot.contour(xaxis_data, yaxis_data,
                    zaxis_data.transpose(), levels=levels)

    return fig, subplot


def surface_plot(xaxis_data, yaxis_data, compute_zaxis_data_func,
                 xlabel=None, ylabel=None,
                 title=None, legend_label=None,
                 fig=None, subplot=None):
    """Plot input data as a 3-D Surface plot.

    If an existing figure is passed in, it is used
    Else a new figure is created

    If an existing subplot is passed in, it is used
    Else a new subplot is created

    Initially let the function create the figure and subplot
    for you.

    Subsequent calls to this function should pass in the
    figure and subplot descriptors

    To Overlay several plots on one graph, pass
    the both the figure and subplot descriptors
    to this funtion.
    xaxis_data = 1-D array
    yaxis_data = 1-D array
    zaxis_data = 2-D array
    title = title for the subplot
    legend_title = label for the plot within this sub-plot
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       legend_label, '3d', fig, subplot)

    nrows = np.size(xaxis_data)
    ncols = np.size(yaxis_data)

    zaxis_data = np.zeros(shape=(nrows, ncols))

    xaxis_data, yaxis_data = np.meshgrid(xaxis_data, yaxis_data)

    for i in range(0, nrows):
        for j in range(0, ncols):
            zaxis_data[i, j] = \
                compute_zaxis_data_func(xaxis_data[i, j],
                                        yaxis_data[i, j])

    # need to transpose cost matrix or else axis gets flipped
    subplot.plot_surface(xaxis_data, yaxis_data, zaxis_data.transpose(),
                         rcount=nrows, ccount=ncols)

    return fig, subplot


if __name__ == '__main__':
    pass
