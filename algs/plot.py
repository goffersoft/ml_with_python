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
                        title=None, projection=None,
                        fig=None, subplot=None,
                        subplot_nrows=1, subplot_ncols=1,
                        subplot_index=1):
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
    """
    if not fig:
        enable_interactive_mode()
        fig = plt.figure()

    if not subplot:
        params_list = {}
        if projection:
            params_list['projection'] = projection
        subplot = fig.add_subplot(subplot_nrows, subplot_ncols,
                                  subplot_index, **params_list)

    if xlabel:
        subplot.set_xlabel(xlabel)

    if ylabel:
        subplot.set_ylabel(ylabel)

    if title:
        subplot.set_title(title)

    return fig, subplot


def line_plot(xaxis_data, yaxis_data,
              xlabel=None, ylabel=None,
              title=None, label=None,
              marker=None, color=None,
              linewidth=None,
              markersize=None,
              fig=None, subplot=None,
              subplot_nrows=1, subplot_ncols=1,
              subplot_index=1):
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
    label = label for the plot within this sub-plot
    xaxis_data = 1-D array
    yaxis_data = 1-D array
    color = color to use for the plot
    marker = marker to use for the data points
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       None, fig, subplot,
                                       subplot_nrows, subplot_ncols,
                                       subplot_index)

    params_list = {}

    if marker:
        params_list['marker'] = marker

    if color is not None:
        params_list['color'] = color

    if linewidth:
        params_list['linewidth'] = linewidth

    if markersize:
        params_list['markersize'] = markersize

    if label:
        params_list['label'] = label

    subplot.plot(xaxis_data, yaxis_data, **params_list)

    if label:
        subplot.legend()

    return fig, subplot


def scatter_plot(xaxis_data, yaxis_data,
                 xlabel=None, ylabel=None,
                 title=None, label=None,
                 marker=None, color=None,
                 linewidths=None,
                 fig=None, subplot=None,
                 subplot_nrows=1, subplot_ncols=1,
                 subplot_index=1):
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
    legend = label for the plot within this sub-plot
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    color = color to use for the plot
    marker = marker to use for the data points
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       None, fig, subplot,
                                       subplot_nrows, subplot_ncols,
                                       subplot_index)

    params_list = {}

    if marker:
        params_list['marker'] = marker

    if color is not None:
        params_list['color'] = color

    if linewidths:
        params_list['linewidths'] = linewidths

    if label:
        params_list['label'] = label

    subplot.scatter(xaxis_data, yaxis_data, **params_list)

    if label:
        subplot.legend()

    return fig, subplot


def contour_plot(xaxis_data, yaxis_data,
                 compute_zaxis_data_func, levels=None,
                 xlabel=None, ylabel=None,
                 title=None, fig=None, subplot=None,
                 subplot_nrows=1, subplot_ncols=1,
                 subplot_index=1,
                 transpose_flag=True):
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
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       None, fig, subplot,
                                       subplot_nrows, subplot_ncols,
                                       subplot_index)

    nrows = np.size(xaxis_data)
    ncols = np.size(yaxis_data)

    zaxis_data = np.reshape(
        [compute_zaxis_data_func(xaxis_data[i], yaxis_data[j])
         for i in range(0, nrows) for j in range(0, ncols)],
        newshape=(nrows, ncols))

    params_list = {}
    if levels is not None:
        params_list['levels'] = levels

    # need to transpose cost matrix or else axis gets flipped
    if transpose_flag:
        zaxis_data = zaxis_data.transpose()

    subplot.contour(xaxis_data, yaxis_data,
                    zaxis_data, **params_list)

    return fig, subplot


def surface_plot(xaxis_data, yaxis_data, compute_zaxis_data_func,
                 xlabel=None, ylabel=None,
                 title=None, fig=None, subplot=None,
                 subplot_nrows=1, subplot_ncols=1,
                 subplot_index=1,
                 transpose_flag=True):
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
    xlabel = label for the X-Axis
    ylabel = label for the Y-Axis
    """
    fig, subplot = set_plot_attributes(xlabel, ylabel, title,
                                       '3d', fig, subplot,
                                       subplot_nrows, subplot_ncols,
                                       subplot_index)

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
    if transpose_flag:
        zaxis_data = zaxis_data.transpose()
    subplot.plot_surface(xaxis_data, yaxis_data, zaxis_data,
                         rcount=nrows, ccount=ncols)

    return fig, subplot


if __name__ == '__main__':
    pass
