"""A module for plotting functions and utilities.

This module contains a set of plotting functions and utilities to help with data
visualization in analysis notebooks. Plotting code can often be complex and take up a
lot of space. While visualizations are crucial for analysis, including all the plotting
code directly in a notebook can make it cluttered and hard to follow. To solve this
problem, we've created this separate module for plotting functions.
"""

import arcadia_pycolor as apc
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook"


def get_plotly_colorscale(name):
    gradient = next(
        iter(gradient for gradient in apc.gradients.all_gradients if gradient.name == name)
    )
    cmap = gradient.to_mpl_cmap()
    return [(i / 255.0, mcolors.rgb2hex(cmap(i / 255.0))) for i in range(256)]


def visualize_js_div_matrix(matrix_values: np.ndarray, js_div_zmax: float = 0.2):
    """Plots a heatmap of the Jenson-Shannon (JS) divergence.

    Args:
        matrix_values: The square matrix of JS-divergence values.
        title: A title for the plot.
        js_div_zmax:
            The JS-divergence value that corresponds to the highest color value. All
            JS-divergence values above this will be floored to this color.
    """
    # Compute the matrix values
    size = matrix_values.shape[0]
    indices = np.arange(size)

    # Create the figure and add the heatmap trace
    fig = go.Figure()
    heatmap_trace = go.Heatmap(
        z=np.swapaxes(matrix_values, 0, 1),
        x=indices,
        y=indices,
        colorscale=get_plotly_colorscale("sages"),
        colorbar=dict(title="JS-divergence"),
        showscale=True,
        zauto=False,
        zmax=js_div_zmax,
        zsmooth=False,
    )
    fig.add_trace(heatmap_trace)

    # Update the layout and axes
    fig.update_layout(
        xaxis_title="Residue j",
        yaxis_title="Residue i",
    )

    return fig


def compare_to_contact_map(js_div: np.ndarray, contact_map: np.ndarray, js_div_zmax: float = 0.1):
    """Plots a conjoined heatmap of the Jenson-Shannon (JS) divergence and the 3D contact map.

    Args:
        js_div: The square matrix of JS-divergence values.
        contact_map: The contact map.
        js_div_zmax:
            The JS-divergence value that corresponds to the highest color value. All
            JS-divergence values above this will be floored to this color.
    """
    upper_triangle_mask = np.triu(np.ones_like(contact_map, dtype=bool), k=0)
    lower_triangle_mask = np.tril(np.ones_like(js_div, dtype=bool), k=-1)

    z_lower = np.where(upper_triangle_mask, contact_map, np.nan)
    z_upper = np.where(lower_triangle_mask, js_div, np.nan)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_lower,
            colorscale=get_plotly_colorscale("verde"),
            colorbar=dict(title="Contact map", x=1.25),
            showscale=True,
            zauto=True,
            zsmooth=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=z_upper,
            colorscale=get_plotly_colorscale("sages"),
            colorbar=dict(title="JS-divergence", x=1.05),
            showscale=True,
            zauto=False,
            zmax=js_div_zmax,
            zsmooth=False,
        )
    )

    fig.update_layout(
        title="Heatmaps of JS-divergence and structural contact map",
        xaxis=dict(title="Position i", constrain="domain"),
        yaxis=dict(title="Position j", constrain="domain", scaleanchor="x", scaleratio=1),
        width=900,
        height=700,
    )

    fig.show()
