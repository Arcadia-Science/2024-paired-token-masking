"""A module for plotting functions and utilities.

This module contains a set of plotting functions and utilities to help with data
visualization in analysis notebooks. Plotting code can often be complex and take up a
lot of space. While visualizations are crucial for analysis, including all the plotting
code directly in a notebook can make it cluttered and hard to follow. To solve this
problem, we've created this separate module for plotting functions.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Plotly figures don't show up in sphinx-rendered notebooks using the default renderer,
# so we use the sphinx_gallery renderer. There may be other, more appropriate options.
# Some more details:
# https://sphinx-gallery.github.io/dev/auto_plotly_examples/plot_0_plotly.html
pio.renderers.default = "sphinx_gallery"


def visualize_js_div_matrix(matrix_values: np.ndarray, title="", js_div_zmax: float = 0.2):
    # Compute the matrix values
    size = matrix_values.shape[0]
    indices = np.arange(size)

    # Create the figure and add the heatmap trace
    fig = go.Figure()
    heatmap_trace = go.Heatmap(
        z=np.swapaxes(matrix_values, 0, 1),
        x=indices,
        y=indices,
        colorscale="Aggrnyl",
        colorbar=dict(title="JS-divergence"),
        showscale=True,
        zauto=False,
        zmax=js_div_zmax,
        zsmooth=False,
    )
    fig.add_trace(heatmap_trace)

    # Update the layout and axes
    fig.update_layout(
        title=title,
        xaxis_title="Residue j",
        yaxis_title="Residue i",
        width=900,
        height=800,
    )

    # Show the figure
    fig.show()


def compare_to_contact_map(js_div: np.ndarray, contact_map: np.ndarray, js_div_zmax: float = 0.1):
    upper_triangle_mask = np.triu(np.ones_like(contact_map, dtype=bool), k=0)
    lower_triangle_mask = np.tril(np.ones_like(js_div, dtype=bool), k=-1)

    z_lower = np.where(upper_triangle_mask, contact_map, np.nan)
    z_upper = np.where(lower_triangle_mask, js_div, np.nan)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_lower,
            colorscale="Teal_r",
            colorbar=dict(title="Contact map", x=1.15),
            showscale=True,
            zauto=True,
            zsmooth=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=z_upper,
            colorscale="Aggrnyl",
            colorbar=dict(title="JS-divergence", x=1.0),
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
        width=1000,
        height=800,
    )

    fig.show()
