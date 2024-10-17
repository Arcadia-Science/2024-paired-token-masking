import numpy as np
import pandas as pd
import plotly.graph_objects as go


def get_js_div_matrix(df: pd.DataFrame) -> np.ndarray:
    indices = sorted(set(df["position_i"]).union(df["position_j"]))
    matrix = pd.DataFrame(index=indices, columns=indices)  # type: ignore

    for _, row in df.iterrows():
        matrix.at[row.position_i, row.position_j] = row.js_div_avg
        matrix.at[row.position_j, row.position_i] = row.js_div_avg

    np.fill_diagonal(matrix.values, np.nan)

    matrix_values = matrix.to_numpy()
    return matrix_values.astype(np.float32)


def visualize_js_div_matrix(df: pd.DataFrame, log: bool, epsilon=1e-4, title="", zmax: float = 0.2):
    # Compute the matrix values
    matrix_values = get_js_div_matrix(df)
    size = matrix_values.shape[0]
    indices = np.arange(size)

    # Compute z_values for the heatmap
    if log:
        z_values = np.log(matrix_values + epsilon)
    else:
        z_values = matrix_values

    # Create the figure and add the heatmap trace
    fig = go.Figure()
    heatmap_trace = go.Heatmap(
        z=np.swapaxes(z_values, 0, 1),
        x=indices,
        y=indices,
        colorscale="Aggrnyl",
        colorbar=dict(title="JS-divergence"),
        showscale=True,
        zauto=False,
        zmax=zmax,
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
