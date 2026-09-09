"""
3D plotter for one-junction trees with N=3.

Visualizes the tree in R^3 = (transverse R^2) x (time R).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from simple_junction.making_tree_simple import make_family_one, tree_at_one

def plot_tree_3d(
    tree,
    *,
    ax=None,
    show_vertices=True,
    vertex_labels=True,
    edge_color="blue",
    vertex_color="red",
    figsize=(6, 5),
):
    """
    Plot a one-junction tree in 3D.

    Parameters
    ----------
    tree : dict
        From tree_at with N=3.
    ax : Axes3D, optional
        Existing 3D axis; if None, a new figure/axis is created.
    show_vertices : bool
        Whether to draw vertex markers.
    vertex_labels : bool
        Whether to label vertices by name.
    edge_color : str
        Color of edges.
    vertex_color : str
        Color of vertex markers.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax
    """
    if tree["N"] != 3:
        raise ValueError("This plotter is for N=3 only.")

    vertices = tree["vertices"]
    edges = tree["edges"]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Plot edges
    for left_name, right_name in edges:
        left = np.asarray(vertices[left_name], dtype=float)
        right = np.asarray(vertices[right_name], dtype=float)

        xs = [left[0], right[0]]
        ys = [left[1], right[1]]
        zs = [left[2], right[2]]

        ax.plot(
            xs, ys, zs,
            color=edge_color,
            linewidth=2,
        )

    # Plot vertices
    if show_vertices:
        for name, pt in vertices.items():
            pt = np.asarray(pt, dtype=float)
            ax.scatter(
                pt[0], pt[1], pt[2],
                color=vertex_color,
                s=40,
            )

            if vertex_labels:
                ax.text(
                    pt[0], pt[1], pt[2],
                    f" {name}",
                    fontsize=9,
                )

    ax.set_xlabel("x1 (transverse)")
    ax.set_ylabel("x2 (transverse)")
    ax.set_zlabel("t (time)")

    ax.set_title("One-junction tree (N=3)")

    plt.tight_layout()
    return fig, ax

    plt.tight_layout()

    return fig, ax, tree
