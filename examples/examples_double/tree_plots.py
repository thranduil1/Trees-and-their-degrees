## Some functions to draw the trees. Recall that the tree depends on two parameters x and t where x is in I^N and t is in I.
## N has to be at least 3, so we draw embedded trees into a 3-dimensional cube for different values of t.
## These trees look different from the ones in my explanatory file, but this is easier to implement
## and I don't think it should make a huge difference, although that's TBD.

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product

from double_junction.Tree_building import tree_at, make_tree_family, junctions

def plot_tree_3d(
    tree,
    *,
    ax=None,
    show_labels=True,
    show_cube=True,
    title=None,
    elev=20,
    azim=-60,
):
    """
    Draw a tree embedded in the unit cube [0, 1]^3.

    Coordinate convention:
        axis 0: a-direction
        axis 1: b-direction
        axis 2: time direction

    Parameters
    ----------
    tree:
        Output of tree_at(...), with N = 3.

    ax:
        Optional existing Matplotlib 3D axis.

    show_labels:
        Whether to label vertices by name.

    show_cube:
        Whether to draw the wireframe of [0, 1]^3.

    title:
        Optional plot title.

    elev, azim:
        Viewing angles for the 3D camera.

    Returns
    -------
    fig, ax
    """
    vertices = tree["vertices"]
    edges = tree["edges"]
    vertex_type = tree["vertex_type"]

    dimensions = {
        np.asarray(position).shape
        for position in vertices.values()
    }

    if dimensions != {(3,)}:
        raise ValueError(
            "plot_tree_3d requires every vertex to lie in R^3."
        )

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(
            111,
            projection="3d",
        )
    else:
        fig = ax.get_figure()

    colors = {
        "leaf": "black",
        "a": "tab:blue",
        "b": "tab:orange",
        "ab": "tab:purple",
        "root": "tab:red",
    }

    markers = {
        "leaf": "o",
        "a": "s",
        "b": "D",
        "ab": "P",
        "root": "*",
    }

    sizes = {
        "leaf": 35,
        "a": 75,
        "b": 75,
        "ab": 120,
        "root": 160,
    }

    # --------------------------------------------------------
    # Draw affine tree edges
    # --------------------------------------------------------
    for child, parent in edges:
        p = vertices[child]
        q = vertices[parent]

        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color="0.35",
            linewidth=1.5,
            zorder=1,
        )

    # --------------------------------------------------------
    # Draw vertices
    # --------------------------------------------------------
    for name, position in vertices.items():
        kind = vertex_type[name]

        ax.scatter(
            position[0],
            position[1],
            position[2],
            color=colors[kind],
            marker=markers[kind],
            s=sizes[kind],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        if show_labels:
            ax.text(
                position[0],
                position[1],
                position[2],
                f"  {name}",
                fontsize=8,
                color="black",
                zorder=4,
            )

    # --------------------------------------------------------
    # Draw the unit-cube wireframe
    # --------------------------------------------------------
    if show_cube:
        corners = np.array(
            list(product([0.0, 1.0], repeat=3))
        )

        for i, j in combinations(range(8), 2):
            difference = np.abs(
                corners[i] - corners[j]
            )

            # Two cube corners share an edge exactly when they
            # differ in precisely one coordinate.
            if np.sum(difference) == 1.0:
                p = corners[i]
                q = corners[j]

                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    [p[2], q[2]],
                    color="0.75",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=0,
                )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    ax.set_xlabel("a-direction")
    ax.set_ylabel("b-direction")
    ax.set_zlabel("time direction")

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)

    if title is not None:
        ax.set_title(title)

    return fig, ax
