

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from making_tree import make_family_one, tree_at_one

def draw_cube(ax, alpha=0.15):
    """Draw the unit cube [0,1]^3."""
    # Vertices
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ])

    # 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        xs = [vertices[i, 0], vertices[j, 0]]
        ys = [vertices[i, 1], vertices[j, 1]]
        zs = [vertices[i, 2], vertices[j, 2]]

        ax.plot(
            xs, ys, zs,
            color="black",
            linewidth=1,
            alpha=0.6,
        )

    # Optional: translucent faces
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]

    for face in faces:
        face_verts = vertices[face]
        ax.add_collection3d(
            plt.Poly3DCollection(
                [face_verts],
                facecolor="lightgray",
                alpha=alpha,
                linewidth=0,
            )
        )


def draw_tree(tree, ax):
    """
    Draw a tree produced by tree_at inside the cube.
    """
    vertices = tree["vertices"]
    edges = tree["edges"]
    vertex_type = tree["vertex_type"]

    # Draw edges
    for left_name, right_name in edges:
        left = np.asarray(vertices[left_name])
        right = np.asarray(vertices[right_name])

        xs = [left[0], right[0]]
        ys = [left[1], right[1]]
        zs = [left[2], right[2]]

        ax.plot(
            xs, ys, zs,
            color="blue",
            linewidth=2,
        )

    # Draw vertices
    for name, point in vertices.items():
        point = np.asarray(point)

        kind = vertex_type.get(name, "unknown")

        if kind == "a":
            color = "red"
            marker = "o"
            size = 80
        elif kind == "root":
            color = "green"
            marker = "^"
            size = 80
        elif kind == "leaf":
            color = "purple"
            marker = "s"
            size = 60
        else:
            color = "black"
            marker = "o"
            size = 40

        ax.scatter(
            [point[0]],
            [point[1]],
            [point[2]],
            c=color,
            marker=marker,
            s=size,
            edgecolors="black",
        )


def plot_tree_in_cube(
    family,
    x,
    figsize=(7, 7),
    elevation=20,
    azimuth=-60,
):
    """
    Create a 3D plot of the tree for parameter x in the unit cube.
    """
    tree = tree_at_one(
        family,
        x,
    )

    fig = plt.figure(
        figsize=figsize,
    )
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    draw_cube(ax)
    draw_tree(tree, ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t (time)")

    ax.view_init(
        elev=elevation,
        azim=azimuth,
    )

    plt.tight_layout()

    return fig, ax, tree
