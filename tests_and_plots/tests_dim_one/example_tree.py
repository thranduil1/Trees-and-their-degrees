import numpy as np
import matplotlib.pyplot as plt

from dimension_one.making_tree import (
    make_family_one,
    tree_at_one,
)

from plot_dim_one_tree import plot_tree_3d

family = make_family_one(
    N=3,
    a=4,
    angle_degrees=30.0,
    edge_length=1.0,
)

p = np.array([0.2, 0.5, 0.6])  # some point in [0,1]^3
tree = tree_at_one(family, p)

fig, ax = plot_tree_3d(
    tree,
    show_vertices=True,
    vertex_labels=True,
)

plt.show()
fig.savefig(
    "results/tree_plot_d1.png",
    dpi=200,
)

plt.close(fig)
