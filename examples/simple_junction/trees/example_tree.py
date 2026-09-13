import matplotlib.pyplot as plt
import numpy as np
from plot_dim_one_tree import plot_tree_3d

from mathmodels.simple_junction.making_tree_simple import (
    make_simple_combinatorial_tree,
    make_simple_geometric_tree_at,
)

simple_combinatorial_tree = make_simple_combinatorial_tree(
    N=3,
    a=4,
    angle_degrees=30.0,
    edge_length=1.0,
)

p = np.array([0.2, 0.5, 0.6])  # some point in [0,1]^3
simple_geometric_tree = make_simple_geometric_tree_at(simple_combinatorial_tree, p)

fig, ax = plot_tree_3d(
    simple_geometric_tree,
    show_vertices=True,
    vertex_labels=True,
)

plt.show()
fig.savefig(
    "results/tree_plot_d1.png",
    dpi=200,
)

plt.close(fig)
