# This file goes through the entire program at each step in the case of
# a simple junction.

import matplotlib.pyplot as plt
import numpy as np
from detector.detector_simple_plot import (
    detector_heatmap_N3,
)
from trees.plot_dim_one_tree import plot_tree_3d

from mathmodels.simple_junction.degree_simple import (
    detector_boundary_degree,
)
from mathmodels.simple_junction.making_tree_simple import (
    make_simple_combinatorial_tree,
    tree_at_one,
)

simple_combinatorial_tree = make_simple_combinatorial_tree(
    N=3,
    a=4,
    angle_degrees=30.0,
    edge_length=1.0,
)

p = np.array([0.2, 0.5, 0.6])  # some point in [0,1]^3
tree = tree_at_one(simple_combinatorial_tree, p)

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

## Now we plot the detector.

fig, ax, grid, values = detector_heatmap_N3(
    simple_combinatorial_tree,
    fixed_coord="p3",
    fixed_value=0.8,
    samples=101,
    cmap="viridis",
    slice_time=0.5,
)

plt.show()

fig.savefig(
    "results/detector_heatmap.png",
    dpi=200,
)

plt.close(fig)

## Now we compute the degree

result = detector_boundary_degree(
    simple_combinatorial_tree,
    regular_value=None,  # defaults to center of transverse cube
    samples=61,  # grid resolution per face
    boundary_tol=1e-7,
)

print("Estimated degree:", result["degree"])
print("Top face (p_N=1):", result["top_face"]["relative_degree"])
print("Bottom face (p_N=0):", result["bottom_face"]["relative_degree"])
