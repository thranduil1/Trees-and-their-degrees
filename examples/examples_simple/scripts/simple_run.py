# This file goes through the entire program at each step in the case of 
# a simple junction.

import numpy as np
import matplotlib.pyplot as plt

from simple_junction.making_tree_simple import (
    make_family_one,
    tree_at_one,
)
from trees.plot_dim_one_tree import plot_tree_3d
from simple_junction.detector_simple import make_slice_detector_map
from examples.examples_simple.detector.detector_1D_plot import (
    detector_heatmap_N3,
)
from simple_junction.detector_simple import detector_boundary_degree



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

## Now we plot the detector.

fig, ax, grid, values = detector_heatmap_N3(
    family,
    fixed_coord="p3",
    fixed_value=0.8,
    samples=101,
    cmap="viridis",
    slice_time = 0.5,
)

plt.show()

fig.savefig(
    "results/detector_heatmap.png",
    dpi=200,
)

plt.close(fig)

## Now we compute the degree

result = detector_boundary_degree(
    family,
    regular_value=None,   # defaults to center of transverse cube
    samples=61,           # grid resolution per face
    boundary_tol=1e-7,
)

print("Estimated degree:", result["degree"])
print("Top face (p_N=1):", result["top_face"]["relative_degree"])
print("Bottom face (p_N=0):", result["bottom_face"]["relative_degree"])
