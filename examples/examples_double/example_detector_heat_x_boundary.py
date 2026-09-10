# We are plotting the detector map on the boundary of our cube C^{N+1}.
# Recall that an element in the cube is a pair (x,t) with x in C^N and t in I.
# On the boundary of C^{N+1}, we have either that x is on the boundary of C^N or t=0 or t=1.
# Right now we are looking at values on the boundary of C^N.

import matplotlib.pyplot as plt
import numpy as np
from double_junction.Tree_building import make_tree_family
from detector_heat_map import plot_detector_slice


family = make_tree_family(a=3, b=2, N=3)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

_, _, distances0, active0 = plot_detector_slice(
    family,
    target_type="a",
    fixed_axis=0,
    fixed_value=0.0,
    varying_axis=1,
    other_x_value=0.5,
    resolution=100,
    title="a-detector: x[0]=0, vary x[1] and t",
    ax=axes[0],
)

_, _, distances1, active1 = plot_detector_slice(
    family,
    target_type="a",
    fixed_axis=0,
    fixed_value=1.0,
    varying_axis=1,
    other_x_value=0.5,
    resolution=100,
    title="a-detector: x[0]=1, vary x[1] and t",
    ax=axes[1],
)
fig.tight_layout()
plt.savefig("results/boundary_x.png", dpi=200, bbox_inches="tight")
plt.show()