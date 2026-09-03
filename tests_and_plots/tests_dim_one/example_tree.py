import numpy as np
import matplotlib.pyplot as plt

from making_tree import (
    make_family_one,
    tree_at_one,
)

from plot_dim_one_tree import plot_tree_in_cube


family = make_family_one(
    ambient_dimension=3,
    junction_time=0.5,
    angle_degrees=30.0,
)

x = np.array([0.5, 0.5])

fig, ax, tree = plot_tree_in_cube(
    family,
    x,
    elevation=20,
    azimuth=-60,
)

fig.savefig(
    "results/tree_plot_d1.png",
    dpi=200,
)

plt.close(fig)
