from dimension_one.making_tree import (
    make_family_one
)

from dimension_one.detector_one import make_slice_detector_map

from detector_1D_plot import (
    detector_heatmap,
    detector_vector_field,
)

import matplotlib.pyplot as plt


family = make_family_one(
    ambient_dimension=3,
    junction_time=0.5,
    angle_degrees=30.0,
)

# Heat map
fig1, ax1, grid, values = detector_heatmap(
    family,
    samples=101,
    cmap="viridis",
)

fig1.savefig(
    "results/detector_heatmap.png",
    dpi=200,
)

plt.close(fig1)

# Vector field
fig2, ax2, X, Y, U, V = detector_vector_field(
    family,
    samples=21,
    arrow_scale=0.15,
)

fig2.savefig(
    "results/detector_vector_field.png",
    dpi=200,
)

plt.close(fig2)
