from simple_junction.making_tree_simple import (
    make_family_one
)

from simple_junction.detector_simple import make_slice_detector_map

from detector_1D_plot import (
    detector_heatmap_N3,
)

import matplotlib.pyplot as plt

family = make_family_one(
    N=3,
    a=4,
    angle_degrees=30.0,
    edge_length=1.0,
)

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
