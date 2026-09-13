import matplotlib.pyplot as plt

from examples.simple_junction.detector.detector_simple_plot import detector_heatmap_N3
from mathmodels.simple_junction.making_tree_simple import make_simple_combinatorial_tree

simple_combinatorial_tree = make_simple_combinatorial_tree(
    N=3,
    a=4,
    angle_degrees=30.0,
    edge_length=1.0,
)

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
