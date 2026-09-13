## Some examples of trees plotted for different values.
import matplotlib.pyplot as plt
import numpy as np
from tree_plots import plot_tree_3d

from mathmodels.double_junction.trees.tree_building import (
    make_double_combinatorial_tree,
    make_double_geometric_tree_at,
)

double_combinatorial_tree = make_double_combinatorial_tree(
    a=3,
    b=2,
    N=3,
)

x = np.array([0.8, 0.8, 0.8])

times = [0.0, 0.5, 1.0]

titles = [
    r"$t=0$: columns grouped first",
    r"$t=1/2$: one ab-junction",
    r"$t=1$: rows grouped first",
]

fig = plt.figure(figsize=(18, 6))

for k, (t, title) in enumerate(
    zip(times, titles),
    start=1,
):
    ax = fig.add_subplot(
        1,
        3,
        k,
        projection="3d",
    )

    double_geometric_tree = make_double_geometric_tree_at(
        double_combinatorial_tree,
        x=x,
        t=t,
    )

    plot_tree_3d(
        double_geometric_tree,
        ax=ax,
        title=title,
        show_labels=True,
    )

plt.tight_layout()
plt.show()

plt.savefig(
    "results/example_tree.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()
