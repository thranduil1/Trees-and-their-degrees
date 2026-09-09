# This file runs through the entire program for the double junction.
# It is a good introduction to the repository, as it contains examples for most major functions.
# We start by building a family and displaying a tree associated to it.

## Maybe this would be more readable as a notebook? ##

import numpy as np
import matplotlib.pyplot as plt 

from tree_plots import plot_tree_3d
from double_junction.Tree_building import tree_at, make_tree_family
from detector_heat_map import plot_detector_slice
from detector_heat_map_TIME import plot_detector_time_face
from double_junction.Detector_maps import detect_a, detect_b
from double_junction.make_detector_map import make_tree_detector_map
from double_junction.Degree_computations import tree_detector_boundary_degree
from double_junction.regular_values import sample_outputs, pick_deep_interior_point

family = make_tree_family(
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

    tree = tree_at(
        family,
        x=x,
        t=t,
    )

    plot_tree_3d(
        tree,
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

# Now we will build the detector and plot it on the boundary of our cube C^{N+1}.
# Recall that an element in the cube is a pair (x,t) with x in C^N and t in I.
# On the boundary of C^{N+1}, we have either that x is on the boundary of C^N or t=0 or t=1.
# Right now we are looking at values on the boundary of C^N.

# fig, ax, distances, active = plot_detector_slice(
#     family,
#     target_type="a",
#     fixed_axis=0,
#     fixed_value=0.0,
#     varying_axis=1,
#     other_x_value=0.5,
#     resolution=150,
#     title="a-detector: x[0]=0, vary x[1] and t",
# )

# plt.show()

# fig, ax, distances, active = plot_detector_slice(
#     family,
#     target_type="a",
#     fixed_axis=0,
#     fixed_value=1.0,
#     varying_axis=1,
#     other_x_value=0.5,
#     resolution=150,
#     title="a-detector: x[0]=1, vary x[1] and t",
# )

# plt.show()

# plt.savefig(
#     "results/boundary_x.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# plt.close()


# # We are now plotting out the detector on 2D slices for t=0 and t=1.

# plot_detector_time_face(
#     family,
#     target_type="a",
#     t_value=0.0,
#     varying_axes=(0, 1),
#     fixed_x_value=0.5,
#     resolution=150,
#     title="a-detector on the t=0 face",
# )

# plt.show()

# plt.savefig(
#     "results/a_t0.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# plt.close()

# plot_detector_time_face(
#     family,
#     target_type="a",
#     t_value=1.0,
#     varying_axes=(0, 1),
#     fixed_x_value=0.2,
#     resolution=150,
#     title="a-detector on the t=1 face",
# )

# plt.show()

# plt.savefig(
#     "results/a_t1.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# plt.close()

# plot_detector_time_face(
#     family,
#     target_type="b",
#     t_value=0.0,
#     varying_axes=(0, 1),
#     fixed_x_value=0.5,
#     resolution=150,
#     title="b-detector on the t=0 face",
# )

# plt.show()

# plt.savefig(
#     "results/b_t0.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# plt.close()

# plot_detector_time_face(
#     family,
#     target_type="b",
#     t_value=1.0,
#     varying_axes=(0, 1),
#     fixed_x_value=0.35,
#     resolution=150,
#     title="b-detector on the t=1 face",
# )

# plt.show()

# plt.savefig(
#     "results/b_t1.png",
#     dpi=200,
#     bbox_inches="tight",
# )

# plt.close()

# The important thing that we notice here is that the only faces which will contribute to the degree are t=0 and t=1, so 2 out of 2N+2.
# This is always true in our case, so we will use this simplification to compute the degree.

# Let us make the maps we want.

f_a = make_tree_detector_map(family, "a")
f_b = make_tree_detector_map(family, "b")

# First, we need to pick a good regular value.
# We do this for the a-detector first.

outs0 = sample_outputs(family, 0.0,f_a,  samples=61)
outs1 = sample_outputs(family, 1.0, f_a, samples=61)

y0_bottom = pick_deep_interior_point(outs0)
y0_top    = pick_deep_interior_point(outs1)

print("Regular value for t=0:", y0_bottom)
print("Regular value for t=1:", y0_top)

# 

result = tree_detector_boundary_degree(
    family,
    detector_type="a",
    regular_value_bottom=y0_bottom,
    regular_value_top=y0_top,
    samples=61,
)

print("Relative degree t=0:", result["bottom_face"]["relative_degree"])
print("Relative degree t=1:", result["top_face"]["relative_degree"])
print("Total degree:", result["degree"])
print("Preimages t=0:", len(result["bottom_face"]["preimages"]))
print("Preimages t=1:", len(result["top_face"]["preimages"]))
