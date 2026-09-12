# We are testing out the detector on 2D slices for t=0 and t=1.

import numpy as np 
import matplotlib.pyplot as plt

from detector_heat_map_TIME import plot_detector_time_face
from double_junction.trees.Tree_building import make_tree_family
from double_junction.detector.Detector_maps import detect_a, detect_b

family = make_tree_family(a=3, b=2, N=3)

# a-detector
fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 6))

_, _, distances_a0, active_a0 = plot_detector_time_face(
    family,
    target_type="a",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.7,
    resolution=100,
    title="a-detector on the t=0 face",
    ax=axes_a[0],
)

_, _, distances_a1, active_a1 = plot_detector_time_face(
    family,
    target_type="a",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.2,
    resolution=100,
    title="a-detector on the t=1 face",
    ax=axes_a[1],
)

fig_a.tight_layout()
fig_a.savefig("results/a_t0_t1.png", dpi=200, bbox_inches="tight")
plt.show()

# Let's plot the b-detector faces in more detail: starting with face t=0
fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 10))

axes_b = axes_b.flatten()

_, _, distances_b00, active_b00 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.2,
    resolution=120,
    title="b-detector: t=0, fixed x=0.2",
    ax=axes_b[0],
)

_, _, distances_b01, active_b01 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.4,
    resolution=120,
    title="b-detector: t=0, fixed x=0.4",
    ax=axes_b[1],
)

_, _, distances_b10, active_b10 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.6,
    resolution=120,
    title="b-detector: t=0, fixed x=0.6",
    ax=axes_b[2],
)

_, _, distances_b11, active_b11 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.8,
    resolution=120,
    title="b-detector: t=0, fixed x=0.8",
    ax=axes_b[3],
)

fig_b.tight_layout()
fig_b.savefig("results/b_t0.png", dpi=200, bbox_inches="tight")
plt.show()

# One thing that's nice to do is plot the b-detector for many slices when t=1

fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 10))

# Flatten for easy indexing: axes_b[0,0], axes_b[0,1], axes_b[1,0], axes_b[1,1]
axes_b = axes_b.flatten()

_, _, distances_b00, active_b00 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.4,
    resolution=120,
    title="b-detector: t=1, fixed x=0.4",
    ax=axes_b[0],
)

_, _, distances_b01, active_b01 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.6,
    resolution=120,
    title="b-detector: t=1, fixed x=0.6",
    ax=axes_b[1],
)

_, _, distances_b10, active_b10 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.65,
    resolution=120,
    title="b-detector: t=1, fixed x=0.65",
    ax=axes_b[2],
)

_, _, distances_b11, active_b11 = plot_detector_time_face(
    family,
    target_type="b",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.75,
    resolution=120,
    title="b-detector: t=1, fixed x=0.75",
    ax=axes_b[3],
)

fig_b.tight_layout()
fig_b.savefig("results/b_t1.png", dpi=200, bbox_inches="tight")
plt.show()