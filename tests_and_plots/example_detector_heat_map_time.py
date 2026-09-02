# We are testing out the detector on 2D slices for t=0 and t=1.
# Might be worth it to make a lot of these inside a single figure but we will see how it goes.

import numpy as np 
import matplotlib.pyplot as plt

from detector_heat_map_TIME import plot_detector_time_face
from Tree_building import make_tree_family
from Detector_maps import detect_a, detect_b

family = make_tree_family(a=3,b=2,N=3,)

plot_detector_time_face(
    family,
    target_type="a",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.5,
    resolution=150,
    title="a-detector on the t=0 face",
)

plt.show()

plt.savefig(
    "results/a_t0.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

plot_detector_time_face(
    family,
    target_type="a",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.2,
    resolution=150,
    title="a-detector on the t=1 face",
)

plt.show()

plt.savefig(
    "results/a_t1.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

plot_detector_time_face(
    family,
    target_type="b",
    t_value=0.0,
    varying_axes=(0, 1),
    fixed_x_value=0.5,
    resolution=150,
    title="b-detector on the t=0 face",
)

plt.show()

plt.savefig(
    "results/b_t0.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()

plot_detector_time_face(
    family,
    target_type="b",
    t_value=1.0,
    varying_axes=(0, 1),
    fixed_x_value=0.35,
    resolution=150,
    title="b-detector on the t=1 face",
)

plt.show()

plt.savefig(
    "results/b_t1.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()
