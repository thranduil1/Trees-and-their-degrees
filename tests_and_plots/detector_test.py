# Testing that the output of the detector on a given tree is coherent with the picture of the tree.

import matplotlib.pyplot as plt
import numpy as np
from higher_dim.Tree_building import make_tree_family, tree_at
from tree_plots import plot_tree_3d
from higher_dim.Detector_maps import detect_a, detect_b

family = make_tree_family(
    a=3,
    b=2,
    N=3,
)

tree = tree_at(
    family,
    x=np.array([0.0, 0.1, 0.1]),
    t=0.2,
)

fig = plt.figure(figsize=(18, 6))

plot_tree_3d(
        tree,
        show_labels=True,)

plt.show()

a_output = detect_a(tree)
b_output = detect_b(tree)

print("a-detector:", a_output)
print("b-detector:", b_output)

info = detector_data(
    tree,
    target_type="a",
)

if info is None:
    print("No isolated a-junction was detected.")
else:
    print("Detected junction:", info["name"])
    print("Physical position:", info["position"])
    print("Cube radius:", info["cube_radius"])
    print("Normalized position:", info["normalized_position"])
