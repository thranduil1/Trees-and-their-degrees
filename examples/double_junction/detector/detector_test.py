# Testing that the output of the detector on a given tree is coherent with the picture of the tree.

import numpy as np

from mathmodels.double_junction.detector.detector_maps import (
    detect_a,
    detect_b,
    detector_data,
)
from mathmodels.double_junction.trees.tree_building import (
    make_double_combinatorial_tree,
    make_double_geometric_tree_at,
)

double_combinatorial_tree = make_double_combinatorial_tree(
    a=3,
    b=2,
    N=3,
)

double_geometric_tree = make_double_geometric_tree_at(
    double_combinatorial_tree,
    x=np.array([0.0, 0.1, 0.1]),
    t=0.2,
)

a_output = detect_a(double_geometric_tree)
b_output = detect_b(double_geometric_tree)

print("a-detector:", a_output)
print("b-detector:", b_output)

info = detector_data(
    double_geometric_tree,
    target_type="a",
)

if info is None:
    print("No isolated a-junction was detected.")
else:
    print("Detected junction:", info["name"])
    print("Physical position:", info["position"])
    print("Cube radius:", info["cube_radius"])
    print("Normalized position:", info["normalized_position"])
