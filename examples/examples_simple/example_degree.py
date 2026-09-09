import numpy as np
import matplotlib.pyplot as plt

from simple_junction.making_tree_simple import make_family_one
from simple_junction.detector_simple import detector_boundary_degree

family = make_family_one(
    N=3,
    a=5,              # or any branching number
    angle_degrees=30.0,
    edge_length=1.0,
)

result = detector_boundary_degree(
    family,
    regular_value=None,   # defaults to center of transverse cube
    samples=61,           # grid resolution per face
    boundary_tol=1e-7,
)

print("Estimated degree:", result["degree"])
print("Top face (p_N=1):", result["top_face"]["relative_degree"])
print("Bottom face (p_N=0):", result["bottom_face"]["relative_degree"])
