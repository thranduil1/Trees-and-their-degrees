import numpy as np
import matplotlib.pyplot as plt

from double_junction.Degree_computations import degree_on_cube_map

def identity_map(x):
    return x.copy()

res = degree_on_cube_map(identity_map, N=3, samples=41)
print(res["relative_degree"])  # should be 1