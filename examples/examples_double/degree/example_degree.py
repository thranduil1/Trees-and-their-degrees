import numpy as np
import matplotlib.pyplot as plt

from double_junction.degree.Degree_computations import degree_on_cube_map

def identity_map(x):
    return x.copy()

res = degree_on_cube_map(identity_map, N=3, samples=41)
print(res["relative_degree"])  # should be 1


def reflect_first_coordinate(x):
    x = np.asarray(x, dtype=float)
    x = x.copy()
    x[0] = 1.0 - x[0]
    return x

res = degree_on_cube_map(reflect_first_coordinate, N=3, samples = 41)
print(res["relative_degree"])  # should be -1