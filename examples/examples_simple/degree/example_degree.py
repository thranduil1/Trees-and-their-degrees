import numpy as np
import matplotlib.pyplot as plt

from simple_junction.degree_simple import degree_on_cube_map

def identity_map(x):
    return x.copy()

res = degree_on_cube_map(identity_map, d=1, samples=61)
print(res["relative_degree"])  # should be 1