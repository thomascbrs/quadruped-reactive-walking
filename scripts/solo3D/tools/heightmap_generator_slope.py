import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from solo3D.tools.heightmap_slope import HeightmapSlope

# --------------------------------- PROBLEM DEFINITION ---------------------------------------------------------------

ENV_HEIGHTMAP = "/home/thomas_cbrs/Desktop/edin/quadruped-files/one_step_large/one_step_large.pickle"
ENV_HEIGHTMAP_SLOPE = "/home/thomas_cbrs/Desktop/edin/quadruped-files/one_step_large/one_step_large_slope.pickle"

N_X = 100
N_Y = 100
X_BOUNDS = [-4.0, 4.0]
Y_BOUNDS = [-4.0, 4.0]

# --------------------------------- MAIN ---------------------------------------------------------------
if __name__ == "__main__":

    heightmap = HeightmapSlope(N_X, N_Y, X_BOUNDS, Y_BOUNDS , ENV_HEIGHTMAP)
    heightmap.build()
    heightmap.save_pickle(ENV_HEIGHTMAP_SLOPE)
