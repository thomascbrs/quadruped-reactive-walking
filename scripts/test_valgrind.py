import libquadruped_reactive_walking as lqrw
from solo3D.SurfacePlannerWrapper import SurfacePlanner_Wrapper
import numpy as np


def main():
    params = lqrw.Params()

    gait = lqrw.Gait()
    gait.initialize(params)

    print(gait.currentGait)

    A = [[-1.0000000, 0.0000000, 0.0000000],
         [0.0000000, -1.0000000, 0.0000000],
         [0.0000000, 1.0000000, 0.0000000],
         [1.0000000, 0.0000000, 0.0000000],
         [0.0000000, 0.0000000, 1.0000000],
         [0.0000000, 0.0000000, -1.0000000]]

    b = [1.3946447, 0.9646447, 0.9646447, 0.5346446, 0.0000, 0.0000]

    vertices = [[-1.3946447276978748, 0.9646446609406726, 0.0], [-1.3946447276978748, -0.9646446609406726, 0.0],
                [0.5346445941834704, -0.9646446609406726, 0.0], [0.5346445941834704, 0.9646446609406726, 0.0]]

    floor_surface = lqrw.Surface(np.array(A), np.array(b), np.array(vertices))

    selected_surfaces = lqrw.SurfaceVector()
    for _ in range(4):
        selected_surfaces.append(floor_surface)

    footTrajectoryGenerator = lqrw.FootTrajectoryGeneratorBezier()
    footTrajectoryGenerator.initialize(params, gait, floor_surface, 0.06, 0.28, 0.06, 8, 10, 7)

    k = 0
    q = np.zeros(18)
    targetFootstep = np.zeros((3, 4))

    footTrajectoryGenerator.update(k, targetFootstep, selected_surfaces, q)
