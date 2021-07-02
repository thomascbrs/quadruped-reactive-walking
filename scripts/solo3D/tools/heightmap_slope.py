from numpy import identity, zeros, ones, array

from sl1m.solver import solve_least_square
import numpy as np
import pickle 

class HeightmapSlope:
    def __init__(self, n_x, n_y, x_lim, y_lim , HEIGHTMAP):
        """
        :param n_x number of samples in x
        :param n_y number of samples in y
        :param x_lim bounds in x
        :param y_lim bounds in y
        """

        self.n_x = n_x
        self.n_y = n_y

        self.x = np.linspace(x_lim[0], x_lim[1], n_x)
        self.y = np.linspace(y_lim[0], y_lim[1], n_y)

        self.result = np.zeros((n_x, n_y , 3 ))
        self.roll = np.zeros((n_x, n_y))
        self.pitch = np.zeros((n_x, n_y))
        self.yaw = np.zeros((n_x, n_y))

        filehandler = open(HEIGHTMAP, 'rb')
        self.map = pickle.load(filehandler)
        self.FIT_SIZE_X = 0.3
        self.FIT_SIZE_Y = 0.3

    def save_pickle(self, filename):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def build(self):
        """
        Build the heightmap and return it
        For each slot in the grid create a vertical segment and check its collisions with the 
        affordances until one is found
        :param affordances list of affordances
        """
        print("Creating height map...")
        for i in range(self.n_x):
            if i % 10 == 0:
                print(100*i/self.n_x, " %")
            for j in range(self.n_y):
                q = np.array([self.x[i], self.y[j], 0.])
                result = self.compute_mean_surface(q)
                self.result[i,j,:] = np.array(result)
                self.roll[i, j] = -np.arctan2(result[1], 1.)
                self.pitch[i, j] = -np.arctan2(result[0], 1.)   
        print("Height map created")
        return 0             

    def map_index(self, x, y):
        """
        Get the i, j indices of a given position in the heightmap
        """
        i = np.searchsorted(self.x, x) - 1
        j = np.searchsorted(self.y, y) - 1
        return i, j

    def compute_mean_surface(self, q ):
        '''  Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - q (array 3x) : current [x,y,z] position in world frame 
        '''
        # Fit the map
        i_min, j_min = self.map.map_index(q[0] - self.FIT_SIZE_X, q[1] - self.FIT_SIZE_Y)
        i_max, j_max = self.map.map_index(q[0] + self.FIT_SIZE_X, q[1] + self.FIT_SIZE_Y)

        n_points = (i_max - i_min) * (j_max - j_min)
        A = np.zeros((n_points, 3))
        b = np.zeros(n_points)
        i_pb = 0
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                A[i_pb, :] = [self.map.x[i], self.map.y[j], 1.]
                b[i_pb] = self.map.z[i, j]
                i_pb += 1

        return solve_least_square(np.array(A), np.array(b)).x

