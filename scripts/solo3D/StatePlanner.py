import numpy as np
import pinocchio as pin
from sl1m.solver import solve_least_square
import pickle


class StatePlanner():

    def __init__(self, dt_mpc, T_mpc, h_ref, HEIGHTMAP, n_surface_configs, T_gait):

        self.h_ref = h_ref
        self.dt_mpc = dt_mpc
        self.T_mpc = T_mpc

        self.n_steps = round(T_mpc / dt_mpc)
        self.referenceStates = np.zeros((12, 1 + self.n_steps))
        self.dt_vector = np.linspace(0, T_mpc, self.n_steps + 1)

        self.FIT_SIZE_X = 0.3
        self.FIT_SIZE_Y = 0.15
        self.surface_equation = np.zeros(3)
        self.surface_point = np.zeros(3)

        self.configs = [np.zeros(7) for _ in range(n_surface_configs)]

        # TODO : 4 is actually the number of phases in a gait. We should use a phase length instead of T_gait/4
        self.dt_vector_config = np.linspace(T_gait/2, n_surface_configs*T_gait/2, num=n_surface_configs)

        filehandler = open(HEIGHTMAP, 'rb')
        self.map = pickle.load(filehandler)

    def computeReferenceStates(self, q,  v, o_vref, z_average, new_step=False):
        '''
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here x,y,z,w quaternion
        - v (6x1) : current v linear in world frame
        - o_vref (6x1) : vref in world frame
        - z_average (float) : z mean of CoM
        '''

        # c++ : ?????
        # Eigen::Quaterniond quat(q(6), q(3), q(4), q(5));  // w, x, y, z
        # RPY_ << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());
        quat = pin.Quaternion(q[3:7])
        RPY = pin.rpy.matrixToRpy(quat.toRotationMatrix())

        # Update the current state
        # self.referenceStates[:3, 0] = q[:3, 0] + pin.rpy.rpyToMatrix(RPY).dot(np.array([-0.04, 0., 0.]))
        self.referenceStates[:3, 0] = q[:3, 0]
        self.referenceStates[3:6, 0] = RPY
        self.referenceStates[6:9, 0] = v[:3, 0]
        self.referenceStates[9:12, 0] = v[3:6, 0]

        for i in range(1, self.n_steps + 1):
            # Displacement following the reference velocity compared to current position
            if o_vref[5, 0] < 10e-3:
                self.referenceStates[0, i] = o_vref[0, 0] * self.dt_vector[i]
                self.referenceStates[1, i] = o_vref[1, 0] * self.dt_vector[i]
            else:
                self.referenceStates[0, i] = (o_vref[0, 0] * np.sin(o_vref[5, 0] * self.dt_vector[i]) + o_vref[1, 0] * (np.cos(o_vref[5, 0] * self.dt_vector[i]) - 1.)) / o_vref[5, 0]
                self.referenceStates[1, i] = (o_vref[1, 0] * np.sin(o_vref[5, 0] * self.dt_vector[i]) - o_vref[0, 0] * (np.cos(o_vref[5, 0] * self.dt_vector[i]) - 1.)) / o_vref[5, 0]

            self.referenceStates[:2, i] += self.referenceStates[:2, 0]

            self.referenceStates[5, i] = o_vref[5, 0] * self.dt_vector[i]

            self.referenceStates[6, i] = o_vref[0, 0] * np.cos(self.referenceStates[5, i]) - o_vref[1, 0] * np.sin(self.referenceStates[5, i])
            self.referenceStates[7, i] = o_vref[0, 0] * np.sin(self.referenceStates[5, i]) + o_vref[1, 0] * np.cos(self.referenceStates[5, i])

            if o_vref[5, 0] < 10e-3:
                self.referenceStates[5, i] = 0.
            else:
                self.referenceStates[5, i] += RPY[2]

            self.referenceStates[11, i] = o_vref[5, 0]

        # Update according to heightmap
        # Get z
        result = self.compute_mean_surface(q[:3, 0])

        # Law for speed and position of roll / pitch
        rot_max = 0.3  # rad.s-1
        k_sp = 1
        r_des = -np.arctan2(result[1], 1.)
        p_des = -np.arctan2(result[0], 1.)

        self.referenceStates[3, 1:] = r_des
        self.referenceStates[4, 1:] = p_des
        
        if r_des != RPY[0]:
            self.referenceStates[9, 1:] = np.sign(r_des - RPY[0]) * min(rot_max, k_sp * abs(r_des - RPY[0]))
        else:
            self.referenceStates[9, 1:] = 0.

        if p_des != RPY[1]:
            self.referenceStates[10, 1:] = np.sign(p_des - RPY[1]) * min(rot_max, k_sp * abs(p_des - RPY[1]))
        else:
            self.referenceStates[10, 1:] = 0.

        # Get z
        for k in range(1, self.n_steps + 1):
            i, j = self.map.map_index(self.referenceStates[0, k], self.referenceStates[1, k])
            self.referenceStates[2, k] = result[0]*self.map.xv[i, j] + result[1]*self.map.yv[i, j] + result[2] + self.h_ref + 0.05
            if k == 1:
                self.surface_point = np.array(result[0]*self.map.xv[i, j] + result[1]*self.map.yv[i, j] + result[2])

        if new_step:
            self.compute_configurations(q, o_vref)        

    def compute_mean_surface(self, q):
        '''  Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - q (array 3x) : current [x,y,z] position in world frame 
        '''
        x = q[0]
        y = q[1]

        # Fit the map
        i_min, j_min = self.map.map_index(x - self.FIT_SIZE_X, y - self.FIT_SIZE_Y)
        i_max, j_max = self.map.map_index(x + self.FIT_SIZE_X, y + self.FIT_SIZE_Y)

        n_points = (i_max - i_min) * (j_max - j_min)
        A = np.zeros((n_points, 3))
        b = np.zeros(n_points)
        i_pb = 0
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                A[i_pb, :] = [self.map.xv[i, j], self.map.yv[i, j], 1.]
                b[i_pb] = self.map.zv[i, j]
                i_pb += 1
        return solve_least_square(np.array(A), np.array(b)).x

    def compute_configurations(self, q, o_vref):
        """  
        Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - q (array 6x) : current [x,y,z, r, p, y] position in world frame 
            - o_vref (array 6x) : cdesired velocity in world frame 
        """
        for k, config in enumerate(self.configs):
            config[:2] = q[:2, 0]
            if o_vref[5, 0] != 0:
                config[0] += (o_vref[0, 0] * np.sin(o_vref[5, 0] * self.dt_vector_config[k]) + o_vref[1, 0] * (np.cos(o_vref[5, 0] * self.dt_vector_config[k]) - 1.)) / o_vref[5, 0]
                config[1] += (o_vref[1, 0] * np.sin(o_vref[5, 0] * self.dt_vector_config[k]) - o_vref[0, 0] * (np.cos(o_vref[5, 0] * self.dt_vector_config[k]) - 1.)) / o_vref[5, 0]
            else:
                config[0] += o_vref[0, 0] * self.dt_vector_config[k]
                config[1] += o_vref[1, 0] * self.dt_vector_config[k]

            rpy = np.zeros(3)
            if o_vref[5, 0] != 0:
                rpy[2] = q[5] + o_vref[5, 0] * self.dt_vector_config[k]

            result = self.compute_mean_surface(config[:3])

            # Get z
            i, j = self.map.map_index(config[0], config[1])
            config[2] = result[0]*self.map.xv[i, j] + result[1]*self.map.yv[i, j] + result[2] + self.h_ref

            rpy[0] = -np.arctan2(result[1], 1.)
            rpy[1] = -np.arctan2(result[0], 1.)
            matrix = pin.rpy.rpyToMatrix(rpy)
            quat = pin.Quaternion(matrix)
            config[3:7] = [quat.x, quat.y, quat.z, quat.w]


    def getSurfaceHeightMap(self):
        '''  Return the equation of the surface : [a,b,c] such as ax + by -z +c = 0
        Returns :
            - surface equation (array 3x) 
        '''
        return self.surface_equation

    def getReferenceStates(self):
        return self.referenceStates
