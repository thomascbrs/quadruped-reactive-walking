import numpy as np
import pinocchio as pin
from solo3D.tools.optimisation import genCost, quadprog_solve_qp, to_least_square
from sl1m.solver import solve_least_square


class StatePlanner():

    def __init__(self, dt_mpc, T_mpc, h_ref, heightMap, n_surface_configs, T_gait):

        self.h_ref = h_ref
        self.dt_mpc = dt_mpc
        self.T_mpc = T_mpc

        self.n_steps = round(T_mpc / dt_mpc)
        self.referenceStates = np.zeros((12, 1 + self.n_steps))
        self.dt_vector = np.linspace(0, T_mpc, self.n_steps + 1)

        # heightMap
        self.heightMap = heightMap

        # Surface equation : [a,b,c] : z = ax + by + c
        # pitch  = - np.arctan2(a, 1.) ..
        self.FIT_SIZE_X = 0.3
        self.FIT_SIZE_Y = 0.15
        self.surface_equation = np.zeros(3)
        self.surface_point = np.zeros(3)
        
        # Boolean if the robot is inside or not the heightmap boundaries
        self.INSIDE_MAP = True

        self.configs = [np.zeros(7) for _ in range(n_surface_configs)]
        # TODO : 4 is actually the number of phases in a gait. We should use a phase length instead of T_gait/4
        self.dt_vector_config = np.linspace(T_gait/2, n_surface_configs*T_gait/2, num=n_surface_configs)

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
        self.referenceStates[:3, 0] = q[:3, 0]
        self.referenceStates[3:6, 0] = RPY
        self.referenceStates[6:9, 0] = v[:3, 0]
        self.referenceStates[9:12, 0] = v[3:6, 0]

        self.referenceStates[:3,0] += pin.rpy.rpyToMatrix(RPY).dot(np.array([-0.04,0.,0.])) 


        for i in range(1, self.n_steps + 1):
            # Displacement following the reference velocity compared to current position
            if o_vref[5,0] < 10e-3 : 
                self.referenceStates[0, i] = o_vref[0, 0] * self.dt_vector[i]
                self.referenceStates[1, i] = o_vref[1, 0] * self.dt_vector[i]
            else : 
                self.referenceStates[0, i] = (o_vref[0, 0] * np.sin(o_vref[5, 0] * self.dt_vector[i]) + o_vref[1, 0] * (np.cos(o_vref[5, 0] * self.dt_vector[i]) - 1.)) / o_vref[5, 0]
                self.referenceStates[1, i] = (o_vref[1, 0] * np.sin(o_vref[5, 0] * self.dt_vector[i]) - o_vref[0, 0] * (np.cos(o_vref[5, 0] * self.dt_vector[i]) - 1.)) / o_vref[5, 0]
            
            
            self.referenceStates[0:2,i] += self.referenceStates[0:2,0] 
            self.referenceStates[2,i] = self.h_ref +  z_average

            self.referenceStates[5, i] = o_vref[5, 0] * self.dt_vector[i]

            self.referenceStates[6, i] = o_vref[0, 0] * np.cos(self.referenceStates[5, i]) - o_vref[1, 0] * np.sin(self.referenceStates[5, i])
            self.referenceStates[7, i] = o_vref[0, 0] * np.sin(self.referenceStates[5, i]) + o_vref[1, 0] * np.cos(self.referenceStates[5, i])

            if o_vref[5,0] < 10e-3 : 
                self.referenceStates[5,i] = 0.
            else : 
                self.referenceStates[5,i] += RPY[2]

            self.referenceStates[11,i] = o_vref[5,0] 

        
        # self.referenceStates[:3,0] += pin.rpy.rpyToMatrix(RPY).dot(np.array([-0.034,0.,0.])) 
        # self.referenceStates[:3,0] += np.array([-0.04,0.,0.])

        ## Update according to heightmap
        
        Z_OFFSET = self.h_ref + 0.0
        
        if self.INSIDE_MAP :
            # Inside the boundaries of the heightmap

            # Law for speed and position of roll / pitch
            rot_max = 0.3  # rad.s-1
            r_des = - 1.*np.arctan2(self.surface_equation[1], 1.)
            p_des = - 1.*np.arctan2(self.surface_equation[0], 1.)
            k_sp = 100 
            epsilon = 0.01

            if r_des != 0 and abs(r_des - RPY[0]) > epsilon:
                self.referenceStates[3, 1:] = r_des
                self.referenceStates[9, 1:] = np.sign(r_des - RPY[0])*min(rot_max, k_sp*abs(r_des - RPY[0]))
            else:
                self.referenceStates[3, 1:] = 0.
                self.referenceStates[9, 1:] = 0.

            if p_des != 0 and abs(p_des - RPY[1]) > epsilon:
                self.referenceStates[4, 1:] = p_des
                self.referenceStates[10, 1:] = np.sign(p_des - RPY[1])*min(rot_max, k_sp*abs(p_des - RPY[1]))
            else:
                self.referenceStates[4, 1:] = 0.
                self.referenceStates[10, 1:] = 0.

            # Get z
            _, i, j = self.heightMap.find_nearest(q[0, 0:1], q[1, 0:1])

            for k in range(self.n_steps):
                _, i, j = self.heightMap.find_nearest(self.referenceStates[0, k+1], self.referenceStates[1, k+1])
                self.referenceStates[2, k + 1] = self.surface_equation[0]*self.heightMap.x[i] + self.surface_equation[1]*self.heightMap.y[j] + self.surface_equation[2] + self.h_ref

                if k == 0:
                    self.surface_point = np.array([self.heightMap.x[i], self.heightMap.y[j], self.surface_equation[0]*self.heightMap.x[i] + self.surface_equation[1]*self.heightMap.y[j] + self.surface_equation[2]])

        if not new_step:
            return

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
                rpy[2] = RPY[2] + o_vref[5, 0] * self.dt_vector_config[k]
            
            isInside_min, i_min, j_min = self.heightMap.find_nearest(config[0] - self.FIT_SIZE_X, config[1] - self.FIT_SIZE_Y)
            isInside_max, i_max, j_max = self.heightMap.find_nearest(config[0] + self.FIT_SIZE_X, config[1] + self.FIT_SIZE_Y)

            if isInside_min and isInside_max:
                n_points = (i_max - i_min) * (j_max - j_min)
                A = np.zeros((n_points, 3))
                b = np.zeros(n_points)
                i_pb = 0
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        A[i_pb, :] = [self.heightMap.x[i], self.heightMap.y[j], 1.]
                        b[i_pb] = self.heightMap.heightMap[i, j][0]
                        i_pb += 1
                result = solve_least_square(np.array(A), np.array(b))

                # Get z
                _, i, j = self.heightMap.find_nearest(config[0], config[1])
                config[2] = result.x[0]*self.heightMap.x[i] + result.x[1]*self.heightMap.y[j] + result.x[2] + self.h_ref


                rpy[0] = -np.arctan2(result.x[1], 1.)
                rpy[1] = -np.arctan2(result.x[0], 1.)
                matrix = pin.rpy.rpyToMatrix(rpy)
                quat = pin.Quaternion(matrix)
                config[3:7] = [quat.x, quat.y, quat.z, quat.w]
        print("configs : " , self.configs)

        return 0

    def computeSurfaceHeightMap(self , pos ) :
        '''  Compute the surface equation to fit the heightmap, [a,b,c] such as ax + by -z +c = 0
        Args :
            - pos (array 3x) : current [x,y,z] position in world frame 
        '''

        isInside_min , i_min, j_min = self.heightMap.find_nearest(pos[0] - self.FIT_SIZE_X , pos[1]  - self.FIT_SIZE_Y )
        isInside_max , i_max, j_max = self.heightMap.find_nearest(pos[0]  + self.FIT_SIZE_X , pos[1]  + self.FIT_SIZE_Y )

        if isInside_min and isInside_max :

            #Update boolean 
            self.INSIDE_MAP = True

            nx = 12
            ny = 12
            X = np.linspace(i_min, i_max , nx).astype(int) 
            Y = np.linspace(j_min, j_max , ny).astype(int) 

            n_points = nx * ny 
            A = np.zeros((n_points, 3))
            b = np.zeros(n_points)
            i_pb = 0
            for i in X:
                for j in Y:
                
                    A[i_pb, :] = [self.heightMap.x[i], self.heightMap.y[j], 1.]
                    b[i_pb] = self.heightMap.heightMap[i, j][0]
                    i_pb += 1

            A , b = to_least_square(A , b )
            result = quadprog_solve_qp(A,b)

            # ax + by + c - z = 0
            self.surface_equation = np.array([result[0] , result[1] , result[2]])

        else : 
            #Update boolean 
            self.INSIDE_MAP = False

            self.surface_equation = np.zeros(3)

        return 0

    def getSurfaceHeightMap(self) :
        '''  Return the equation of the surface : [a,b,c] such as ax + by -z +c = 0
        Returns :
            - surface equation (array 3x) 
        '''
        return self.surface_equation

    def getReferenceStates(self) :

        return self.referenceStates
