import numpy as np
import pinocchio as pin
from solo3D.tools.optimisation import genCost , quadprog_solve_qp , to_least_square

class StatePlanner() :

    def __init__(self , dt_mpc, T_mpc, h_ref , heightMap):
        
        self.h_ref = h_ref
        self.dt_mpc = dt_mpc
        self.T_mpc = T_mpc

        self.n_steps = round(T_mpc / dt_mpc)
        self.referenceStates = np.zeros((12, 1 + self.n_steps))
        self.dt_vector = np.linspace( dt_mpc, T_mpc , self.n_steps)

        # heightMap
        self.heightMap = heightMap 

    
    def computeReferenceStates(self , q,  v, o_vref, z_average) : 
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
        self.referenceStates[:3,0] = q[:3 , 0]
        self.referenceStates[3:6 , 0] = RPY
        self.referenceStates[6:9 , 0] = v[:3 ,0]
        self.referenceStates[9:12 , 0] = v[3:6 ,0]

        for i in range(self.n_steps) : 
             # Displacement following the reference velocity compared to current position
            if o_vref[5,0] != 0 :               
                self.referenceStates[0,i+1] = (o_vref[0,0] * np.sin(o_vref[5,0] * self.dt_vector[i] ) + o_vref[1,0] * (np.cos(o_vref[5,0] * self.dt_vector[i]) - 1.)   ) / o_vref[5,0]
                self.referenceStates[1,i+1] = (o_vref[1,0] * np.sin(o_vref[5,0] * self.dt_vector[i] ) - o_vref[0,0] * (np.sin(o_vref[5,0] * self.dt_vector[i]) - 1.)   ) / o_vref[5,0]
            else :
                
                self.referenceStates[0,i+1] = o_vref[0,0]* self.dt_vector[i]
                self.referenceStates[1,i+1] = o_vref[1,0]* self.dt_vector[i]
            
            self.referenceStates[0:2,i+1] += self.referenceStates[0:2,0] 
            self.referenceStates[2,i+1] = self.h_ref +  z_average

            self.referenceStates[5,i+1] = o_vref[5,0] * self.dt_vector[i]


            self.referenceStates[6,i+1] =  o_vref[0,0] * np.cos(self.referenceStates[5,1 + i]) - o_vref[1,0] * np.sin(self.referenceStates[5,1 + i])
            self.referenceStates[7,i+1] =  o_vref[0,0] * np.sin(self.referenceStates[5,1 + i]) + o_vref[1,0] * np.cos(self.referenceStates[5,1 + i])

            #self.referenceStates[5,i+1] += RPY[2]

            self.referenceStates[11,i+1] = o_vref[5,0] 

        

        ## Update according to heightmap
        FIT_SIZE_X = 0.3
        FIT_SIZE_Y = 0.1
        Z_OFFSET = self.h_ref 

        isInside , i_min, j_min = self.heightMap.find_nearest(q[0, 0:1] - 0.2*FIT_SIZE_X , q[1, 0:1]  - FIT_SIZE_Y )
        isInside , i_max, j_max = self.heightMap.find_nearest(q[0, 0:1]  + FIT_SIZE_X , q[1, 0:1]  + FIT_SIZE_Y )

        nx = 20
        ny = 2
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

        # Law for speed and position of roll / pitch
        # Speed = min ( - K (P_des - P_cur) , rot_max )
        rot_max = 0.3  # rad.s-1
        r_des = - 1.05*np.arctan2(result[1], 1.)
        p_des = - 1.05*np.arctan2(result[0], 1.)
        k_sp = 100 
        epsilon = 0.01

        # print("Desired pitch angle [rad , deg] : " , p_des , " ; " , 57*p_des)

        if r_des != 0 and abs(r_des - RPY[0]) > epsilon :
            self.referenceStates[3,1:] = r_des
            self.referenceStates[9,1:] = np.sign(r_des - RPY[0] )*min( rot_max  , k_sp*abs(r_des - RPY[0])  )
        else : 
            self.referenceStates[3,1:] = 0.
            self.referenceStates[9,1:] = 0.

        if p_des != 0 and abs(p_des - RPY[1]) > epsilon : 
            self.referenceStates[4,1:] = p_des
            self.referenceStates[10,1:] = np.sign(p_des - RPY[1] )*min( rot_max  , k_sp*abs(p_des - RPY[1])  )
        else : 
            self.referenceStates[4,1:] = 0.
            self.referenceStates[10,1:] = 0.


        # Get z
        isInside , i, j = self.heightMap.find_nearest(q[0, 0:1] , q[1, 0:1])

        # if p_des != 0 :
        #     Z_OFFSET = self.h_ref - abs(p_des*0.05)
            
        for k in range(self.n_steps) :
            isInside , i, j = self.heightMap.find_nearest(self.referenceStates[0,k+1] , self.referenceStates[1,k+1])
            self.referenceStates[2 , 1:] = result[0]*self.heightMap.x[i] + result[1]*self.heightMap.y[j] + result[2] + Z_OFFSET      
        

        return 0

    def getReferenceStates(self) :

        return self.referenceStates

    

    
      





