import numpy as np 
import utils_mpc
import math
from solo3D.tools.optimisation import genCost , quadprog_solve_qp , to_least_square

class FootStepPlannerQP:
    ''' Python class to compute the target foot step and the ref state using QP optimisation 
    to avoid edges of surface.
    '''

    def __init__(self , dt , dt_wbc , T_gait , h_ref, k_mpc , heightMap) :

        # Time step of the contact sequence
        self.dt = dt  # dt mpc
        self.dt_wbc = dt_wbc 

        self.k_mpc = k_mpc

        # Gait duration
        self.T_gait = T_gait

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(self.T_gait/self.dt)

        self.dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)
        self.is_static = False  # Flag for static gait
        self.q_static = np.zeros((19, 1))
        self.RPY_static = np.zeros((3, 1))
        self.RPY = np.zeros((3, 1))

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])
        
        # Feet matrix
        self.o_feet_contact = self.shoulders.ravel(order='F').copy()

        # To store the result of the compute_next_footstep function
        self.next_footstep = np.zeros((3, 4))

        self.h_ref = h_ref

        self.flag_rotation_command = int(0)

        # Predefined matrices for compute_footstep function
        self.R = np.zeros((3, 3, self.gait.shape[0]))
        self.R[2, 2, :] = 1.0

        self.b_v_cur = np.zeros((3,1))
        self.b_v_ref = np.zeros((3,1))

        self.k_feedback = 0.03
        self.g = 9.81
        self.L = 0.155

        # heightMap
        self.heightMap = heightMap   

        # Coefficients QP
        self.weight_vref = 0.06
        self.weight_alpha = 1.

        # Store optim v_ref
        self.v_ref_qp =  np.zeros((6,1))

        # Store surface selected for the current swing phase
        self.surface_selected = [None , None , None , None ]
        self.feet = []
        self.t0s = np.zeros((4, ))
        self.t_remaining = 0.
        self.t_stance = np.zeros((4, ))  # Total duration of current stance phase for each foot
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot



        

    def getRefStates(self, q, v, vref, z_average ):
        """Compute the reference trajectory of the CoM for each time step of the
        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
        linear velocity and angular velocity vertically stacked. The first column contains
        the current state while the remaining N columns contains the desired future states.

        Args:
            T_gait (float): duration of one period of gait
            q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            vref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """
        # Update Vref with optimised speed
        vref[0,0] =  self.v_ref_qp[0,0]
        vref[1,0] =  self.v_ref_qp[1,0] 


        # Get the reference velocity in world frame (given in base frame)
        self.RPY = utils_mpc.quaternionToRPY(q[3:7, 0])

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)
        # yaw = np.linspace(0, self.T_gait-self.dt, self.n_steps) * vref[5, 0]

        # Update yaw and yaw velocity
        # dt_vector = np.linspace(self.dt, T_gait, self.n_steps)
        self.xref[5, 1:] = vref[5, 0] * self.dt_vector
        self.xref[11, 1:] = vref[5, 0]

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # yaw = np.linspace(0, T_gait-self.dt, self.n_steps) * v_ref[5, 0]
        self.xref[6, 1:] = vref[0, 0] * np.cos(self.xref[5, 1:]) - vref[1, 0] * np.sin(self.xref[5, 1:])
        self.xref[7, 1:] = vref[0, 0] * np.sin(self.xref[5, 1:]) + vref[1, 0] * np.cos(self.xref[5, 1:])

        # Update x and y depending on x and y velocities (cumulative sum)
        if vref[5, 0] != 0:
            self.xref[0, 1:] = (vref[0, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) +
                                vref[1, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
            self.xref[1, 1:] = (vref[1, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) -
                                vref[0, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
        else:
            self.xref[0, 1:] = vref[0, 0] * self.dt_vector[:]
            self.xref[1, 1:] = vref[1, 0] * self.dt_vector[:]
        # self.xref[0, 1:] = self.dx  # dt_vector * self.xref[6, 1:]
        # self.xref[1, 1:] = self.dy  # dt_vector * self.xref[7, 1:]

        # Start from position of the CoM in local frame
        #self.xref[0, 1:] += q[0, 0]
        #self.xref[1, 1:] += q[1, 0]

        #self.xref[5, 1:] += self.RPY[2, 0]

        # Desired height is supposed constant
        self.xref[2, 1:] = self.h_ref + z_average
        self.xref[8, 1:] = 0.0

        # No need to update Z velocity as the reference is always 0
        # No need to update roll and roll velocity as the reference is always 0 for those
        # No need to update pitch and pitch velocity as the reference is always 0 for those

        # Update the current state
        self.xref[0:3, 0:1] = q[0:3, 0:1]
        self.xref[3:6, 0:1] = self.RPY
        self.xref[6:9, 0:1] = v[0:3, 0:1]
        self.xref[9:12, 0:1] = v[3:6, 0:1]

        # Threshold for gamepad command (since even if you do not touch the joystick it's not 0.0)
        step = 0.05

        # Detect if command is above threshold
        if (np.abs(vref[2, 0]) > step) and (self.flag_rotation_command != 1):
            self.flag_rotation_command = 1


        # Current state vector of the robot
        # self.x0 = self.xref[:, 0:1]

        self.xref[0, 1:] += self.xref[0, 0]
        self.xref[1, 1:] += self.xref[1, 0]

        ### Update according to heightmap
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

        if r_des != 0 and abs(r_des - self.RPY[0,0]) > epsilon :
            self.xref[3,1:] = r_des
            self.xref[9,1:] = np.sign(r_des - self.RPY[0,0] )*min( rot_max  , k_sp*abs(r_des - self.RPY[0,0])  )
        else : 
            self.xref[3,1:] = 0.
            self.xref[9,1:] = 0.

        if p_des != 0 and abs(p_des - self.RPY[1,0]) > epsilon : 
            self.xref[4,1:] = p_des
            self.xref[10,1:] = np.sign(p_des - self.RPY[1,0] )*min( rot_max  , k_sp*abs(p_des - self.RPY[1,0])  )
        else : 
            self.xref[4,1:] = 0.
            self.xref[10,1:] = 0.


        # Get z
        isInside , i, j = self.heightMap.find_nearest(q[0, 0:1] , q[1, 0:1])

        if p_des != 0 :
            Z_OFFSET = self.h_ref - abs(p_des*0.05)
            
        for k in range(self.n_steps) :
            isInside , i, j = self.heightMap.find_nearest(self.xref[0,k+1] , self.xref[1,k+1])
            self.xref[2 , 1:] = result[0]*self.heightMap.x[i] + result[1]*self.heightMap.y[j] + result[2] + Z_OFFSET           
        


        if self.is_static:
            self.xref[0:3, 1:] = self.q_static[0:3, 0:1]
            self.xref[3:6, 1:] = (utils_mpc.quaternionToRPY(self.q_static[3:7, 0])).reshape((3, 1))

        return self.xref
    
    
    
    def compute_footsteps(self, k ,q_cur, v_cur, v_ref, reduced  , gaitPlanner):
        """Compute the desired location of footsteps over the prediction horizon

        Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
        For feet currently touching the ground the desired position is where they currently are.

        Args:
            q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """
        # Get the reference velocity in world frame (given in base frame)
        self.RPY = utils_mpc.quaternionToRPY(q_cur[3:7, 0])

        # Update intern gait matrix
        self.gait = gaitPlanner.getCurrentGait().astype(int) 

        self.fsteps[:, 0] = self.gait[:, 0]
        self.fsteps[:, 1:] = 0.

        self.update_remaining_time( k , gaitPlanner ) 
        # print("\n")
        # print(" k : " , k)
        # print(" k% mpc : " , k%self.k_mpc)
        # print("gait  : " , self.gait[:2, :])
        # print("feet : " , self.feet)
        # print("t0s : " , self.t0s)
        # print("selected surfaces : " , self.surface_selected )
        # print("\n")

        i = 1

        rpt_gait = np.repeat(self.gait[:, 1:] == 1, 3, axis=1)
       

        # Set current position of feet for feet in stance phase
      
        (self.fsteps[0, 1:])[rpt_gait[0, :]] = (gaitPlanner.o_feet_contact)[rpt_gait[0, :]]
      

        # Get future desired position of footsteps
        next_footstep = self.compute_next_footstep(q_cur, v_cur, v_ref)

        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        dt_cum = np.cumsum(self.gait[:, 0]) * self.dt

        # Get future yaw angle compared to current position
        angle = v_ref[5, 0] * dt_cum + self.RPY[2, 0]
        c = np.cos(angle)
        s = np.sin(angle)
        self.R[0:2, 0:2, :] = np.array([[c, -s], [s, c]])

        # Displacement following the reference velocity compared to current position
        if v_ref[5, 0] != 0:
            dx = (v_cur[0, 0] * np.sin(v_ref[5, 0] * dt_cum) +
                  v_cur[1, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
            dy = (v_cur[1, 0] * np.sin(v_ref[5, 0] * dt_cum) -
                  v_cur[0, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
        else:
            dx = v_cur[0, 0] * dt_cum
            dy = v_cur[1, 0] * dt_cum


        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        # Store the foot that are not in contact to run optimisation

         # L contains the indice of the moving feet 
        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]
        L = []


        while (self.gait[i, 0] != 0):

            # Feet that were in stance phase and are still in stance phase do not move
            A = rpt_gait[i-1, :] & rpt_gait[i, :]
            if np.any(rpt_gait[i-1, :] & rpt_gait[i, :]):
                (self.fsteps[i, 1:])[A] = (self.fsteps[i-1, 1:])[A]


            # Feet that were in swing phase and are now in stance phase need to be updated
            A = np.logical_not(rpt_gait[i-1, :]) & rpt_gait[i, :]
            q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height
            if np.any(A):

                # Get desired position of footstep compared to current position
                next_ft = (np.dot(self.R[:, :, i-1], next_footstep) + q_tmp +
                           np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')

                # Assignement only to feet that have been in swing phase
                (self.fsteps[i, 1:])[A] = next_ft[A]

                A = np.logical_not(self.gait[i-1, 1:]) & self.gait[i, 1:]

                # List of the new feet, to add to fsteps after the optimisation
                # Store some info about the feet position
                for j in np.where(A)[0] :

                    # px = fsteps[i , 1 + 3*j ]
                    # py = fsteps[i , 1 + 3*j + 1]
                    if i == 1 : # first line --> swing phase
                        
                        if j in self.feet and (k % self.k_mpc) == 0 and self.t0s[j] < 10e-4 :
                            # Only modyfying the surface for the flying feet at the beginning of the phase

                            px1 , py1 = next_ft[3*j]  , next_ft[3*j+1]   

                            height , id_surface = self.heightMap.getHeight(px1 , py1)  

                            self.surface_selected[j] = id_surface

                        id_surface = self.surface_selected[j] 
                        if id_surface != None :             
                            
                            nb_ineq = self.heightMap.Surfaces[id_surface].ineq_vect.shape[0] - 1       
                            L.append([i,j,id_surface , int(nb_ineq) ])                
                            # the feet is on a surface 
                        else :
                            L.append([i,j, -99, 0 ])
                        
                    else :         
                        px1 , py1 = next_ft[3*j]  , next_ft[3*j+1]   

                        height , id_surface = self.heightMap.getHeight(px1 , py1)  
                    
                        if id_surface != None :                 
                            
                            nb_ineq = self.heightMap.Surfaces[id_surface].ineq_vect.shape[0] - 1 
                            L.append([i,j,id_surface , int(nb_ineq) ])                        
                            # the feet is on a surface 
                        else :
                            L.append([i,j, -99, 0 ])

            i += 1

        # The number of new contact
        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]  (gait of size 4, not the nb iteration in first row !!)

        # print(L)

        self.run_optimisation(np.array(L) ,q_cur, v_cur, v_ref , dx , dy)

        # print(self.fsteps[0:2, 2::3])
        return self.fsteps


    def compute_next_footstep(self, q_cur, v_cur, v_ref , v_ref_bool = True):
        """Compute the desired location of footsteps for a given pair of current/reference velocities

    Compute a 3 by 1 matrix containing the desired location of each feet considering the current velocity of the
    robot and the reference velocity

    Args:
        q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
        v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
        v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        v_ref_bool (bool): Usefull to compute inequalities matrix
        """

        next_footstep = np.zeros((3, 4))

        c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])
        self.b_v_cur = R.transpose() @ v_cur[0:3, 0:1]
        self.b_v_ref = R.transpose() @ v_ref[0:3, 0:1]

        # TODO: Automatic detection of t_stance to handle arbitrary gaits
        t_stance = self.T_gait * 0.5
      
        # Add symmetry term
        next_footstep[0:2, :] = t_stance * 0.5 * self.b_v_cur[0:2, 0:1]  # + q_cur[0:2, 0:1]

        # Add feedback term
        if v_ref_bool :
            next_footstep[0:2, :] += self.k_feedback * (self.b_v_cur[0:2, 0:1] - self.b_v_ref[0:2, 0:1]) 
        else :
            next_footstep[0:2, :] += self.k_feedback * self.b_v_cur[0:2, 0:1]     

        # Add centrifugal term
        cross = self.cross3(np.array(self.b_v_cur[0:3, 0]), v_ref[3:6, 0])     

        next_footstep[0:2, :] += 0.5 * math.sqrt(self.h_ref/self.g) * cross[0:2, 0:1]

        # Add shoulders
        # add yaw rotation for test : 
        c, s = math.cos(self.RPY[1, 0]), math.sin(self.RPY[1, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])

        next_footstep[0:2, :] += (R @ self.shoulders[0:3, :])[0:2,:] 

        return next_footstep
    
    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])

                        
    def run_optimisation(self , L , q_cur, v_cur, v_ref , dx , dy) :
        ''' Update ftseps with the optimised positions. L is an array containing the foot position in the gait matrix i,j, 
        the surface associated and the nb of inequalities in the surface (usefull to determined the size of the inequalities matrix)
        Arg : L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]  (gait of size 4, not the nb iteration in first row !!)
        '''
        # if np.sum(L[:,3]) != 0 :
        #     print("--------- surface detetced--------------------------------")
        #     print(L)

        ineqMatrix = np.zeros(( int(np.sum(L[:,3])) , 3*L.shape[0] + 2  ))
        ineqVector = np.zeros( int(np.sum(L[:,3])) )

        eqMatrix = np.zeros((L.shape[0]  , 3*L.shape[0] + 2))
        eqVector = np.zeros( L.shape[0] )

        next_fstep_tmp = np.zeros((3,4))
        # Compute next footstep in base frame without the K_feedback of the reference speed

        next_fstep_tmp = self.compute_next_footstep(q_cur, v_cur, v_ref , False)

        count = 0 
        count_eq = 0

        for indice_L ,[i,j,indice_s , nb_ineq] in enumerate(L) :
    
            if indice_s == -99 :  # z = 0, ground 
                eqMatrix[count_eq,3*indice_L +2 + 2] = 1.
            else : 
                q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height

                next_ft = (np.dot(self.R[:, :, i-1], np.resize(next_fstep_tmp[:,j] , (3,1) ) ) + q_tmp +
                                np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')

                surface = self.heightMap.Surfaces[indice_s]

                # S * [Vrefx , Vrefy]
                ineqMatrix[ count:count + nb_ineq , :2 ]  = -self.k_feedback*surface.ineq_inner[:-1,:2]

                # S * [alphax  , aplhay  Pz]
                ineqMatrix[count:count+nb_ineq, 3*indice_L +2 : 3*indice_L +2 + 2] = surface.ineq_inner[:-1,:-1]
                ineqMatrix[count:count+nb_ineq, 3*indice_L +2 + 2] = surface.ineq_inner[:-1,-1]


                ineqVector[count:count+nb_ineq] = surface.ineq_vect_inner[:-1] - np.dot(surface.ineq_inner[:-1,:2] , next_ft[:2] )

                # S * [Vrefx , Vrefy]
                eqMatrix[ count_eq , :2 ]  = -self.k_feedback*surface.ineq_inner[-1,:2]

                # S * [alphax  , aplhay  Pz]
                eqMatrix[count_eq, 3*indice_L +2 : 3*indice_L +2 + 2] = surface.ineq_inner[-1,:-1]
                eqMatrix[count_eq, 3*indice_L +2 + 2] = surface.ineq_inner[-1,-1]

                eqVector[count_eq] = surface.ineq_vect_inner[-1] - np.dot(surface.ineq_inner[-1,:2] , next_ft[:2] )

            count_eq += 1 
            count += nb_ineq 

        P = np.identity(2 + L.shape[0]*3)
        q = np.zeros(2 + L.shape[0]*3)

        P[:2,:2] = self.weight_vref*np.identity(2)
        q[:2] = -self.weight_vref*np.array([v_ref[0,0] , v_ref[1,0]])
        res = quadprog_solve_qp(P, q,  G=ineqMatrix, h = ineqVector ,C=eqMatrix , d=eqVector)

        
        # Store the value for getRefState
        self.v_ref_qp[0,0] = res[0]
        self.v_ref_qp[1,0] = res[1]

        

        # if res[0] != v_ref[0,0] :
        #     print("V_ref : " , v_ref)
        #     print("V_optim : " , self.v_ref_qp)

        v_ref[0,0] = res[0]
        v_ref[1,0] = res[1]


        next_fstep_tmp = self.compute_next_footstep(q_cur, v_cur, v_ref , True)

        for indice_L ,[i,j,indice_s , nb_ineq] in enumerate(L) : 

            q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height

            next_ft = (np.dot(self.R[:, :, i-1], np.resize(next_fstep_tmp[:,j] , (3,1) ) ) + q_tmp +
                            np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')
            
            next_ft += np.array([res[2+3*indice_L] , res[2+3*indice_L+1] , res[2+3*indice_L+2] ])
            # next_ft += np.array([0., 0. , res[2+3*indice_L+2] ])

            self.fsteps[i ,1+ 3*j :1 + 3*j+3 ] = next_ft

        return 0

    def update_remaining_time(self , k , gaitPlanner ) : 

        if (k % self.k_mpc) == 0 :
            self.feet = []
            self.t0s = np.zeros((4, ))

            # Indexes of feet in swing phase
            self.feet = np.where(self.gait[0, 1:] == 0)[0]
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0
            
            # For each foot in swing phase get remaining duration of the swing phase
            for i in self.feet :              
                self.t_swing[i] = gaitPlanner.getPhaseDuration(0,i,0.)   # 0. for swing phase   
                self.remainingTime = gaitPlanner.getRemainingTime()
                value =  self.t_swing[i] - (self.remainingTime * self.k_mpc - ((k+1)%self.k_mpc) )*self.dt_wbc - self.dt_wbc
                self.t0s[i] =   np.max( [value , 0.] ) 
                

        else :
            # If no foot in swing phase
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # Increment of one time step for feet in swing phase
            for i in self.feet :
                self.t0s[i] = np.max(  [self.t0s[i] + self.dt_wbc, 0.0])

        return 0
