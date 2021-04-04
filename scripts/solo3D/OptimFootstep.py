import math 

import quadprog
from numpy import array, hstack, vstack
import numpy as np
import time 
from solo3D.tools.Surface import Surface

def load_data():
    try :
        with open("heightMap.dat" , "rb") as f :
            hm = pickle.load(f)
        with open("surfaces.dat" , "rb") as g :
            Sf = pickle.load(g)
    except :
        hm = []
        Sf = []
    return hm , Sf

def save_data(data_heightmap , data_surfaces) :
    with open("surfaces.dat" , "wb") as f :
        pickle.dump(data_heightmap,f)
    
    with open("surfaces.dat" , "wb") as f :
        pickle.dump(data_surfaces,f)

def quadprog_solve_qp(P, q, G=None, h=None, C=None, d=None, verbose=False):
    """
    min (1/2)x' P x + q' x
    subject to  G x <= h
    subject to  C x  = d
    """
    # qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_G = .5 * (P + P.T)  # make sure P is symmetric
    qp_a = -q
    qp_C = None
    qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C = -vstack([C, G]).T
            qp_b = -hstack([d, h])
        else:
            qp_C = -C.transpose()
            qp_b = -d
        meq = C.shape[0]
    elif G is not None:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
    ti = time.clock()
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    te = time.clock()
    print("QP execution time [ms] : " , (te - ti)*1000 )
    if verbose:
        return res
    # print('qp status ', res)
    return res[0]


def find_nearest(xt , yt):
    idx = (np.abs(x - xt)).argmin()
    idy = (np.abs(y - yt)).argmin()
    
    return idx, idy

def cross3( left, right):
    """Numpy is inefficient for this"""
    return np.array([[left[1] * right[2] - left[2] * right[1]],
                     [left[2] * right[0] - left[0] * right[2]],
                     [left[0] * right[1] - left[1] * right[0]]])





class OptimFsteps():
    def __init__(self , T_gait , h_ref , dt , g , k_feedback) :
        self.heightMap , self.Surfaces = load_data()
        
        #TODO improve with heighmap
        self.Nx = 500
        self.Ny = 500
        self.x = np.linspace(-2.0,2.0,self.Nx)
        self.y = np.linspace(-1.0,2.0,self.Ny)

         # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)

        self.footstep_tmp = np.zeros((3, 4))

        self.h_ref = h_ref
        self.RPY = np.zeros((3, 1))
        self.RPY[2, 0] = 0.
        self.R = np.zeros((3, 3, self.gait.shape[0]))
        self.R[2, 2, :] = 0.
        self.dt = dt 

        self.g = g
        self.T_gait = T_gait
        self.k_feedback = k_feedback
        self.t_stance = self.T_gait * 0.5

        # Coefficients QP
        self.weight_vref = 0.01
        self.weight_alpha = 1.


    def run_QP(self , q_cur , v_cur , v_ref , fsteps , gait , RPY):

        self.RPY = RPY
        self.gait = gait.astype(int)
        self.fsteps = fsteps
        # L contains the indice of the moving feet 
        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]
        L = []

        # New feet position need to be computed 
        # This is the same for each row of the gait, just need to add dx,dy and rotation
        # for the row that need computation of the feet
        #next_footstep = self.compute_next_footstep(q_cur, v_cur, v_ref , True)


        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        dt_cum = np.cumsum(self.gait[:, 0]) * self.dt

        # Get future yaw angle compared to current position
        # RPY = [roll, pitch , yaw]
        # Integration from the desired yaw velocity
        angle = v_ref[5, 0] * dt_cum + RPY[2, 0]
        c = np.cos(angle)
        s = np.sin(angle)
        # Rotation matrix for the gait
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

        
        i = 1 
        while (self.gait[i,0] != 0) :
            A = np.logical_not(self.gait[i-1, 1:]) & self.gait[i, 1:]

            if np.any(A) :             
                # Position of the center of mass
                # q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height
                
                # next_ft = (np.dot(self.R[:, :, i-1], next_footstep) + q_tmp +
                #                 np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')               
        
                # List of the new feet, to add to fsteps after the optimisation
                for j in np.where(A)[0] :

                    px = fsteps[i , 1 + 3*j ]
                    py = fsteps[i , 1 + 3*j + 1]
                    
                    # If needed to compute the intial footsteps, that are given by fsteps
                    # px1 , py1 = next_ft[3*j]  , next_ft[3*j+1]         
                    
                    if len(self.heightMap) == 0 :
                        id_surface = np.nan
                    elif abs(px) > x[-1] or abs(py) > y[-1] :
                        id_surface = np.nan
                    else :
                        idx , idy = find_nearest(px,py)
                        id_surface  = self.heightMap[idx,idy][1]
                    
                    if not(np.isnan(id_surface) ) :
                        # nb inequalities that define the surface
                        # Ineq matrix : --> -1 because normal vector at the end 
                        # [[ ineq1            [ x
                        #   ...                 y       =   ineq vector          
                        #   ineq 4              z ] 
                        #   normal eq ]]                        
                        
                        nb_ineq = Surfaces[int(id_surface)].ineq_vect.shape[0] - 1            
                    
                        L.append([i,j,int(id_surface) , int(nb_ineq) ])
                    
                        # the feet is on a surface 
                    else :
                        L.append(np.array([i,j, -99, 0 ]))
            i += 1 

        # The number of new contact
        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]  (gait of size 4, not the nb iteration in first row !!)
        L = np.array(L)

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

                surface = Surfaces[indice_s]

                # S * [Vrefx , Vrefy]
                ineqMatrix[ count:count + nb_ineq , :2 ]  = -k*surface.ineq_inner[:-1,:2]

                # S * [alphax  , aplhay  Pz]
                ineqMatrix[count:count+nb_ineq, 3*indice_L +2 : 3*indice_L +2 + 2] = surface.ineq_inner[:-1,:-1]
                ineqMatrix[count:count+nb_ineq, 3*indice_L +2 + 2] = surface.ineq_inner[:-1,-1]


                ineqVector[count:count+nb_ineq] = surface.ineq_vect_inner[:-1] - np.dot(surface.ineq_inner[:-1,:2] , next_ft[:2] )

                # S * [Vrefx , Vrefy]
                eqMatrix[ count_eq , :2 ]  = -k*surface.ineq_inner[-1,:2]

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

        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]  (gait of size 4, not the nb iteration in first row !!)

        v_ref[0,0] = res[0]
        v_ref[1,0] = res[1]
        next_fstep_tmp = self.compute_next_footstep(q_cur, v_cur, v_ref , True)

        for indice_L ,[i,j,indice_s , nb_ineq] in enumerate(L) : 

            q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height

            next_ft = (np.dot(self.R[:, :, i-1], np.resize(next_fstep_tmp[:,j] , (3,1) ) ) + q_tmp +
                            np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')
            
            next_ft += np.array([res[2+3*indice_L] , res[2+3*indice_L+1] , res[2+3*indice_L+2] ])

            self.fsteps[i ,1+ 3*j :1 + 3*j+3 ] = next_ft

        return self.fsteps
    
    def compute_next_footstep(self , q_cur, v_cur, v_ref, v_ref_bool = True) :  
        shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                    [0.14695, -0.14695, 0.14695, -0.14695],
                                    [0.0, 0.0, 0.0, 0.0]])
        
        c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])
        b_v_cur = R.transpose() @ v_cur[0:3, 0:1]
        b_v_ref = R.transpose() @ v_ref[0:3, 0:1]
     
        next_footstep = np.zeros((3, 4))

        next_footstep[0:2, :] = self.t_stance * 0.5 * b_v_cur[0:2, 0:1]
        
        if v_ref_bool :
            next_footstep[0:2, :] += self.k_feedback * (b_v_cur[0:2, 0:1] - b_v_ref[0:2, 0:1])
        else :
            next_footstep[0:2, :] += self.k_feedback * (b_v_cur[0:2, 0:1])
        cross = cross3(np.array(b_v_cur[0:3, 0]), v_ref[3:6, 0])
        next_footstep[0:2, :] += 0.5 * math.sqrt(self.h_ref/self.g) * cross[0:2, 0:1]
        next_footstep[0:2, :] += shoulders[0:2, :]

        return next_footstep