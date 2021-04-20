import numpy as np
from solo3D.tools.optimisation import genCost , quadprog_solve_qp
from solo3D.tools.collision_tool import get_intersect_segment , doIntersect_segment
import eigenpy
eigenpy.switchToNumpyArray()
#importing the bezier curve class
from curves import (bezier)
from curves.optimization import (problem_definition, setup_control_points)
from curves.optimization import constraint_flag

class FootTrajectoryGeneratorBezier:

    def __init__(self , T_gait , dt_tsid , k_mpc , fsteps_init , heightMap):

        self.T_gait = T_gait
        self.dt_wbc = dt_tsid
        self.k_mpc = k_mpc

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        # Foot trajectory generator
        self.max_height_feet = 0.035  # Rgeular height on the ground
        # self.max_height_switch_surface = 0.03  # height added to the surface height
        self.t_lock_before_touchdown = 0.07

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)

        # Variables for foot trajectory generator
        self.t_remaining = 0.
        self.t_stance = np.zeros((4, ))  # Total duration of current stance phase for each foot
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot
        self.footsteps_target = (self.shoulders[0:3, :]).copy()
        self.goals = fsteps_init.copy()  # Store 3D target position for feet
        self.vgoals = np.zeros((3, 4))  # Store 3D target velocity for feet
        self.agoals = np.zeros((3, 4))  # Store 3D target acceleration for feet

        self.feet = []
        self.t0s = np.zeros((4, ))

        self.Ax = np.zeros((6,4))
        self.Ay = np.zeros((6,4))
        self.Az = np.zeros((7,4))

        # heightMap
        self.heightMap = heightMap   

        # Bezier parameters

        #dimension of our problem (here 3 as our curve is 3D)
        self.N_int = 12 # Number of points in the least square problem
        self.dim = 3
        self.degree = 8  # Degree of the Bezier curve to match the polys
        self.pDs = []
        self.problems = []
        self.variableBeziers = []
        self.fitBeziers = []

        # Results
        res_size = self.dim*(self.degree + 1- 6)
        self.res = []

        self.t0_bezier = np.zeros((4,))

        self.new_surface = np.array([-99,-99,-99,-99])
        self.past_surface = np.array([-99,-99,-99,-99])
        self.ineq = [0]*4 
        self.ineq_vect = [0]*4
        self.x_margin = [0.008]*4
       

        # A bezier curve for each foot
        for i in range(4) :
            pD = problem_definition(self.dim)
            pD.degree = self.degree

            #values are 0 by default
            # Reduce the size of the problem by fixing intial and final points
            pD.init_pos = np.array([i,0.,0.]).T
            pD.end_pos   = np.array([[0., 0., 0.]]).T
            pD.init_vel = np.array([[0., 0., 0.]]).T
            pD.init_acc = np.array([[0., 0., 0.]]).T
            pD.end_vel = np.array([[0., 0., 0.]]).T
            pD.end_acc = np.array([[0., 0., 0.]]).T

            pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL  | constraint_flag.END_VEL  | constraint_flag.INIT_ACC  | constraint_flag.END_ACC

            
            self.pDs.append(pD)                
            self.problems.append( setup_control_points(pD) )
            self.variableBeziers.append( self.problems[-1].bezier() )
            self.fitBeziers.append( self.variableBeziers[-1].evaluate(np.zeros((res_size,1)) ) )
            self.res.append(np.zeros((res_size,1)))

       


    

    def updatePolyCoeff_XY(self,i_foot , x_init, v_init, a_init , x_end ,  t0 , t1 , h ) :
        ''' Compute coefficient for polynomial 5D curve for X and Y trajectory. Vel, Acc final is nulle. 
        Args:
        - i_foot (int): indice of foot to update the coefficients 
        - x_init (np.array x3) : initial position [x0,y0,z0]
        - v_init (np.array x3) : initial velocity [dx0,dy0,dz0]
        - a_init (np.array x3) : initial acceleration [ddx0,ddy0,ddz0]
        - x_end  (np.array x3) : end position [x1,y1,z1]
        - t0, t1 (float): intial and final time
        - h (float): height for z curve
        '''
        x0 , y0 , z0        = x_init
        dx0 , dy0 , dz0     = v_init
        ddx0 , ddy0 , ddz0  = a_init
        x1 , y1 , z1        = x_end

        # compute polynoms coefficients for x and y
        self.Ax[5,i_foot] = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12 *
                            x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[4,i_foot]  = (30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 +
                            2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[3,i_foot]  = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0 *
                            t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[2,i_foot]  = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1 **
                            2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[1,i_foot]  = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2 *
                                t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[0,i_foot]  = (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0 **
                                2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

        self.Ay[5,i_foot] = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12 *
                        y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[4,i_foot]  = (30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 +
                            2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[3,i_foot]  = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0 *
                        t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[2,i_foot]  = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1 **
                        2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[1,i_foot]  = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2 *
                            t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[0,i_foot]  = (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0 **
                        2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        
        return 0

        

    def updatePolyCoeff_Z(self,i_foot , x_init, v_init, a_init , x_end ,  t0 , t1 , h ) :
        ''' Compute coefficient for polynomial 5D curve for Z trajectory. Vel, Acc final is nulle. 
        Args:
        - i_foot (int): indice of foot to update the coefficients 
        - x_init (np.array x3) : initial position [x0,y0,z0]
        - v_init (np.array x3) : initial velocity [dx0,dy0,dz0]
        - a_init (np.array x3) : initial acceleration [ddx0,ddy0,ddz0]
        - x_end  (np.array x3) : end position [x1,y1,z1]
        - t0, t1 (float): intial and final time
        - h (float): height for z curve 
        '''
        x0 , y0 , z0        = x_init
        dx0 , dy0 , dz0     = v_init
        ddx0 , ddy0 , ddz0  = a_init
        x1 , y1 , z1        = x_end

        # coefficients for z (deterministic)
        # Version 2D (z1 = 0)
        # self.Az[6,i_foot] = -h/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[5,i_foot]  = (3*t1*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[4,i_foot]  = -(3*t1**2*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[3,i_foot]  = (t1**3*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[:3,i_foot] = 0

        # Version 3D (z1 != 0)
        self.Az[6,i_foot] =  (32.*z0 + 32.*z1 - 64.*h)/(t1**6)
        self.Az[5,i_foot] = - (102.*z0 + 90.*z1 - 192.*h)/(t1**5)
        self.Az[4,i_foot] = (111.*z0 + 81.*z1 - 192.*h)/(t1**4)
        self.Az[3,i_foot] = - (42.*z0 + 22.*z1 - 64.*h)/(t1**3)
        self.Az[2,i_foot] = 0
        self.Az[1,i_foot] = 0
        self.Az[0,i_foot] = z0

        return 0
 
    
    def evaluatePoly(self, i_foot , indice ,  t ) :
        '''Evaluate the polynomial curves at t or its derivatives 

        Args:
            - i_foot (int): indice of foot 
            - indice (int):  f indice derivative : 0 --> poly(t) ; 1 --> poly'(t)  (0,1 or 2)
            - t (float) : time 
        '''
        if indice == 0 :
            x = self.Ax[0,i_foot] + self.Ax[1,i_foot]*t + self.Ax[2,i_foot]*t**2 + self.Ax[3,i_foot]*t**3 + self.Ax[4,i_foot]*t**4 + self.Ax[5,i_foot]*t**5
            y = self.Ay[0,i_foot] + self.Ay[1,i_foot]*t + self.Ay[2,i_foot]*t**2 + self.Ay[3,i_foot]*t**3 + self.Ay[4,i_foot]*t**4 + self.Ay[5,i_foot]*t**5
            z = self.Az[0,i_foot] + self.Az[1,i_foot]*t + self.Az[2,i_foot]*t**2 + self.Az[3,i_foot]*t**3 + self.Az[4,i_foot]*t**4 + self.Az[5,i_foot]*t**5+ self.Az[6,i_foot]*t**6

            return np.array([x,y,z])

        elif indice == 1 :
            vx = self.Ax[1,i_foot] + 2*self.Ax[2,i_foot]*t + 3*self.Ax[3,i_foot]*t**2 + 4*self.Ax[4,i_foot]*t**3 + 5*self.Ax[5,i_foot]*t**4
            vy = self.Ay[1,i_foot] + 2*self.Ay[2,i_foot]*t + 3*self.Ay[3,i_foot]*t**2 + 4*self.Ay[4,i_foot]*t**3 + 5*self.Ay[5,i_foot]*t**4
            vz = self.Az[1,i_foot] + 2*self.Az[2,i_foot]*t + 3*self.Az[3,i_foot]*t**2 + 4*self.Az[4,i_foot]*t**3 + 5*self.Az[5,i_foot]*t**4 + 6*self.Az[6,i_foot]*t**5

            return np.array([vx,vy,vz])

        elif indice == 2 :
            ax = 2*self.Ax[2,i_foot] + 6*self.Ax[3,i_foot]*t + 12*self.Ax[4,i_foot]*t**2 + 20*self.Ax[5,i_foot]*t**3
            ay = 2*self.Ay[2,i_foot] + 6*self.Ay[3,i_foot]*t + 12*self.Ay[4,i_foot]*t**2 + 20*self.Ay[5,i_foot]*t**3
            az = 2*self.Az[2,i_foot] + 6*self.Az[3,i_foot]*t + 12*self.Az[4,i_foot]*t**2 + 20*self.Az[5,i_foot]*t**3 + 30*self.Az[6,i_foot]*t**4

            return np.array([ax,ay,az])

        else :
            return np.array([0.,0.,0.])




    def updateFootPosition(self,  k , i_foot ):
        """Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
        to the desired position on the ground (computed by the footstep planner)

        Args:
            k (int): number of time steps since the start of the simulation
            """
        t0 = self.t0s[i_foot]
        t1 = self.t_swing[i_foot]
        dt = self.dt_wbc
        h = self.max_height_feet  

        delta_t = t1 - t0

        if t0 < t1 - self.t_lock_before_touchdown :  

            # compute polynoms coefficients for x and y
            #self.updatePolyCoeff_Z(i_foot , self.goals[:, i_foot] , self.vgoals[:, i_foot] , self.agoals[:, i_foot] , self.footsteps_target[:,i_foot] , t0,t1,h )

            # compute polynoms coefficients for x and y
            if self.t0s[i_foot] < 10e-4  or k == 0 :  

                
                if self.new_surface[i_foot] != -99 : 
                    self.updatePolyCoeff_Z(i_foot , self.goals[:, i_foot] , np.zeros(3)  , np.zeros(3) , self.footsteps_target[:,i_foot] , t0,t1,0.03  + self.footsteps_target[:,i_foot][2])
                else : 
                    self.updatePolyCoeff_Z(i_foot , self.goals[:, i_foot] , np.zeros(3)  , np.zeros(3) , self.footsteps_target[:,i_foot] , t0,t1,h  + self.footsteps_target[:,i_foot][2])

                self.updatePolyCoeff_XY(i_foot , self.goals[:, i_foot] , np.zeros(3) , np.zeros(3) , self.footsteps_target[:,i_foot] , t0,t1,h ) 

                # New swing phase --> ineq surface 
                # Inequalities : 
                # Selected surface for arriving point : 
                if self.past_surface[i_foot] != self.new_surface[i_foot] :
                    id_surface = self.new_surface[i_foot]
                    surface = self.heightMap.Surfaces[id_surface] 
                    nb_vert = surface.vertices.shape[0]
                    vert = surface.vertices

                    P1 = self.goals[:2,i_foot]
                    P2 = self.footsteps_target[:2,i_foot]

                    for k in range(nb_vert) : 
                        Q1 = np.array([vert[k,0] , vert[k,1]])
                        if k < nb_vert - 1 :         
                            Q2 = np.array([vert[k+1,0] , vert[k+1,1]])
                        else : 
                            Q2 = np.array([vert[0,0] , vert[0,1]])

                        if doIntersect_segment(P1 , P2 , Q1 , Q2 ) :
                            P_r = get_intersect_segment(P1, P2, Q1, Q2)                            


                            # Should be sorted
                            self.ineq[i_foot] = surface.ineq[k,:]
                            self.ineq_vect[i_foot] = surface.ineq_vect[k] 

                            # x margin :
                            xt , yt , zt = self.evaluatePoly(i_foot , 0 , t0 )   
                            self.x_margin[i_foot] = min(0.008 , abs(P_r[0] -  xt ))

                else :
                    self.ineq[i_foot] = 0
                    self.ineq_vect[i_foot] = 0 
                

                self.pDs[i_foot].init_pos = np.array([self.goals[0,i_foot],   self.goals[1,i_foot],   self.evaluatePoly(i_foot , 0 , t0)[2]  ]).T
                self.pDs[i_foot].init_vel = delta_t*np.array([[0., 0., self.evaluatePoly(i_foot , 1 , t0)[2] ]]).T 
                self.pDs[i_foot].init_acc = (delta_t**2) * np.array([[0., 0., self.evaluatePoly(i_foot , 2 , t0)[2] ]]).T

            else :           
                self.updatePolyCoeff_XY(i_foot , self.goals[:, i_foot] , self.vgoals[:, i_foot] , self.agoals[:, i_foot] , self.footsteps_target[:,i_foot] , t0,t1,h )

                self.pDs[i_foot].init_pos = np.array([self.goals[:,i_foot]]).T
                self.pDs[i_foot].init_vel = delta_t*np.array([self.vgoals[:,i_foot]]).T
                self.pDs[i_foot].init_acc = (delta_t**2) * np.array([self.agoals[:,i_foot]]).T

            
            self.pDs[i_foot].end_pos   = np.array([self.footsteps_target[:,i_foot]]).T
            self.pDs[i_foot].end_vel = np.array([[0., 0., 0.]]).T
            self.pDs[i_foot].end_acc = np.array([[0., 0., 0.]]).T

            self.pDs[i_foot].flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL  | constraint_flag.END_VEL  | constraint_flag.INIT_ACC  | constraint_flag.END_ACC
            
            #generates the variable bezier curve with the parameters of problemDefinition
            self.problems[i_foot] = setup_control_points(self.pDs[i_foot])
            self.variableBeziers[i_foot] =  self.problems[i_foot].bezier()

            # Generate the points for the least square problem
            # t bezier : 0 --> 1
            # t polynomial curve : t0 --> t1
            ptsTime = [(self.evaluatePoly(i_foot , 0 , t0 + (t1-t0)*t_b ),t_b) for t_b in np.linspace(0,1,self.N_int)]

           
            xt , yt , zt = self.evaluatePoly(i_foot , 0 , t0 )   
               
            if np.sum(abs(self.ineq[i_foot])) != 0 and zt < self.footsteps_target[2,i_foot]: # No surface switch or already overpass the critical point
    
                vb = self.variableBeziers[i_foot]
                ineqMatrix = []
                ineqVector = []
                t_margin = 0.15 # 10% around the limit point !inferior to 1/nb point in linspace
                z_margin = self.footsteps_target[2,i_foot]*0.1 # 10% around the limit height
                # margin_x = 0.00
                margin_x = self.x_margin[i_foot]
                
                t_stop = 0.


                for ts in np.linspace(0,1,20) :
                    xt , yt , zt = self.evaluatePoly(i_foot , 0 , t0 + (t1-t0)*ts ) 

                    
                    if ts < t_stop + t_margin :                        
                        if zt < self.footsteps_target[2,i_foot] + z_margin  :
                            t_stop = ts
                            
                        t_curve = ts 
                        wayPoint = vb(ts)
                        ineqMatrix.append(-self.ineq[i_foot] @ wayPoint.B())
                        ineqVector.append( self.ineq[i_foot] @ wayPoint.c() - self.ineq_vect[i_foot] - margin_x )


                ineqMatrix = np.array(ineqMatrix)
                ineqVector = np.array(ineqVector)
              
            else : 
                ineqMatrix = None 
                ineqVector = None

            # RUN QP optimization

            A, b = genCost(self.variableBeziers[i_foot], ptsTime)
            #regularization matrix 
            reg = np.identity(A.shape[1]) * 0.00
            # self.res = quadprog_solve_qp(A+reg, b,  G=self.ineqMatrix, h = self.ineqVector )
            #self.res[i_foot] = quadprog_solve_qp(A+reg, b , G=None, h = None  ).reshape((-1,1))
            self.res[i_foot] = quadprog_solve_qp(A+reg, b , G=ineqMatrix, h = ineqVector  ).reshape((-1,1))

            self.fitBeziers[i_foot] = self.variableBeziers[i_foot].evaluate(self.res[i_foot] ) 

            self.t0_bezier[i_foot] = t0
        
        # Get the next point
        ev = t0 + dt
        evz = ev

        t_b = min( (ev - self.t0_bezier[i_foot])/(t1 -  self.t0_bezier[i_foot] )   , 1. )
        delta_t = t1 - self.t0_bezier[i_foot]

        # TODO need to fix bug : if surface is modified during the swing phase --> problem
        self.goals[:, i_foot] = self.fitBeziers[i_foot](t_b)
        self.vgoals[:, i_foot] =  self.fitBeziers[i_foot].derivate(t_b,1)/delta_t
        self.agoals[:, i_foot] = self.fitBeziers[i_foot].derivate(t_b,2)/delta_t**2

        
        # self.goals[:, i_foot] = self.evaluatePoly( i_foot , 0 , ev )
        # self.vgoals[:, i_foot] =  self.evaluatePoly( i_foot , 1 , ev )
        # self.agoals[:, i_foot] = self.evaluatePoly( i_foot , 2 , ev )



        if t0 < 0. or t0 > t1 :

            self.agoals[:2, i_foot] = np.zeros(2)
            self.vgoals[:2, i_foot] = np.zeros(2)        
          
        return 0


    def update_foot_trajectory(self, k , fsteps , gaitPlanner , fstepsPlanner):

        self.gait = gaitPlanner.getCurrentGait() 
        

        # Update foot target 
        self.footsteps_target = np.zeros((3, 4))

        for i in range(4) :
            index = 0
            while fsteps[index, 1 + 3*i] == 0. :
                index += 1 
            self.footsteps_target[:, i] = fsteps[index, (1+i*3):(4+i*3)]



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
            
       

        # Update new surface and past if t0 == 0 (new swing phase)
        if (k % self.k_mpc) == 0  :
            for i_foot in range(4) : 
                if self.t0s[i_foot] < 10e-6 :
                    self.past_surface[i_foot] = self.new_surface[i_foot]
                    if fstepsPlanner.surface_selected[i_foot] != None :
                        self.new_surface[i_foot] = fstepsPlanner.surface_selected[i_foot]
                    else : 
                        self.new_surface[i_foot] = -99  

        print("sf selected" , fstepsPlanner.surface_selected)
        print("new surface : " , self.new_surface)
    

        # print("\n")
        # print("t0 : " , self.t0s )
        # print("past_surface : " , self.past_surface )  
        # print("new_surface : " , self.new_surface )       
        # print("ftsep target : " , self.footsteps_target)
        # print("position init : " , self.goals)
        # print("\n")
        
        for i in self.feet :
            
            self.updateFootPosition(k,i)


        return 0

    def getFootPosition(self) :
        return self.goals

    def getFootVelocity(self) :
        return self.vgoals

    def getFootAcceleration(self) :
        return self.agoals

    