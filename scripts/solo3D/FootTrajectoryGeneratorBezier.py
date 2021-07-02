from example_robot_data import load
import pinocchio as pin
import pybullet as pyb
from ndcurves.optimization import constraint_flag
from ndcurves.optimization import (problem_definition, setup_control_points)
from ndcurves import (bezier)
import numpy as np
from solo3D.tools.optimisation import genCost, quadprog_solve_qp
from solo3D.tools.collision_tool import get_intersect_segment, doIntersect_segment
import eigenpy
eigenpy.switchToNumpyArray()
#importing the bezier curve class
from ndcurves import (bezier)
from ndcurves.optimization import (problem_definition, setup_control_points)
from ndcurves.optimization import constraint_flag
import pybullet as pyb
import pinocchio as pin
from example_robot_data import load
from solo3D.tools.ProfileWrapper import ProfileWrapper

# Store the results from cprofile
profileWrap = ProfileWrapper()

class FootTrajectoryGeneratorBezier:

    def __init__(self, T_gait, dt_tsid, k_mpc, fsteps_init, gait, footStepPlannerQP):

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
        self.t_lock_before_touchdown = 0.03

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

        self.Ax = np.zeros((6, 4))
        self.Ay = np.zeros((6, 4))
        self.Az = np.zeros((7, 4))

        # Coefficient used for simple trajectories (v,acc = 0)
        self.Ax_s = np.zeros((6, 4))
        self.Ay_s = np.zeros((6, 4))
        self.Az_s = np.zeros((7, 4))

        # Bezier parameters
        # dimension of our problem (here 3 as our curve is 3D)
        self.N_int = 15  # Number of points in the least square problem
        self.dim = 3
        self.degree = 7  # Degree of the Bezier curve to match the polys
        self.pDs = []
        self.problems = []
        self.variableBeziers = []
        self.fitBeziers = []

        # Results
        res_size = self.dim*(self.degree + 1 - 6)
        self.res = []

        self.t0_bezier = np.zeros((4,))
        
        # self.new_surface = np.array([-99, -99, -99, -99])
        # self.past_surface = np.array([-99, -99, -99, -99])
        self.ineq = [0]*4
        self.ineq_vect = [0]*4
        self.x_margin_max = 0.04
        self.x_margin = [self.x_margin_max]*4
        self.t_stop = [0.]*4
        self.t_margin = 0.15  # 1 % of the curve after critical point
        self.z_margin = 0.01  # 1 % of the height of the obstacle around the critical point

        # A bezier curve for each foot
        for i in range(4):
            pD = problem_definition(self.dim)
            pD.degree = self.degree

            # values are 0 by default
            # Reduce the size of the problem by fixing intial and final points
            pD.init_pos = np.array([i, 0., 0.]).T
            pD.end_pos = np.array([[0., 0., 0.]]).T
            pD.init_vel = np.array([[0., 0., 0.]]).T
            pD.init_acc = np.array([[0., 0., 0.]]).T
            pD.end_vel = np.array([[0., 0., 0.]]).T
            pD.end_acc = np.array([[0., 0., 0.]]).T

            pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL | constraint_flag.END_VEL | constraint_flag.INIT_ACC | constraint_flag.END_ACC

            self.pDs.append(pD)
            self.problems.append(setup_control_points(pD))
            self.variableBeziers.append(self.problems[-1].bezier())
            self.fitBeziers.append(self.variableBeziers[-1].evaluate(np.zeros((res_size, 1))))
            self.res.append(np.zeros((res_size, 1)))

        # Load the URDF model to get Pinocchio data and model structures
        robot = load('solo12')
        self.data = robot.data.copy()  # for velocity estimation (forward kinematics)
        self.model = robot.model.copy()  # for velocity estimation (forward kinematics)

        self.footStepPlannerQP = footStepPlannerQP
        self.gait = gait

        self.new_surface = [self.footStepPlannerQP.get_selected_surface(0)]*4
        self.past_surface = [self.footStepPlannerQP.get_selected_surface(0)]*4

    def updatePolyCoeff_XY(self, i_foot, x_init, v_init, a_init, x_end,  t0, t1, h):
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
        x0, y0, z0 = x_init
        dx0, dy0, dz0 = v_init
        ddx0, ddy0, ddz0 = a_init
        x1, y1, z1 = x_end

        # compute polynoms coefficients for x and y
        self.Ax[5, i_foot] = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12 *
                              x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[4, i_foot] = (30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 +
                              2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[3, i_foot] = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0 *
                              t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[2, i_foot] = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1 **
                               2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[1, i_foot] = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2 *
                               t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax[0, i_foot] = (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0 **
                              2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

        self.Ay[5, i_foot] = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12 *
                              y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[4, i_foot] = (30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 +
                              2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[3, i_foot] = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0 *
                              t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[2, i_foot] = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1 **
                               2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[1, i_foot] = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2 *
                               t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay[0, i_foot] = (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0 **
                              2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

        return 0

    def updatePolyCoeff_Z(self, i_foot, x_init, v_init, a_init, x_end,  t0, t1, h):
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
        x0, y0, z0 = x_init
        dx0, dy0, dz0 = v_init
        ddx0, ddy0, ddz0 = a_init
        x1, y1, z1 = x_end

        # coefficients for z (deterministic)
        # Version 2D (z1 = 0)
        # self.Az[6,i_foot] = -h/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[5,i_foot]  = (3*t1*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[4,i_foot]  = -(3*t1**2*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[3,i_foot]  = (t1**3*h)/((t1/2)**3*(t1 - t1/2)**3)
        # self.Az[:3,i_foot] = 0

        # Version 3D (z1 != 0)
        self.Az[6, i_foot] = (32.*z0 + 32.*z1 - 64.*h)/(t1**6)
        self.Az[5, i_foot] = - (102.*z0 + 90.*z1 - 192.*h)/(t1**5)
        self.Az[4, i_foot] = (111.*z0 + 81.*z1 - 192.*h)/(t1**4)
        self.Az[3, i_foot] = - (42.*z0 + 22.*z1 - 64.*h)/(t1**3)
        self.Az[2, i_foot] = 0
        self.Az[1, i_foot] = 0
        self.Az[0, i_foot] = z0

        return 0

    def evaluateBezier(self, i_foot, indice,  t):
        '''Evaluate the polynomial curves at t or its derivatives 

        Args:
            - i_foot (int): indice of foot 
            - indice (int):  f indice derivative : 0 --> poly(t) ; 1 --> poly'(t)  (0,1 or 2)
            - t (float) : time in polynomial ref (t_bezier : 0 <--> 1)
        '''
        t1 = self.t_swing[i_foot]
        delta_t = t1 - self.t0_bezier[i_foot]
        t_b = min((t - self.t0_bezier[i_foot])/(t1 - self.t0_bezier[i_foot]), 1.)

        if indice == 0:
            return self.fitBeziers[i_foot](t_b)

        elif indice == 1:
            return self.fitBeziers[i_foot].derivate(t_b, 1)/delta_t

        elif indice == 2:
            return self.fitBeziers[i_foot].derivate(t_b, 2)/delta_t**2

        else:
            return np.array([0., 0., 0.])

    def evaluatePoly(self, i_foot, indice,  t):
        '''Evaluate the polynomial ndcurves at t or its derivatives 

        Args:
            - i_foot (int): indice of foot 
            - indice (int):  f indice derivative : 0 --> poly(t) ; 1 --> poly'(t)  (0,1 or 2)
            - t (float) : time 
        '''
        if indice == 0:
            x = self.Ax[0, i_foot] + self.Ax[1, i_foot]*t + self.Ax[2, i_foot]*t**2 + \
                self.Ax[3, i_foot]*t**3 + self.Ax[4, i_foot]*t**4 + self.Ax[5, i_foot]*t**5
            y = self.Ay[0, i_foot] + self.Ay[1, i_foot]*t + self.Ay[2, i_foot]*t**2 + \
                self.Ay[3, i_foot]*t**3 + self.Ay[4, i_foot]*t**4 + self.Ay[5, i_foot]*t**5
            z = self.Az[0, i_foot] + self.Az[1, i_foot]*t + self.Az[2, i_foot]*t**2 + self.Az[3, i_foot] * \
                t**3 + self.Az[4, i_foot]*t**4 + self.Az[5, i_foot]*t**5 + self.Az[6, i_foot]*t**6

            return np.array([x, y, z])

        elif indice == 1:
            vx = self.Ax[1, i_foot] + 2*self.Ax[2, i_foot]*t + 3*self.Ax[3, i_foot] * \
                t**2 + 4*self.Ax[4, i_foot]*t**3 + 5*self.Ax[5, i_foot]*t**4
            vy = self.Ay[1, i_foot] + 2*self.Ay[2, i_foot]*t + 3*self.Ay[3, i_foot] * \
                t**2 + 4*self.Ay[4, i_foot]*t**3 + 5*self.Ay[5, i_foot]*t**4
            vz = self.Az[1, i_foot] + 2*self.Az[2, i_foot]*t + 3*self.Az[3, i_foot]*t**2 + 4 * \
                self.Az[4, i_foot]*t**3 + 5*self.Az[5, i_foot]*t**4 + 6*self.Az[6, i_foot]*t**5

            return np.array([vx, vy, vz])

        elif indice == 2:
            ax = 2*self.Ax[2, i_foot] + 6*self.Ax[3, i_foot]*t + 12*self.Ax[4, i_foot]*t**2 + 20*self.Ax[5, i_foot]*t**3
            ay = 2*self.Ay[2, i_foot] + 6*self.Ay[3, i_foot]*t + 12*self.Ay[4, i_foot]*t**2 + 20*self.Ay[5, i_foot]*t**3
            az = 2*self.Az[2, i_foot] + 6*self.Az[3, i_foot]*t + 12*self.Az[4, i_foot] * \
                t**2 + 20*self.Az[5, i_foot]*t**3 + 30*self.Az[6, i_foot]*t**4

            return np.array([ax, ay, az])

        else:
            return np.array([0., 0., 0.])

    def updateFootPosition(self,  k, i_foot):
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

        if t0 < t1 - self.t_lock_before_touchdown:

            # # compute polynoms coefficients for x and y and z : without Bezier optim
            # if self.t0s[i_foot] < 10e-4 or k == 0: 
            #     self.updatePolyCoeff_Z(i_foot , self.goals[:, i_foot] , self.vgoals[:, i_foot] , self.agoals[:, i_foot] , self.footsteps_target[:,i_foot] , t0,t1,h )
            #     self.updatePolyCoeff_XY(i_foot, self.goals[:, i_foot], np.zeros(3), np.zeros(3), self.footsteps_target[:, i_foot], t0, t1, h)
            # else : 
            #     self.updatePolyCoeff_XY(i_foot, self.goals[:, i_foot], self.vgoals[:, i_foot] , self.agoals[:, i_foot] , self.footsteps_target[:, i_foot], t0, t1, h)


            # compute polynoms coefficients for x and y
            if self.t0s[i_foot] < 10e-4 or k == 0:

                if self.new_surface[i_foot].get_height(self.footsteps_target[:2, i_foot]) >= 10-4:
                    self.updatePolyCoeff_Z(i_foot, self.goals[:, i_foot], np.zeros(3), np.zeros(
                        3), self.footsteps_target[:, i_foot], t0, t1, 0.03 + self.footsteps_target[:, i_foot][2])
                else:
                    # Walking on the floor 
                    self.updatePolyCoeff_Z(i_foot, self.goals[:, i_foot], np.zeros(3), np.zeros(
                        3), self.footsteps_target[:, i_foot], t0, t1, h + self.footsteps_target[:, i_foot][2])

                self.updatePolyCoeff_XY(i_foot, self.goals[:, i_foot], np.zeros(
                    3), np.zeros(3), self.footsteps_target[:, i_foot], t0, t1, h)

                # New swing phase --> ineq surface
                # Inequalities :
                # Selected surface for arriving point :

                self.t_stop[i_foot] = 0.

                if abs(self.past_surface[i_foot].get_height(self.footsteps_target[:2, i_foot]) - self.new_surface[i_foot].get_height(self.footsteps_target[:2, i_foot])) >= 10e-4:
                    surface = self.new_surface[i_foot]
                    vert = surface.get_vertices()
                    nb_vert = vert.shape[0]

                    P1 = self.goals[:2, i_foot]
                    P2 = self.footsteps_target[:2, i_foot]

                    for k in range(nb_vert):
                        Q1 = np.array([vert[k, 0], vert[k, 1]])
                        if k < nb_vert - 1:
                            Q2 = np.array([vert[k+1, 0], vert[k+1, 1]])
                        else:
                            Q2 = np.array([vert[0, 0], vert[0, 1]])

                        if doIntersect_segment(P1, P2, Q1, Q2):
                            P_r = get_intersect_segment(P1, P2, Q1, Q2)
                        
                            # Should be sorted
                            # self.ineq[i_foot] = surface.ineq[k, :]
                            # self.ineq_vect[i_foot] = surface.ineq_vect[k]
                            a = 0.
                            if (Q1[0] - Q2[0]) != 0. :
                                a = (Q2[1] - Q1[1])/(Q2[0] - Q1[0])
                                b = Q1[1] - a*Q1[0]
                                self.ineq[i_foot] = np.array([-a , 1. , 0.]) # -ax + y = b
                                self.ineq_vect[i_foot] = b 
                            else : 
                                # Inequality of the surface corresponding to these vertices
                                self.ineq[i_foot] = np.array([-1. , 0. , 0.])
                                self.ineq_vect[i_foot] = -Q1[0]
                            
                            if np.dot(self.ineq[i_foot] , self.footsteps_target[:, i_foot]) > self.ineq_vect[i_foot] :
                                # Wrong side, the targeted point is inside the surface
                                self.ineq[i_foot] = - self.ineq[i_foot]
                                self.ineq_vect[i_foot] = -self.ineq_vect[i_foot] 

                            # If foot position already closer than margin
                            self.x_margin[i_foot] = max(min(self.x_margin_max, abs(P_r[0] - P1[0]) - 0.001), 0.)
                            

                else:
                    self.ineq[i_foot] = 0
                    self.ineq_vect[i_foot] = 0

                self.pDs[i_foot].init_pos = np.array(
                    [self.goals[0, i_foot],   self.goals[1, i_foot],   self.evaluatePoly(i_foot, 0, t0)[2]]).T
                self.pDs[i_foot].init_vel = delta_t*np.array([[0., 0., self.evaluatePoly(i_foot, 1, t0)[2]]]).T
                self.pDs[i_foot].init_acc = (delta_t**2) * np.array([[0., 0., self.evaluatePoly(i_foot, 2, t0)[2]]]).T

            else:
                self.updatePolyCoeff_XY(i_foot, self.goals[:, i_foot], self.vgoals[:, i_foot],
                                        self.agoals[:, i_foot], self.footsteps_target[:, i_foot], t0, t1, h)

                self.pDs[i_foot].init_pos = np.array([self.goals[:, i_foot]]).T
                self.pDs[i_foot].init_vel = delta_t*np.array([self.vgoals[:, i_foot]]).T
                self.pDs[i_foot].init_acc = (delta_t**2) * np.array([self.agoals[:, i_foot]]).T

            self.pDs[i_foot].end_pos = np.array([self.footsteps_target[:, i_foot]]).T
            self.pDs[i_foot].end_vel = np.array([[0., 0., 0.]]).T
            self.pDs[i_foot].end_acc = np.array([[0., 0., 0.]]).T

            self.pDs[i_foot].flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL | constraint_flag.END_VEL | constraint_flag.INIT_ACC | constraint_flag.END_ACC

            # generates the variable bezier curve with the parameters of problemDefinition
            self.problems[i_foot] = setup_control_points(self.pDs[i_foot])
            self.variableBeziers[i_foot] = self.problems[i_foot].bezier()

            # Generate the points for the least square problem
            # t bezier : 0 --> 1
            # t polynomial curve : t0 --> t1
            ptsTime = [(self.evaluatePoly(i_foot, 0, t0 + (t1-t0)*t_b), t_b) for t_b in np.linspace(0, 1, self.N_int)]

            xt, yt, zt = self.evaluatePoly(i_foot, 0, t0)
            t_margin = self.t_margin*t1  # 10% around the limit point !inferior to 1/nb point in linspace

            # No surface switch or already overpass the critical point
            if np.sum(abs(self.ineq[i_foot])) != 0 and ((zt < self.footsteps_target[2, i_foot]) or (self.t0s[i_foot] < self.t_stop[i_foot] + t_margin)) and self.x_margin[i_foot] != 0.:
          
                # Modify (should not be used) x_margin during flight, if problem during flight    
                surface = self.new_surface[i_foot]
                vert = surface.get_vertices()
                nb_vert = vert.shape[0]

                P1 = self.goals[:2, i_foot]
                P2 = self.footsteps_target[:2, i_foot]

                for k in range(nb_vert):
                    Q1 = np.array([vert[k, 0], vert[k, 1]])
                    if k < nb_vert - 1:
                        Q2 = np.array([vert[k+1, 0], vert[k+1, 1]])
                    else:
                        Q2 = np.array([vert[0, 0], vert[0, 1]])

                    if doIntersect_segment(P1, P2, Q1, Q2):
                        P_r = get_intersect_segment(P1, P2, Q1, Q2)
                        # x margin :
                        # xt , yt , zt = self.evaluatePoly(i_foot , 0 , t0 )
                        self.x_margin[i_foot] = max(min(self.x_margin_max, abs(P_r[0] - P1[0]) - 0.001), 0.)

                vb = self.variableBeziers[i_foot]
                ineqMatrix = []
                ineqVector = []

                z_margin = self.footsteps_target[2, i_foot]*self.z_margin  # 10% around the limit height
                margin_x = self.x_margin[i_foot]
                ct = 0

                for ts in np.linspace(0, 1, 20):
                    xt, yt, zt = self.evaluatePoly(i_foot, 0, t0 + (t1-t0)*ts)

                    if t0 + (t1-t0)*ts < self.t_stop[i_foot] + t_margin:
                        if zt < self.footsteps_target[2, i_foot] + z_margin:
                            self.t_stop[i_foot] = t0 + (t1-t0)*ts

                        t_curve = ts
                        wayPoint = vb(ts)
                        ineqMatrix.append(-self.ineq[i_foot] @ wayPoint.B())
                        ineqVector.append(self.ineq[i_foot] @ wayPoint.c() - self.ineq_vect[i_foot] - (margin_x + ct))
                        ct += 0.00

                if len(ineqVector) == 0:
                    ineqMatrix = None
                    ineqVector = None
                else:
                    ineqMatrix = np.array(ineqMatrix)
                    ineqVector = np.array(ineqVector)

            else:
                ineqMatrix = None
                ineqVector = None

            # RUN QP optimization

            A, b = genCost(self.variableBeziers[i_foot], ptsTime)
            # regularization matrix
            reg = np.identity(A.shape[1]) * 0.00

            try:
                self.res[i_foot] = quadprog_solve_qp(A+reg, b, G=ineqMatrix, h=ineqVector).reshape((-1, 1))
            except:
                print("ERROR QP : CONSTRAINT NO CONSISTENT")

            self.fitBeziers[i_foot] = self.variableBeziers[i_foot].evaluate(self.res[i_foot])

            self.t0_bezier[i_foot] = t0

        # Get the next point
        ev = t0 + dt
        evz = ev

        t_b = min((ev - self.t0_bezier[i_foot])/(t1 - self.t0_bezier[i_foot]), 1.)
        delta_t = t1 - self.t0_bezier[i_foot]

        # TODO need to fix bug : if surface is modified during the swing phase --> problem
        self.goals[:, i_foot] = self.fitBeziers[i_foot](t_b)
        self.vgoals[:, i_foot] = self.fitBeziers[i_foot].derivate(t_b, 1)/delta_t
        self.agoals[:, i_foot] = self.fitBeziers[i_foot].derivate(t_b, 2)/delta_t**2

        # self.goals[:, i_foot] = self.evaluatePoly( i_foot , 0 , ev )
        # self.vgoals[:, i_foot] =  self.evaluatePoly( i_foot , 1 , ev )
        # self.agoals[:, i_foot] = self.evaluatePoly( i_foot , 2 , ev )

        if t0 < 0. or t0 > t1:

            self.agoals[:2, i_foot] = np.zeros(2)
            self.vgoals[:2, i_foot] = np.zeros(2)

        return 0

    @profileWrap.profile
    def update(self, k, targetFootstep, device, q_filt, v_filt):

        gait = self.gait.getCurrentGait()

        # Update foot target
        self.footsteps_target = targetFootstep

        # Update current foot position with pybullet feedback
        # Only for Bezier curves
        self.updateFootPositionFeedback(k, q_filt, v_filt, device)

        if (k % self.k_mpc) == 0:
            self.feet = []
            self.t0s = np.zeros((4, ))

            # Indexes of feet in swing phase
            self.feet = np.where(gait[0, :] == 0)[0]
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # For each foot in swing phase get remaining duration of the swing phase
            for i in self.feet:
                self.t_swing[i] = self.gait.getPhaseDuration(0, int(i), 0.)   # 0. for swing phase
                self.remainingTime = self.gait.getRemainingTime()
                value = self.t_swing[i] - (self.remainingTime * self.k_mpc - ((k+1) %self.k_mpc))*self.dt_wbc - self.dt_wbc
                self.t0s[i] = np.max([value, 0.])

        else:
            # If no foot in swing phase
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # Increment of one time step for feet in swing phase
            for i in self.feet:
                self.t0s[i] = np.max([self.t0s[i] + self.dt_wbc, 0.0])

        

        # Update new surface and past if t0 == 0 (new swing phase)
        if (k % self.k_mpc) == 0:
            for i_foot in range(4):
                if self.t0s[i_foot] < 10e-6:
                    self.past_surface[i_foot] = self.new_surface[i_foot]
                    self.new_surface[i_foot] = self.footStepPlannerQP.get_selected_surface(i_foot)
        
        for i in self.feet:
            # Only for 5th order polynomial curve
            # if self.t0s[i] < 10e-4 and (k % self.k_mpc) == 0 :
            #     self.updateFootPositionFeedback(k, q_filt, v_filt, device)

            self.updateFootPosition(k, i)

        return 0

    def getFootPosition(self):
        return self.goals

    def getFootVelocity(self):
        return self.vgoals

    def getFootAcceleration(self):
        return self.agoals

    def updateFootPositionFeedback(self, k, q_filt, v_filt, device):

        # Update current position : Pybullet feedback, directly
        ##########################

        # linkId = [3, 7 ,11 ,15]
        # if k != 0 :
        #     links = pyb.getLinkStates(device.pyb_sim.robotId, linkId , computeForwardKinematics=True , computeLinkVelocity=True )

        #     for j in range(4) :
        #         self.goals[:,j] = np.array(links[j][4])[:]   # pos frame world for feet
        #         self.goals[2,j] -= 0.016988                  #  Z offset due to position of frame in object
        #         self.vgoals[:,j] = np.array(links[j][6])     # vel frame world for feet

        # Update current position : Pybullet feedback, with forward dynamics
        ##########################

        if k > 0:    # Dummy device for k == 0
            qmes = np.zeros((19, 1))
            revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
            jointStates = pyb.getJointStates(device.pyb_sim.robotId, revoluteJointIndices)
            baseState = pyb.getBasePositionAndOrientation(device.pyb_sim.robotId)
            qmes[:3, 0] = baseState[0]
            qmes[3:7, 0] = baseState[1]
            qmes[7:, 0] = [state[0] for state in jointStates]
            pin.forwardKinematics(self.model, self.data, qmes, v_filt)
        else:
            pin.forwardKinematics(self.model, self.data, q_filt, v_filt)

        # Update current position : Estimator feedback, with forward dynamics
        ##########################

        # pin.forwardKinematics(self.model, self.data, q_filt, v_filt)

        contactFrameId = [10, 18, 26, 34]   # = [ FL , FR , HL , HR]

        for j in range(4):
            framePlacement = pin.updateFramePlacement(
                self.model, self.data, contactFrameId[j])    # = solo.data.oMf[18].translation
            frameVelocity = pin.getFrameVelocity(self.model, self.data, contactFrameId[j], pin.ReferenceFrame.LOCAL)

            self.goals[:, j] = framePlacement.translation[:]
            self.goals[2, j] -= 0.016988                     # Pybullet offset on Z
            # self.vgoals[:,j] = frameVelocity.linear       # velocity feedback not working

        return 0

    def updatePolyCoeff_simple(self, i_foot, x_init, x_end,  t1):
        ''' Compute coefficients for polynomial 5D curve for X,Y,Z trajectory. Vel, Acc initial and final are nulle. 
        Args:
        - i_foot (int): indice of foot to update the coefficients 
        - x_init (np.array x3) : initial position [x0,y0,z0]
        - x_end  (np.array x3) : end position [x1,y1,z1]
        -  t1 (float): final time
        -  h (float): height for z curve 
        '''
        x0, y0, z0 = x_init
        x1, y1, z1 = x_end

        h = z1 + self.max_height_feet

        self.Ax_s[5, i_foot] = - (6 * x0 - 6*x1)/(t1**5)
        self.Ax_s[4, i_foot] = - (- 15*x0 + 15*x1)/(t1**4)
        self.Ax_s[3, i_foot] = - (10*x0 - 10*x1)/(t1**3)
        self.Ax_s[2, i_foot] = 0.
        self.Ax_s[1, i_foot] = 0.
        self.Ax_s[0, i_foot] = x0

        self.Ay_s[5, i_foot] = - (6 * y0 - 6*y1)/(t1**5)
        self.Ay_s[4, i_foot] = - (- 15*y0 + 15*y1)/(t1**4)
        self.Ay_s[3, i_foot] = - (10*y0 - 10*y1)/(t1**3)
        self.Ay_s[2, i_foot] = 0.
        self.Ay_s[1, i_foot] = 0.
        self.Ay_s[0, i_foot] = y0

        # Version 3D
        self.Az_s[6, i_foot] = (32.*z0 + 32.*z1 - 64.*h)/(t1**6)
        self.Az_s[5, i_foot] = - (102.*z0 + 90.*z1 - 192.*h)/(t1**5)
        self.Az_s[4, i_foot] = (111.*z0 + 81.*z1 - 192.*h)/(t1**4)
        self.Az_s[3, i_foot] = - (42.*z0 + 22.*z1 - 64.*h)/(t1**3)
        self.Az_s[2, i_foot] = 0
        self.Az_s[1, i_foot] = 0
        self.Az_s[0, i_foot] = z0

        return 0

    def evaluatePoly_simple(self, i_foot, indice,  t):
        '''Evaluate the polynomial curves at t or its derivatives 

        Args:
            - i_foot (int): indice of foot 
            - indice (int):  f indice derivative : 0 --> poly(t) ; 1 --> poly'(t)  (0,1 or 2)
            - t (float) : time 
        '''
        if indice == 0:
            x = self.Ax_s[0, i_foot] + self.Ax_s[1, i_foot]*t + self.Ax_s[2, i_foot]*t**2 + \
                self.Ax_s[3, i_foot]*t**3 + self.Ax_s[4, i_foot]*t**4 + self.Ax_s[5, i_foot]*t**5
            y = self.Ay_s[0, i_foot] + self.Ay_s[1, i_foot]*t + self.Ay_s[2, i_foot]*t**2 + \
                self.Ay_s[3, i_foot]*t**3 + self.Ay_s[4, i_foot]*t**4 + self.Ay_s[5, i_foot]*t**5
            z = self.Az_s[0, i_foot] + self.Az_s[1, i_foot]*t + self.Az_s[2, i_foot]*t**2 + self.Az_s[3,
                                                                                                      i_foot]*t**3 + self.Az_s[4, i_foot]*t**4 + self.Az_s[5, i_foot]*t**5 + self.Az_s[6, i_foot]*t**6

            return np.array([x, y, z])

        elif indice == 1:
            vx = self.Ax_s[1, i_foot] + 2*self.Ax_s[2, i_foot]*t + 3*self.Ax_s[3, i_foot] * \
                t**2 + 4*self.Ax_s[4, i_foot]*t**3 + 5*self.Ax_s[5, i_foot]*t**4
            vy = self.Ay_s[1, i_foot] + 2*self.Ay_s[2, i_foot]*t + 3*self.Ay_s[3, i_foot] * \
                t**2 + 4*self.Ay_s[4, i_foot]*t**3 + 5*self.Ay_s[5, i_foot]*t**4
            vz = self.Az_s[1, i_foot] + 2*self.Az_s[2, i_foot]*t + 3*self.Az_s[3, i_foot]*t**2 + 4 * \
                self.Az_s[4, i_foot]*t**3 + 5*self.Az_s[5, i_foot]*t**4 + 6*self.Az_s[6, i_foot]*t**5

            return np.array([vx, vy, vz])

        elif indice == 2:
            ax = 2*self.Ax_s[2, i_foot] + 6*self.Ax_s[3, i_foot]*t + 12 * \
                self.Ax_s[4, i_foot]*t**2 + 20*self.Ax_s[5, i_foot]*t**3
            ay = 2*self.Ay_s[2, i_foot] + 6*self.Ay_s[3, i_foot]*t + 12 * \
                self.Ay_s[4, i_foot]*t**2 + 20*self.Ay_s[5, i_foot]*t**3
            az = 2*self.Az_s[2, i_foot] + 6*self.Az_s[3, i_foot]*t + 12*self.Az_s[4, i_foot] * \
                t**2 + 20*self.Az_s[5, i_foot]*t**3 + 30*self.Az_s[6, i_foot]*t**4

            return np.array([ax, ay, az])

        else:
            return np.array([0., 0., 0.])

    def print_profile(self , output_file):
        ''' Print the profile computed with cProfile
        Args : 
        - output_file (str) :  file name
        '''
        profileWrap.print_stats(output_file)
        
        return  0

