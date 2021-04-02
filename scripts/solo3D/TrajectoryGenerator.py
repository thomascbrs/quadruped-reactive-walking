import numpy as np
#use array representation for binding eigen objects to python
import eigenpy
eigenpy.switchToNumpyArray()
#importing the bezier curve class
from curves import (bezier)

#importing tools to plot bezier curves
from curves.plot import (plotBezier)
import matplotlib.pyplot as plt
from curves.optimization import (problem_definition, setup_control_points)
from curves.optimization import constraint_flag

import quadprog
from numpy import array, hstack, vstack
import time

# importing classical numpy objects
from numpy import zeros, array, identity, dot


# 5D polynome for X,Y trajectories 
# Acc,speed = 0 , position fixed
def poly_5D(x0,x1,T,t) :
    a0 = x0 
    dx = x1-x0
    T3 = T**3 
    T4 = T**4 
    T5 = T**5
    a3 = 10*dx/T3
    a4 = -15*dx/T4
    a5 = 6*dx/T5
    return a0 + a3*t**3 + a4*t**4 + a5*t**5

# return 5D for x,y and 6D for z 
def poly_naive(x0,x1,h,T,t) :
    x = poly_5D(x0[0],x1[0],T,t )
    y = poly_5D(x0[1],x1[1],T,t )
    z = poly_6D(x0[2],x1[2],h,T,t )
    return np.array([x,y,z])

# 5D polynome for X,Y trajectories 
# Acc,speed = 0 , position + height fixed
def poly_6D(x0,x1,h,T,t) :
    a0 = x0 
    dx = x1-x0
    T3 = T**3 
    T4 = T**4 
    T5 = T**5
    T6 = T**6
    a3 = -(22*dx - 64*h)/T3
    a4 = (81*dx - 192*h)/T4
    a5 = -(90*dx - 192*h)/T5
    a6 = (32*dx - 64*h)/T6
    return a0 + a3*t**3 + a4*t**4 + a5*t**5 + a6*t**6

def to_least_square(A, b):
    return np.dot(A.T, A), - np.dot(A.T, b)

def genCost(variableBezier, ptsTime):
    #first evaluate variableBezier for each time sampled
    allsEvals = [(variableBezier(time), pt) for (pt,time) in ptsTime]
    #then compute the least square form of the cost for each points
    allLeastSquares = [to_least_square(el.B(), -el.c() + pt) for (el, pt) in  allsEvals]
    #and finally sum the costs
    Ab = [sum(x) for x in zip(*allLeastSquares)]
    return Ab[0], Ab[1]

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
    t_init = time.clock()
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    t_end = time.clock()  - t_init
    print("time optim coeff Bezier : " , t_end*1000 , " [ms]")
    if verbose:
        return res
    # print('qp status ', res)
    return res[0]

class polynomial_curve(object) :
    def __init__(self):
        self.Ax0 , self.Ax1 , self.Ax2 , self.Ax3 , self.Ax4 , self.Ax5 = 0. , 0. , 0. , 0., 0. , 0.
        self.Ay0 , self.Ay1 , self.Ay2 , self.Ay3 , self.Ay4 , self.Ay5 = 0. , 0. , 0. , 0., 0. , 0.
        self.Az0 , self.Az1 , self.Az2 , self.Az3 , self.Az4 , self.Az5 , self.Az6 = 0. , 0. , 0. , 0., 0. , 0. , 0.    


    def compute_coefficient(self,x0, dx0, ddx0, y0, dy0, ddy0, z0, dz0,ddz0 ,x1,y1,z1, t0 , t1 , h) :
        epsilon = 0.00
        t2 = t1
        t3 = t0
        t1 -= 2*epsilon
        t0 -= epsilon
        self.Ax5 = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12 *
            x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax4 = (30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 +
            2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax3 = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0 *
            t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax2 = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1 **
                2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax1 = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2 *
                t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ax0 = (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0 **
            2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

        self.Ay5 = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12 *
            y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay4 = (30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 +
            2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay3 = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0 *
            t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay2 = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1 **
                2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay1 = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2 *
                t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        self.Ay0 = (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0 **
            2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))   
        # Version 1 : 2D (z0 = z1 = 0)
        #self.Az6 = -h/((t2/2)**3*(t2 - t2/2)**3)
        #self.Az5 = (3*t2*h)/((t2/2)**3*(t2 - t2/2)**3)
        #self.Az4 = -(3*t2**2*h)/((t2/2)**3*(t2 - t2/2)**3)
        #self.Az3 = (t2**3*h)/((t2/2)**3*(t2 - t2/2)**3)
        #self.Az2 = 0
        #self.Az1 = 0
        #self.Az0 = 0
        
        # Version 2 : 3D (z0 and z1 != 0)
        self.Az6 =  (32.*z0 + 32.*z1 - 64.*h)/(t2**6)
        self.Az5 = - (102.*z0 + 90.*z1 - 192.*h)/(t2**5)
        self.Az4 = (111.*z0 + 81.*z1 - 192.*h)/(t2**4)
        self.Az3 = - (42.*z0 + 22.*z1 - 64.*h)/(t2**3)
        self.Az2 = 0
        self.Az1 = 0
        self.Az0 = z0
        
    def evaluate(self,t) :
        x = self.Ax0 + self.Ax1*t + self.Ax2*t**2 + self.Ax3*t**3 + self.Ax4*t**4 + self.Ax5*t**5
        y = self.Ay0 + self.Ay1*t + self.Ay2*t**2 + self.Ay3*t**3 + self.Ay4*t**4 + self.Ay5*t**5
        z = self.Az0 + self.Az1*t + self.Az2*t**2 + self.Az3*t**3 + self.Az4*t**4 + self.Az5*t**5 + self.Az6*t**6
        return np.array([x,y,z])
    
    
    def derivate(self,t , indice ) :
        '''Compute derivative of the polynomial curve

    Args:
        - t (float): time
        - indice (int): f indice derivative : 0 --> poly(t) ; 1 --> poly'(t) ; ...
    '''
        
       
        if indice == 0 :
            return self.evaluate(t)
        elif indice == 1 :
            x =  self.Ax1 + 2*self.Ax2*t + 3*self.Ax3*t**2 + 4*self.Ax4*t**3 + 5*self.Ax5*t**4
            y =  self.Ay1 + 2*self.Ay2*t + 3*self.Ay3*t**2 + 4*self.Ay4*t**3 + 5*self.Ay5*t**4
            z =  self.Az1 + 2*self.Az2*t + 3*self.Az3*t**2 + 4*self.Az4*t**3 + 5*self.Az5*t**4 + 6*self.Az6*t**5
            return np.array([x,y,z])
        elif indice == 2 :
            x =  2*self.Ax2 + 6*self.Ax3*t + 12*self.Ax4*t**2 + 20*self.Ax5*t**3
            y =  2*self.Ay2 + 6*self.Ay3*t + 12*self.Ay4*t**2 + 20*self.Ay5*t**3
            z =  2*self.Az2 + 6*self.Az3*t + 12*self.Az4*t**2 + 20*self.Az5*t**3 + 30*self.Az6*t**4
            return np.array([x,y,z])
        else :
            return np.array([0.,0.,0.])
    





class Foot_trajectory_generator(object):
    '''This class provide adaptative 3d trajectory for a foot from (x0,y0) to (x1,y1) using polynoms

    A foot trajectory generator that handles the generation of a 3D trajectory
    with a 5th order polynomial to lead each foot from its location at the start of
    its swing phase to its final location that has been decided by the FootstepPlanner

    Args:
        - h (float): the height at which feet should be raised at the apex of the wing phase
        - time_adaptative_disabled (float): how much time before touchdown is the desired position locked
    '''

    def __init__(self, h=0.03, time_adaptative_disabled=0.200, x_init=0.0, y_init=0.0):
        # maximum heigth for the z coordonate
        self.h = h

        # when there is less than this time for the trajectory to finish, disable adaptative (using last computed coefficients)
        # this parameter should always be a positive number less than the durration of a step
        self.time_adaptative_disabled = time_adaptative_disabled

        # memory of the last coeffs
        self.poly_curve = polynomial_curve()        
        
        self.x1 = x_init
        self.y1 = y_init

        self.z_margin = 0.05
        self.T = 0.16

        # heightMap
        # --> To turn into a class 
        self.x = np.linspace(-2.0,2.0,100)
        self.y = np.linspace(-1.0,2.0,100)
        #self.heightMap = np.load("solo3D/tools/heightMap_z.npy")
        self.heightMap = np.zeros((100,100))

        # Discretization of the the inital 6D curve 
        self.N_int = 5
        self.ptsTime = None

        #dimension of our problem (here 3 as our curve is 3D)
        self.dim = 3
        self.pD = problem_definition(self.dim)
        self.pD.degree = 8

        #values are 0 by default, so if the constraint is zero this can be skipped
        self.pD.init_pos = np.array([[x_init,y_init, 0.]]).T
        self.pD.end_pos   = np.array([[0., 0., 0.]]).T
        self.pD.init_vel = np.array([[0., 0., 0.]]).T
        self.pD.init_acc = np.array([[0., 0., 0.]]).T
        self.pD.end_vel = np.array([[0., 0., 0.]]).T
        self.pD.end_acc = np.array([[0., 0., 0.]]).T

        self.pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL  | constraint_flag.END_VEL  | constraint_flag.INIT_ACC  | constraint_flag.END_ACC

        #generates the variable bezier curve with the parameters of problemDefinition
        self.problem = setup_control_points(self.pD)
        self.variableBezier = self.problem.bezier()

        self.N_split = 2
        self.Dt_split = np.linspace(0,1 ,self.N_split,endpoint = False)[1:]    
        # Middle of the interval  
        #self.Dt_split_interval =   np.linspace(0,self.T,self.N_int,endpoint = False)[:] + 0.5*self.T/self.N_int 

        self.piecewiseCurve = self.variableBezier.split(self.Dt_split.T)

        #nb of waypoints for each constrainted curve, ( = dim of the Bezier curve + 1)
        self.nbWaypoints = self.problem.numVariables
        self.num_curves = self.piecewiseCurve.num_curves()
        self.nConstraints = self.nbWaypoints*self.num_curves

        # 1 constraints for each waypoint :  z > height map
        self.ineqMatrix = np.zeros((self.nConstraints, self.problem.numVariables*3))
        self.ineqVector = np.zeros(self.nConstraints )        

        self.res = None
    
    def find_nearest(self,xt , yt):
        idx = (np.abs(self.x - xt)).argmin()
        idy = (np.abs(self.y - yt)).argmin()
    
        return idx, idy


    def get_next_foot(self, x0, dx0, ddx0, y0, dy0, ddy0, z0, dz0,ddz0 , x1, y1,z1, t0, t1,  dt , k):
            '''how to reach a foot position (here using polynomials profiles)'''

            epsilon = 0.00
            t2 = t1
            t3 = t0
            t1 -= 2*epsilon
            t0 -= epsilon
            delta_t = t1 - t0

            self.poly_curve.compute_coefficient(x0, dx0, ddx0, y0, dy0, ddy0, z0, dz0,ddz0 , x1,y1,z1,t0 , t1 , self.h)

            # Final and initial conditions        
            # self.pD.init_pos = np.array([[x0,y0, z0]]).T
            # self.pD.init_vel = delta_t*np.array([[dx0, dy0, dz0]]).T
            # self.pD.init_acc = (delta_t**2) * np.array([[ddx0, ddy0, ddz0]]).T

            self.pD.init_pos = np.array([[x0,y0, self.poly_curve.derivate(t0,0)[2]   ]]).T
            self.pD.init_vel = delta_t*np.array([[dx0, dy0, self.poly_curve.derivate(t0,1)[2] ]]).T
            self.pD.init_acc = (delta_t**2) * np.array([[ddx0, ddy0, self.poly_curve.derivate(t0,2)[2] ]]).T
            self.pD.end_pos   = np.array([[x1, y1, z1]]).T
    
            self.pD.end_vel = np.array([[0., 0., 0.]]).T
            self.pD.end_acc = np.array([[0., 0., 0.]]).T

            self.pD.flag = constraint_flag.END_POS | constraint_flag.INIT_POS | constraint_flag.INIT_VEL  | constraint_flag.END_VEL  | constraint_flag.INIT_ACC  | constraint_flag.END_ACC

            #generates the variable bezier curve with the parameters of problemDefinition
            self.problem = setup_control_points(self.pD)
            self.variableBezier = self.problem.bezier()

            # Split the curves : t bezier : 0 --> 1
            # t polynomial curve : t0 --> t1
            # self.Dt_split = np.linspace(0,1 ,self.N_int,endpoint = False)[1:]  
            # self.piecewiseCurve = self.variableBezier.split(self.Dt_split.T)

            # Middle of the interval  
            #self.Dt_split_interval =   np.linspace(t0,t1,self.N_int,endpoint = False)[:] + 0.5*(t1-t0)/self.N_int 

            adaptative_mode = (t1 - t0) > self.time_adaptative_disabled
            if(adaptative_mode):       
                 
                # Generate the points for the least square problem
                # t bezier : 0 --> 1
                # t polynomial curve : t0 --> t1
                self.ptsTime = [(self.poly_curve.evaluate(t0 + (t1-t0)*t_b ),t_b) for t_b in np.linspace(0,1,self.N_int)]
                # Inequality constraints
                k = 0
                for i in range(self.num_curves) :
                    
                    # Approximation of x(t) for the i-th interval at T_interval/2 (only i) to retrieve the closest constraints 
                    # It can be more precise and evaluated for each point of the interval (i,j)
                    # x_approx = poly_5D(x0,x1,t1-t0,self.Dt_split_interval[i])   
                    # y_approx = poly_5D(x0,x1,t1-t0,self.Dt_split_interval[i])    
                    idx , idy = self.find_nearest(0.0 , 0.0)
                    z_limit = self.heightMap[idx,idy] 
                   
                    if z_limit > 0 :
                        z_limit += self.z_margin                    
                    
                    for j in range(self.nbWaypoints) :
                        wayPoint = self.piecewiseCurve.curve_at_index(i).waypointAtIndex(j)
                        
                        
                        self.ineqMatrix[k,:] = -wayPoint.B()[2,:]
                        self.ineqVector[k] =  wayPoint.c()[2] - z_limit                         
                            
                        k = k+1
                # RUN QP optimization

                A, b = genCost(self.variableBezier, self.ptsTime)
                #regularization matrix 
                reg = identity(A.shape[1]) * 0.00
                # self.res = quadprog_solve_qp(A+reg, b,  G=self.ineqMatrix, h = self.ineqVector )
                self.res = quadprog_solve_qp(A+reg, b)



                fitBezie = self.variableBezier.evaluate(self.res.reshape((-1,1)) ) 

            else:
                fitBezie = self.variableBezier.evaluate(self.res.reshape((-1,1)) ) 


            # get the next point
            ev = t0 + dt
            evz = t3+dt
            x1 = self.x1
            y1 = self.y1
            
            t_b = (ev - t0)/(t1 - t0)

            Az3 = self.poly_curve.Az3
            Az4 = self.poly_curve.Az4
            Az5 = self.poly_curve.Az5
            Az6 = self.poly_curve.Az6
            
            # z0 = Az3*evz**3 + Az4*evz**4 + Az5*evz**5 + Az6*evz**6
            # dz0 = 3*Az3*evz**2 + 4*Az4*evz**3 + 5*Az5*evz**4 + 6*Az6*evz**5
            # ddz0 = 2*3*Az3*evz + 3*4*Az4*evz**2 + 4*5*Az5*evz**3 + 5*6*Az6*evz**4


            if (t3 < epsilon) or (t3 > (t2-epsilon)):

                z0 = fitBezie(t_b)[2]
                dz0 = fitBezie.derivate(t_b,1)[2]/delta_t
                ddz0 = fitBezie.derivate(t_b,2)[2]/delta_t**2
                
                return [x0, 0.0, 0.0,  y0, 0.0, 0.0,  z0, dz0, ddz0, self.x1, self.y1]
            else:
                print("t0 : " , t0)               


                x0 , y0 , z0 = fitBezie(t_b)
                dx0 , dy0 , dz0 = fitBezie.derivate(t_b,1)/delta_t
                ddx0 , ddy0 , ddz0 = fitBezie.derivate(t_b,2)/delta_t**2
                

                # z0 = Az3*evz**3 + Az4*evz**4 + Az5*evz**5 + Az6*evz**6
                # dz0 = 3*Az3*evz**2 + 4*Az4*evz**3 + 5*Az5*evz**4 + 6*Az6*evz**5
                # ddz0 = 2*3*Az3*evz + 3*4*Az4*evz**2 + 4*5*Az5*evz**3 + 5*6*Az6*evz**4

                print("x0,y0,z0          : " , [x0,y0,z0])
                print("x0,y0,z0 (poly)   : " , self.poly_curve.evaluate(t0+dt))

                

                

                return [x0, dx0, ddx0,  y0, dy0, ddy0,  z0, dz0, ddz0, self.x1, self.y1]
