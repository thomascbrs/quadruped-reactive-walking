# coding: utf8

import Joystick
#from matplotlib import pyplot as plt
import time
import numpy as np
import math
import pinocchio as pin
import tsid
import utils_mpc

# Separate classes for planner 
from solo3D.FootStepPlanner import FootStepPlanner
from solo3D.FootTrajectoryGenerator import FootTrajectoryGenerator
from solo3D.GaitPlanner import GaitPlanner
from solo3D.LoggerPlanner import LoggerPlanner

from solo3D.FootTrajectoryGeneratorBezier import FootTrajectoryGeneratorBezier
from solo3D.tools.HeightMap import HeightMap

import libquadruped_reactive_walking as la
np.set_printoptions(precision=3, linewidth=300)

class PyPlanner:
    """Planner that outputs current and future locations of footsteps, the reference trajectory of the base and
    the position, velocity, acceleration commands for feet in swing phase based on the reference velocity given by
    the user and the current position/velocity of the base in TSID world
    """

    def __init__(self, dt, dt_tsid, T_gait, T_mpc, k_mpc, on_solo8, h_ref, fsteps_init , N_SIMULATION ):

        # Time step of the contact sequence
        self.dt = dt

        # Time step of TSID
        self.dt_tsid = dt_tsid

        # Gait duration
        self.T_gait = T_gait
        self.T_mpc = T_mpc

        # Reference height for the trunk
        self.h_ref = h_ref

        # Number of TSID iterations for one iteration of the MPC
        self.k_mpc = k_mpc

        # Feedback gain for the feedback term of the planner
        self.k_feedback = 0.03

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        # Value of the gravity acceleartion
        self.g = 9.81

        # Value of the maximum allowed deviation due to leg length
        self.L = 0.155

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(self.T_gait/self.dt)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)
        self.is_static = False  # Flag for static gait
        self.q_static = np.zeros((19, 1))
        self.RPY_static = np.zeros((3, 1))

        # Foot trajectory generator
        max_height_feet = 0.05
        t_lock_before_touchdown = 0.00

        self.goals = fsteps_init.copy()  # Store 3D target position for feet
        self.vgoals = np.zeros((3, 4))  # Store 3D target velocity for feet
        self.agoals = np.zeros((3, 4))  # Store 3D target acceleration for feet

        # Load Heightmap 
        path_ = "solo3D/heightmap/"
        surface_margin = 0.05
        self.heightMap = HeightMap(path_ , surface_margin)
       
        # C++ class
        self.Cplanner = la.Planner(dt, dt_tsid, T_gait, T_mpc, k_mpc, on_solo8, h_ref, fsteps_init)

        # Log values from planner 
        self.logger = LoggerPlanner(self.dt , N_SIMULATION)

        # FootStepPlanner
        self.footStepPlanner = FootStepPlanner(dt,T_gait , h_ref , self.k_feedback , self.g , self.L , on_solo8 , k_mpc)

        #FootTrajectoryGenerator
        #self.footTrajectoryGenerator = FootTrajectoryGenerator(T_gait , dt_tsid ,k_mpc ,  fsteps_init)
        self.footTrajectoryGenerator = FootTrajectoryGeneratorBezier(T_gait , dt_tsid ,k_mpc ,  fsteps_init , self.heightMap)

        # Gait planner 
        self.gaitPlanner = GaitPlanner(T_gait , self.dt , T_mpc)
        self.gait_test = self.gait



        # Log the value from c++
        self.xref_cpp = np.zeros((12, 1 + self.n_steps))        
        self.fsteps_cpp = np.full((self.gait.shape[0], 13), 0.)
        self.gait_cpp = fsteps_init.copy() 
        self.goals_cpp = np.zeros((3, 4))
        self.vgoals_cpp = np.zeros((3, 4))
        self.agoals_cpp = np.zeros((3, 4))
   

    def run_planner(self, k, k_mpc, q, v, b_vref, h_estim, z_average, joystick=None , device = None):

        # Get the reference velocity in world frame (given in base frame)
        self.RPY = utils_mpc.quaternionToRPY(q[3:7, 0])
        c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        vref = b_vref.copy()
        vref[0:2, 0:1] = np.array([[c, -s], [s, c]]) @ b_vref[0:2, 0:1]

        joystick_code = 0
        if joystick is not None:
            if joystick.northButton:
                joystick_code = 1
                self.new_desired_gait = self.pacing_gait()
                self.is_static = False
                joystick.northButton = False
            elif joystick.eastButton:
                joystick_code = 2
                self.new_desired_gait = self.bounding_gait()
                self.is_static = False
                joystick.eastButton = False
            elif joystick.southButton:
                joystick_code = 3
                self.new_desired_gait = self.trot_gait()
                self.is_static = False
                joystick.southButton = False
            elif joystick.westButton:
                joystick_code = 4
                self.new_desired_gait = self.static_gait()
                self.is_static = True
                self.q_static[0:7, 0:1] = q.copy()
                joystick.westButton = False


        # C++ Planner
        self.Cplanner.run_planner(k, q, v, b_vref, np.double(h_estim), np.double(z_average), joystick_code)        

        if ((k % k_mpc) == 0):
        # Move one step further in the gait 
            self.gait = self.gaitPlanner.roll(k , self.fsteps)

        # Compute ref state
        self.xref = self.footStepPlanner.getRefStates(q, v, vref, z_average)
        # Compute fsteps
        self.fsteps = self.footStepPlanner.compute_footsteps(k,q, v, vref, joystick.reduced,self.gaitPlanner)
        # Compute foot trajectory
        self.goals , self.vgoals  , self.agoals  = self.footTrajectoryGenerator.update_foot_trajectory( k , self.fsteps, self.gaitPlanner)


        # C++ Planner
        # self.Cplanner.run_planner(k, q, v, b_vref, np.double(h_estim), np.double(z_average), joystick_code)        

        self.xref_cpp = self.Cplanner.get_xref()
        self.fsteps_cpp = self.Cplanner.get_fsteps()
        self.gait_cpp = self.Cplanner.get_gait()
        self.goals_cpp = self.Cplanner.get_goals()
        self.vgoals_cpp = self.Cplanner.get_vgoals()
        self.agoals_cpp = self.Cplanner.get_agoals()

        # self.xref = self.Cplanner.get_xref()
        # self.fsteps = self.Cplanner.get_fsteps()
        # self.gait = self.Cplanner.get_gait()
        # self.goals = self.Cplanner.get_goals()
        # self.vgoals = self.Cplanner.get_vgoals()
        # self.agoals = self.Cplanner.get_agoals()
        
        

        # Comparison c++ pyhton
        # diff_xref = np.sum((self.xref_cpp - self.xref ) > 10e-5 )
        # diff_gait = np.sum((self.gait_cpp - self.gait ) > 10e-5 )
        # diff_pos = np.sum((self.goals_cpp  - self.goals ) > 10e-5 )
        # diff_vel = np.sum((self.vgoals_cpp  - self.vgoals ) > 10e-5 )
        # diff_acc = np.sum((self.agoals_cpp  - self.agoals ) > 10e-5 )
        # diff_fsteps = np.sum((self.fsteps_cpp - self.fsteps ) > 10e-5 )

        # if diff_gait != 0 :
        #     print("----------------GAIT PB----------------")
        # if diff_pos != 0 :
        #     print("----------------POS PB----------------")
        #     print(self.goals_cpp)
        #     print(self.goals)
        # if diff_vel != 0 :
        #     print("----------------POS VEL----------------")
        #     print(self.vgoals_cpp)
        #     print(self.vgoals)
        # if diff_acc != 0 :
        #     print("----------------ACC VEL----------------")
        # if diff_fsteps != 0 :
        #     print("----------------FSTEPS VEL----------------")
        #     if diff_xref != 0 :
        #         print("----------------XREF VEL----------------")


        ##########
        # LOGGER 
        ##########
        self.logger.log_feet( k , device , self.goals , self.vgoals, self.agoals, self.fsteps)

        return 0

    

