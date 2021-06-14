# coding: utf8
import numpy as np
import utils_mpc
import time
from time import perf_counter as clock

from QP_WBC import wbc_controller
import MPC_Wrapper
import pybullet as pyb
import pinocchio as pin
from solopython.utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
import libquadruped_reactive_walking as lqrw

from solo3D.FootTrajectoryGeneratorBezier import FootTrajectoryGeneratorBezier
from solo3D.StatePlanner import StatePlanner
from solo3D.LoggerPlanner import LoggerPlanner
from solo3D.SurfacePlannerWrapper import SurfacePlanner_Wrapper

from solo3D.tools.vizualization import PybVisualizationTraj
from example_robot_data import load

from solo3D.tools.geometry import inertiaTranslation
from time import perf_counter as clock

ENV_URDF = "/local/users/frisbourg/install/share/hpp_environments/urdf/Solo3D/stairs_rotation.urdf"
HEIGHTMAP = "/local/users/frisbourg/install/share/hpp_environments/heightmaps/Solo3D/stairs_rotation.pickle"


class Result:
    """Object to store the result of the control loop
    It contains what is sent to the robot (gains, desired positions and velocities,
    feedforward torques)"""

    def __init__(self):

        self.P = 0.0
        self.D = 0.0
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)


class dummyHardware:
    """Fake hardware for initialisation purpose"""

    def __init__(self):

        pass

    def imu_data_attitude(self, i):

        return 0.0


class dummyDevice:
    """Fake device for initialisation purpose"""

    def __init__(self):

        self.hardware = dummyHardware()


class Controller:

    def __init__(self, q_init, envID, velID, dt_wbc, dt_mpc, k_mpc, t, T_gait, T_mpc, N_SIMULATION, type_MPC,
                 use_flat_plane, predefined_vel, enable_pyb_GUI, kf_enabled, N_gait, isSimulation):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            envID (int): identifier of the current environment to be able to handle different scenarios
            velID (int): identifier of the current velocity profile to be able to handle different scenarios
            dt_wbc (float): time step of the whole body control
            dt_mpc (float): time step of the MPC
            k_mpc (int): number of iterations of inverse dynamics for one iteration of the MPC
            t (float): time of the simulation
            T_gait (float): duration of one gait period in seconds
            T_mpc (float): duration of mpc prediction horizon
            N_SIMULATION (int): number of iterations of inverse dynamics during the simulation
            type_mpc (bool): True to have PA's MPC, False to have Thomas's MPC
            use_flat_plane (bool): to use either a flat ground or a rough ground
            predefined_vel (bool): to use either a predefined velocity profile or a gamepad
            enable_pyb_GUI (bool): to display PyBullet GUI or not
            kf_enabled (bool): complementary filter (False) or kalman filter (True)
            N_gait (int): number of spare lines in the gait matrix
            isSimulation (bool): if we are in simulation mode
        """

        ########################################################################
        #                        Parameters definition                         #
        ########################################################################

        # Lists to log the duration of 1 iteration of the MPC/TSID
        self.t_list_filter = [0] * int(N_SIMULATION)
        self.t_list_planner = [0] * int(N_SIMULATION)
        self.t_list_mpc = [0] * int(N_SIMULATION)
        self.t_list_wbc = [0] * int(N_SIMULATION)
        self.t_list_loop = [0] * int(N_SIMULATION)

        self.t_list_InvKin = [0] * int(N_SIMULATION)
        self.t_list_QPWBC = [0] * int(N_SIMULATION)

        # Init joint torques to correct shape
        self.jointTorques = np.zeros((12, 1))

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Enable/Disable Gepetto viewer
        self.enable_gepetto_viewer = True

        # Enable/Disable perfect estimator
        perfectEstimator = True
        if not isSimulation:
            perfectEstimator = False  # Cannot use perfect estimator if we are running on real robot

        # Initialisation of the solo model/data and of the Gepetto viewer
        self.solo, self.fsteps_init, self.h_init = utils_mpc.init_robot(q_init, self.enable_gepetto_viewer)

        # Create Joystick, FootstepPlanner, Logger and Interface objects
        self.joystick, self.logger, self.estimator = utils_mpc.init_objects(
            dt_wbc, dt_mpc, N_SIMULATION, k_mpc, T_gait, type_MPC, predefined_vel, self.h_init, kf_enabled,
            perfectEstimator)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        self.h_ref = self.h_init
        self.q = np.zeros((19, 1))
        self.q[0:7, 0] = np.array([0.0, 0.0, self.h_ref, 0.0, 0.0, 0.0, 1.0])
        self.q[7:, 0] = q_init
        self.v = np.zeros((18, 1))
        self.b_v = np.zeros((18, 1))
        self.o_v_filt = np.zeros((18, 1))

        # self.statePlanner = lqrw.StatePlanner()
        # self.statePlanner.initialize(dt_mpc, T_mpc, self.h_ref)

        self.gait = lqrw.Gait()
        self.gait.initialize(dt_mpc, T_gait, T_mpc, N_gait)

        shoulders = np.zeros((3, 4))
        shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
        shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
        # self.footstepPlanner = lqrw.FootstepPlanner()
        # self.footstepPlanner.initialize(dt_mpc, T_mpc, self.h_ref, shoulders.copy(), self.gait, N_gait)

        # self.footTrajectoryGenerator = lqrw.FootTrajectoryGenerator()
        # self.footTrajectoryGenerator.initialize(0.05, 0.07, self.fsteps_init.copy(), shoulders.copy(),
        #                                         dt_wbc, k_mpc, self.gait)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        # First argument to True to have PA's MPC, to False to have Thomas's MPC
        self.enable_multiprocessing = False
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(type_MPC, dt_mpc, np.int(T_mpc/dt_mpc),
                                                   k_mpc, T_mpc, N_gait, self.q, self.enable_multiprocessing)

        # ForceMonitor to display contact forces in PyBullet with red lines
        # import ForceMonitor
        # myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

        # Define the default controller
        self.myController = wbc_controller(dt_wbc, N_SIMULATION)

        self.envID = envID
        self.velID = velID
        self.dt_wbc = dt_wbc
        self.dt_mpc = dt_mpc
        self.k_mpc = k_mpc
        self.t = t
        self.T_gait = T_gait
        self.T_mpc = T_mpc
        self.N_SIMULATION = N_SIMULATION
        self.type_MPC = type_MPC
        self.use_flat_plane = use_flat_plane
        self.predefined_vel = predefined_vel
        self.enable_pyb_GUI = enable_pyb_GUI

        self.k = 0

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        self.error_flag = 0
        self.q_security = np.array([np.pi*0.4, np.pi*80/180, np.pi] * 4)

        # Interface with the PD+ on the control board
        self.result = Result()

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.q_mes = q_init
        dDevice.v_mes = np.zeros(12)
        dDevice.baseLinearAcceleration = np.zeros(3)
        dDevice.baseAngularVelocity = np.zeros(3)
        dDevice.baseOrientation = np.array([0.0, 0.0, 0.0, 1.0])
        dDevice.dummyPos = np.array([0.0, 0.0, q_init[2]])
        dDevice.b_baseVel = np.zeros(3)

        # Solo3D python class
        n_surface_configs = 3
        self.surfacePlanner = SurfacePlanner_Wrapper(ENV_URDF, T_gait, N_gait, n_surface_configs)

        self.statePlanner = StatePlanner(dt_mpc, T_mpc, self.h_ref, HEIGHTMAP, n_surface_configs, T_gait)

        # List of floor surface initialisation
        self.footstepPlanner = lqrw.FootstepPlannerQP()

        self.footstepPlanner.initialize(dt_mpc, T_mpc, self.h_ref, k_mpc, dt_wbc, shoulders.copy(), self.gait, N_gait, self.surfacePlanner.floor_surface)
        self.footTrajectoryGenerator = FootTrajectoryGeneratorBezier(T_gait, dt_wbc, k_mpc,  self.fsteps_init, self.gait, self.footstepPlanner)
        # Pybullet Trajectory
        self.pybVisualizationTraj = PybVisualizationTraj(self.gait, self.footstepPlanner, self.statePlanner,  self.footTrajectoryGenerator, enable_pyb_GUI, ENV_URDF)

        # Pybullet Trajectory
        self.pybVisualizationTraj = PybVisualizationTraj(self.gait, self.footstepPlanner, self.statePlanner,  self.footTrajectoryGenerator, enable_pyb_GUI, ENV_URDF)

        # pinocchio model and data, CoM and Inertia estimation for MPC
        robot = load('solo12')
        self.data = robot.data.copy()  # for velocity estimation (forward kinematics)
        self.model = robot.model.copy()  # for velocity estimation (forward kinematics)
        self.q_neutral = pin.neutral(self.model).reshape((19, 1))  # column vector

        # Log values for planner
        self.loggerPlanner = LoggerPlanner(dt_mpc, N_SIMULATION, T_gait, k_mpc)

        self.compute(dDevice)

    def compute(self, device):
        """Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """

        t_start = time.time()

        # Update the reference velocity coming from the gamepad
        self.joystick.update_v_ref(self.k, self.velID)

        # Process state estimator
        self.estimator.run_filter(self.k, self.gait.getCurrentGait(),
                                  device, self.footTrajectoryGenerator.getFootPosition())
        t_filter = time.time()

        # Update state for the next iteration of the whole loop
        if self.k > 1:
            self.q[:, 0] = self.estimator.q_filt[:, 0]
            oMb = pin.SE3(pin.Quaternion(self.q[3:7, 0:1]), self.q[0:3, 0:1])
            self.v[0:3, 0:1] = oMb.rotation @ self.estimator.v_filt[0:3, 0:1]
            self.v[3:6, 0:1] = oMb.rotation @ self.estimator.v_filt[3:6, 0:1]
            self.v[6:, 0] = self.estimator.v_filt[6:, 0]

            # Update estimated position of the robot
            self.v_estim[0:3, 0:1] = oMb.rotation.transpose() @ self.joystick.v_ref[0:3, 0:1]
            self.v_estim[3:6, 0:1] = oMb.rotation.transpose() @ self.joystick.v_ref[3:6, 0:1]
            self.q_estim[:, 0] = pin.integrate(self.solo.model,
                                               self.q, self.v_estim * self.myController.dt)
            self.yaw_estim = (utils_mpc.quaternionToRPY(self.q_estim[3:7, 0]))[2, 0]
            self.roll_estim = (utils_mpc.quaternionToRPY(self.q_estim[3:7, 0]))[0, 0]
            self.pitch_estim = (utils_mpc.quaternionToRPY(self.q_estim[3:7, 0]))[1, 0]
        else:
            self.yaw_estim = 0.0
            self.roll_estim = 0.0
            self.pitch_estim = 0.0
            self.q_estim = self.q.copy()
            oMb = pin.SE3(pin.Quaternion(self.q[3:7, 0:1]), self.q[0:3, 0:1])
            self.v_estim = self.v.copy()

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, self.q[0:7, 0:1], self.joystick.joystick_code)
        cgait = self.gait.getCurrentGait()

        o_v_ref = np.zeros((6, 1))
        o_v_ref[0:3, 0:1] = oMb.rotation @ self.joystick.v_ref[0:3, 0:1]
        o_v_ref[3:6, 0:1] = oMb.rotation @ self.joystick.v_ref[3:6, 0:1]

        # Compute target footstep based on current and reference velocities
        # targetFootstep = self.footstepPlanner.computeTargetFootstep(self.q[0:7, 0:1], self.v[0:6, 0:1].copy(), o_v_ref)

        # Update pos, vel and acc references for feet
        # TODO: Make update take as parameters current gait, swing phase duration and remaining time
        # self.footTrajectoryGenerator.update(self.k, targetFootstep)

        # Retrieve data from C++ planner

        # Run state planner (outputs the reference trajectory of the CoM / base)
        # self.statePlanner.computeReferenceStates(self.q[0:7, 0:1], self.v[0:6, 0:1].copy(), o_v_ref, 0.0)
        # Result can be retrieved with self.statePlanner.getReferenceStates()
        # xref = self.statePlanner.getReferenceStates()
        # fsteps = self.footstepPlanner.getFootsteps()

        ################
        # solo3D python
        ################
        # Update footsteps if new contact phase
        new_step = self.k % self.k_mpc == 0 and self.gait.isNewPhase()
        if new_step and self.k != 0:
            self.footstepPlanner.updateNewContact()

            # New phase, results from MIP should be available
            # Only usefull for multiprocessing
            if self.surfacePlanner.first_iteration:
                self.surfacePlanner.first_iteration = False
            else:
                self.surfacePlanner.update_latest_results()

        targetFootstep = self.footstepPlanner.computeTargetFootstep(self.k, self.q[0:7, 0:1], self.v[0:6, 0:1].copy(
        ), o_v_ref, self.surfacePlanner.potential_surfaces, self.surfacePlanner.selected_surfaces, self.surfacePlanner.mip_success, self.surfacePlanner.mip_iteration)
        fsteps = self.footstepPlanner.getFootsteps()

        # Compute target footstep based on current and reference velocities
        self.statePlanner.computeReferenceStates(self.q[:7, 0].copy(), self.v[:6, 0].copy(), o_v_ref[:, 0].copy(), new_step)
        xref = self.statePlanner.getReferenceStates()

        # Compute foot trajectory
        self.footTrajectoryGenerator.update(self.k, targetFootstep, device, self.q, self.v)

        if new_step:
            self.surfacePlanner.run(self.statePlanner.configs, cgait, targetFootstep, o_v_ref)
            if not self.surfacePlanner.multiprocessing:
                self.pybVisualizationTraj.updateSl1M_target(self.surfacePlanner.all_feet_pos)

        t_planner = time.time()

        # Process MPC once every k_mpc iterations of TSID
        if (self.k % self.k_mpc) == 0:
            try:
                # Take into account only the leg configuration :
                self.q_neutral[7:, :] = self.q[7:, :]
                # Inertia matrix is computed in c = (0,0,0) with ccrba (frame of the body_0)
                pin.ccrba(self.model, self.data, self.q_neutral, self.v)
                # Position of CoM in frame body_0
                CoM_offset = pin.centerOfMass(self.model, self.data, self.q_neutral).reshape((3, 1))
                # Compute the inertia matrix in CoM frame (body_0 expressed in CoM frame --> -CoM_offset )
                inertia_CoM = inertiaTranslation(self.data.Ig.inertia, -CoM_offset, 2.5)
                # xref is given for the position of body_0, apply offset for CoM
                xref_CoM = xref.copy()
                xref_CoM[:3, :] += CoM_offset
                # Update inertia matrix of MPC
                self.mpc_wrapper.mpc.I = inertia_CoM
                self.mpc_wrapper.solve(self.k, xref_CoM, fsteps, cgait)

            except ValueError:
                print("MPC Problem")

        # Retrieve reference contact forces
        self.x_f_mpc = self.mpc_wrapper.get_latest_result()

        t_mpc = time.time()

        # Target state for the whole body control
        self.x_f_wbc = (self.x_f_mpc[:, 0]).copy()
        self.x_f_wbc[0] = self.q_estim[0, 0]
        self.x_f_wbc[1] = self.q_estim[1, 0]
        # Get Z value using next xref
        self.x_f_wbc[2] = - (self.dt_mpc - self.dt_wbc) * (xref[2, 2] - xref[2, 1])/self.dt_mpc + xref[2, 1]
        self.x_f_wbc[3] = - (self.dt_mpc - self.dt_wbc) * (xref[3, 2] - xref[3, 1])/self.dt_mpc + xref[3, 1]
        self.x_f_wbc[4] = - (self.dt_mpc - self.dt_wbc) * (xref[4, 2] - xref[4, 1])/self.dt_mpc + xref[4, 1]
        self.x_f_wbc[5] = self.yaw_estim
        self.x_f_wbc[6:12] = xref[6:, 1]

        # Whole Body Control
        # If nothing wrong happened yet in the WBC controller
        if (not self.myController.error) and (not self.joystick.stop):
            # Get velocity in base frame for pinocchio
            self.b_v[0:3, 0:1] = oMb.rotation.transpose() @ self.v[0:3, 0:1]
            self.b_v[3:6, 0:1] = oMb.rotation.transpose() @ self.v[3:6, 0:1]
            self.b_v[6:, 0] = self.v[6:, 0]

            # Run InvKin + WBC QP
            self.myController.compute(self.q, self.b_v, self.x_f_wbc[:12],
                                      self.x_f_wbc[12:], cgait[0, :],
                                      self.footTrajectoryGenerator.getFootPosition(),
                                      self.footTrajectoryGenerator.getFootVelocity(),
                                      self.footTrajectoryGenerator.getFootAcceleration())

            # Quantities sent to the control board
            self.result.P = 2.0 * np.ones(12)
            self.result.D = 0.2 * np.ones(12)
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:, 0]
            self.result.tau_ff[:] = 0.5 * self.myController.tau_ff

        t_wbc = time.time()

        # Security check
        self.security_check()

        # Update PyBullet camera
        self.pybVisualizationTraj.update(self.k, device)

        # Logs
        self.log_misc(t_start, t_filter, t_planner, t_mpc, t_wbc)

        # Log Planner
        self.loggerPlanner.log_mpc(self.k, self.x_f_mpc)
        self.loggerPlanner.log_feet(self.k, device, self.footTrajectoryGenerator.getFootPosition(),
                                    self.footTrajectoryGenerator.getFootVelocity(),
                                    self.footTrajectoryGenerator.getFootAcceleration(), targetFootstep,
                                    self.q, self.v)
        self.loggerPlanner.log_state(self.k, self.q[:7], self.v[:6], o_v_ref, self.joystick.v_ref[0:6, 0:1], oMb.rotation,  xref)

        # Increment loop counter
        self.k += 1

        return 0.0

    def pyb_camera(self, device):
        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if self.k > 10 and self.enable_pyb_GUI:
            # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
            #                                cameraTargetPosition=[1.0, 0.3, 0.25])
            # pyb.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=215, cameraPitch=-25.9,
            #                                cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])
            pass

        # TODO : One class for pybullet visualization
        # self.pybVisualizationTraj.vizuFootTraj(self.k , self.fsteps , self.gait , device)

    def security_check(self):

        # if (self.error_flag == 0) and (not self.myController.error) and (not self.joystick.stop):
        #     if np.any(np.abs(self.estimator.q_filt[7:, 0]) > self.q_security):
        #         self.myController.error = True
        #         self.error_flag = 1
        #         self.error_value = self.estimator.q_filt[7:, 0] * 180 / 3.1415
        #     if np.any(np.abs(self.estimator.v_secu) > 50):
        #         self.myController.error = True
        #         self.error_flag = 2
        #         self.error_value = self.estimator.v_secu
        #     if np.any(np.abs(self.myController.tau_ff) > 8):
        #         self.myController.error = True
        #         self.error_flag = 3
        #         self.error_value = self.myController.tau_ff

        # If something wrong happened in TSID controller we stick to a security controller
        if self.myController.error or self.joystick.stop:

            # Quantities sent to the control board
            self.result.P = np.zeros(12)
            self.result.D = 0.1 * np.ones(12)
            self.result.q_des[:] = np.zeros(12)
            self.result.v_des[:] = np.zeros(12)
            self.result.tau_ff[:] = np.zeros(12)

    def log_misc(self, tic, t_filter, t_planner, t_mpc, t_wbc):

        # Log joystick command
        if self.joystick is not None:
            self.estimator.v_ref = self.joystick.v_ref

        self.t_list_filter[self.k] = t_filter - tic
        self.t_list_planner[self.k] = t_planner - t_filter
        self.t_list_mpc[self.k] = t_mpc - t_planner
        self.t_list_wbc[self.k] = t_wbc - t_mpc
        self.t_list_loop[self.k] = time.time() - tic
        self.t_list_InvKin[self.k] = self.myController.tac - self.myController.tic
        self.t_list_QPWBC[self.k] = self.myController.toc - self.myController.tac
