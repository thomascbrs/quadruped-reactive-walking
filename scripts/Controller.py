# coding: utf8

import numpy as np
import utils_mpc
import time

from QP_WBC import wbc_controller
import MPC_Wrapper
import pybullet as pyb
import pinocchio as pin
from solopython.utils.viewerClient import viewerClient, NonBlockingViewerFromRobot
import libquadruped_reactive_walking as lqrw


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
        '''if self.enable_gepetto_viewer:
            self.view = viewerClient()'''

        # Enable/Disable perfect estimator
        perfectEstimator = False
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

        self.statePlanner = lqrw.StatePlanner()
        self.statePlanner.initialize(dt_mpc, T_mpc, self.h_ref)

        self.gait = lqrw.Gait()
        self.gait.initialize(dt_mpc, T_gait, T_mpc, N_gait)

        """from IPython import embed
        embed()"""

        shoulders = np.zeros((3, 4))
        shoulders[0, :] = [0.1946, 0.1946, -0.1946, -0.1946]
        shoulders[1, :] = [0.14695, -0.14695, 0.14695, -0.14695]
        self.footstepPlanner = lqrw.FootstepPlanner()
        self.footstepPlanner.initialize(dt_mpc, T_mpc, self.h_ref, shoulders.copy(), self.gait, N_gait)

        self.footTrajectoryGenerator = lqrw.FootTrajectoryGenerator()
        self.footTrajectoryGenerator.initialize(0.05, 0.07, self.fsteps_init.copy(), shoulders.copy(),
                                                dt_wbc, k_mpc, self.gait)

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
            if not self.gait.getIsStatic():
                self.q_estim[:, 0] = pin.integrate(self.solo.model,
                                                   self.q, self.v_estim * self.myController.dt)
                self.yaw_estim = (utils_mpc.quaternionToRPY(self.q_estim[3:7, 0]))[2, 0]
            else:
                self.planner.q_static[:] = pin.integrate(self.solo.model,
                                                         self.planner.q_static, self.v_estim * self.myController.dt)
                self.planner.RPY_static[:, 0:1] = utils_mpc.quaternionToRPY(self.planner.q_static[3:7, 0])
        else:
            self.yaw_estim = 0.0
            self.q_estim = self.q.copy()
            oMb = pin.SE3(pin.Quaternion(self.q[3:7, 0:1]), self.q[0:3, 0:1])
            self.v_estim = self.v.copy()

        # Update gait
        self.gait.updateGait(self.k, self.k_mpc, self.q[0:7, 0:1], self.joystick.joystick_code)

        # Update footsteps if new contact phase
        if(self.k % self.k_mpc == 0 and self.k != 0 and self.gait.isNewPhase()):
            self.footstepPlanner.updateNewContact()

        """// Get the reference velocity in world frame (given in base frame)
        Eigen::Quaterniond quat(q(6, 0), q(3, 0), q(4, 0), q(5, 0));  // w, x, y, z
        RPY << pinocchio::rpy::matrixToRpy(quat.toRotationMatrix());
        double c = std::cos(RPY(2, 0));
        double s = std::sin(RPY(2, 0));
        R_2.block(0, 0, 2, 2) << c, -s, s, c;
        R_2(2, 2) = 1.0;
        vref_in.block(0, 0, 3, 1) = R_2 * b_vref_in.block(0, 0, 3, 1);
        vref_in.block(3, 0, 3, 1) = b_vref_in.block(3, 0, 3, 1);"""

        o_v_ref = np.zeros((6, 1))
        o_v_ref[0:3, 0:1] = oMb.rotation @ self.joystick.v_ref[0:3, 0:1]
        o_v_ref[3:6, 0:1] = oMb.rotation @ self.joystick.v_ref[3:6, 0:1]

        # Compute target footstep based on current and reference velocities
        targetFootstep = self.footstepPlanner.computeTargetFootstep(self.q[0:7, 0:1], self.v[0:6, 0:1].copy(), o_v_ref)

        # Update pos, vel and acc references for feet
        # TODO: Make update take as parameters current gait, swing phase duration and remaining time
        self.footTrajectoryGenerator.update(self.k, targetFootstep)

        # Retrieve data from C++ planner
        # TODO
        """self.fsteps = self.planner.get_fsteps()
        self.gait = self.planner.get_gait()
        self.goals = self.planner.get_goals()
        self.vgoals = self.planner.get_vgoals()
        self.agoals = self.planner.get_agoals()"""

        # Run state planner (outputs the reference trajectory of the CoM / base)
        self.statePlanner.computeReferenceStates(self.q[0:7, 0:1], self.v[0:6, 0:1].copy(), o_v_ref, 0.0)
        # Result can be retrieved with self.statePlanner.getReferenceStates()
        xref = self.statePlanner.getReferenceStates()
        fsteps = self.footstepPlanner.getFootsteps()
        cgait = self.gait.getCurrentGait()

        t_planner = time.time()

        """if(self.k > 11000):

            from IPython import embed
            embed()"""

        # self.planner.setGait(self.planner.gait)

        # Process MPC once every k_mpc iterations of TSID
        if (self.k % self.k_mpc) == 0:
            try:
                self.mpc_wrapper.solve(self.k, xref, fsteps, cgait)
            except ValueError:
                print("MPC Problem")

        # Retrieve reference contact forces
        self.x_f_mpc = self.mpc_wrapper.get_latest_result()

        t_mpc = time.time()

        # Target state for the whole body control
        self.x_f_wbc = (self.x_f_mpc[:, 0]).copy()
        if not self.gait.getIsStatic():
            self.x_f_wbc[0] = self.q_estim[0, 0]
            self.x_f_wbc[1] = self.q_estim[1, 0]
            self.x_f_wbc[2] = self.h_ref
            self.x_f_wbc[3] = 0.0
            self.x_f_wbc[4] = 0.0
            self.x_f_wbc[5] = self.yaw_estim
        else:  # Sort of position control to avoid slow drift
            self.x_f_wbc[0:3] = self.planner.q_static[0:3, 0]
            self.x_f_wbc[3:6] = self.planner.RPY_static[:, 0]
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
            self.result.P = 3.0 * np.ones(12)
            self.result.D = 0.2 * np.ones(12)
            self.result.q_des[:] = self.myController.qdes[7:]
            self.result.v_des[:] = self.myController.vdes[6:, 0]
            self.result.tau_ff[:] = 0.5 * self.myController.tau_ff

            """if self.k % 5 == 0:
                self.solo.display(self.q)
                #self.view.display(self.q)
                #print("Pass")
                np.set_printoptions(linewidth=200, precision=2)
                print("###")
                #print(self.q.ravel())
                print(self.myController.tau_ff)"""

        t_wbc = time.time()

        # Security check
        self.security_check()

        # Update PyBullet camera
        self.pyb_camera(device)

        # Logs
        self.log_misc(t_start, t_filter, t_planner, t_mpc, t_wbc)

        # Increment loop counter
        self.k += 1

        return 0.0

    def pyb_camera(self, device):

        # Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        if self.k > 10 and self.enable_pyb_GUI:
            # pyb.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-30,
            #                                cameraTargetPosition=[1.0, 0.3, 0.25])
            pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                           cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

    def security_check(self):

        if (self.error_flag == 0) and (not self.myController.error) and (not self.joystick.stop):
            if np.any(np.abs(self.estimator.q_filt[7:, 0]) > self.q_security):
                self.myController.error = True
                self.error_flag = 1
                self.error_value = self.estimator.q_filt[7:, 0] * 180 / 3.1415
            if np.any(np.abs(self.estimator.v_secu) > 50):
                self.myController.error = True
                self.error_flag = 2
                self.error_value = self.estimator.v_secu
            if np.any(np.abs(self.myController.tau_ff) > 8):
                self.myController.error = True
                self.error_flag = 3
                self.error_value = self.myController.tau_ff

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
