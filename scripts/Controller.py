from cmath import nan
import numpy as np
import utils_mpc
import time

import MPC_Wrapper
import pybullet as pyb
import pinocchio as pin
import libquadruped_reactive_walking as lqrw

from solo3D.tools.utils import quaternionToRPY


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


class dummyIMU:
    """Fake IMU for initialisation purpose"""

    def __init__(self):

        self.linear_acceleration = np.zeros(3)
        self.gyroscope = np.zeros(3)
        self.attitude_euler = np.zeros(3)
        self.attitude_quaternion = np.zeros(4)


class dummyJoints:
    """Fake joints for initialisation purpose"""

    def __init__(self):

        self.positions = np.zeros(12)
        self.velocities = np.zeros(12)


class dummyDevice:
    """Fake device for initialisation purpose"""

    def __init__(self):

        self.hardware = dummyHardware()
        self.imu = dummyIMU()
        self.joints = dummyJoints()
        self.dummyPos = np.zeros(3)
        self.dummyPos[2] = 0.1944
        self.b_baseVel = np.zeros(3)


class Controller:

    def __init__(self, params, q_init, t):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """

        ########################################################################
        #                        Parameters definition                         #
        ########################################################################

        # Init joint torques to correct shape
        self.jointTorques = np.zeros((12, 1))

        # List to store the IDs of debug lines
        self.ID_deb_lines = []

        # Disable perfect estimator if we are not in simulation
        if not params.SIMULATION:
            params.perfectEstimator = False  # Cannot use perfect estimator if we are running on real robot

        # Initialisation of the solo model/data and of the Gepetto viewer
        self.solo = utils_mpc.init_robot(q_init, params)

        # Create Joystick object
        self.joystick = lqrw.Joystick()
        self.joystick.initialize(params)

        # Enable/Disable hybrid control
        self.enable_hybrid_control = True

        self.h_ref = params.h_ref
        self.h_ref_mem = params.h_ref
        self.q = np.zeros((18, 1))  # Orientation part is in roll pitch yaw
        self.q[:6, 0] = np.array([0.0, 0.0, self.h_ref, 0.0, 0.0, 0.0])
        self.q[6:, 0] = q_init
        self.q_init = q_init.copy()
        self.v = np.zeros((18, 1))
        self.b_v = np.zeros((18, 1))
        self.o_v_filt = np.zeros((18, 1))

        self.q_wbc = np.zeros((18, 1))
        self.dq_wbc = np.zeros((18, 1))
        self.xgoals = np.zeros((12, 1))
        self.xgoals[2, 0] = self.h_ref

        self.gait = lqrw.Gait()
        self.gait.initialize(params)

        self.estimator = lqrw.Estimator()
        self.estimator.initialize(params)

        self.wbcWrapper = lqrw.WbcWrapper()
        self.wbcWrapper.initialize(params)

        # Wrapper that makes the link with the solver that you want to use for the MPC
        self.mpc_wrapper = MPC_Wrapper.MPC_Wrapper(params, self.q)
        self.o_targetFootstep = np.zeros((3, 4))  # Store result for MPC_planner

        self.DEMONSTRATION = params.DEMONSTRATION
        self.solo3D = params.solo3D
        self.SIMULATION = params.SIMULATION
        if params.solo3D:
            from solo3D.SurfacePlannerWrapper import SurfacePlanner_Wrapper
            if self.SIMULATION:
                from solo3D.tools.pyb_environment_3D import PybEnvironment3D

        self.enable_multiprocessing_mip = params.enable_multiprocessing_mip
        self.offset_perfect_estimator = 0.
        if self.solo3D:
            self.surfacePlanner = SurfacePlanner_Wrapper(params)  # MIP Wrapper

            self.statePlanner = lqrw.StatePlanner3D()
            self.statePlanner.initialize(params)

            self.footstepPlanner = lqrw.FootstepPlannerQP()
            self.footstepPlanner.initialize(params, self.gait, self.surfacePlanner.floor_surface)

            # Trajectory Generator Bezier
            x_margin_max_ = 0.06  # 4cm margin
            t_margin_ = 0.28  # 15% of the curve around critical point
            z_margin_ = 0.06  # 1% of the curve after the critical point
            N_sample = 8  # Number of sample in the least square optimisation for Bezier coeffs
            N_sample_ineq = 10  # Number of sample while browsing the curve
            degree = 7  # Degree of the Bezier curve

            self.footTrajectoryGenerator = lqrw.FootTrajectoryGeneratorBezier()
            self.footTrajectoryGenerator.initialize(params, self.gait, self.surfacePlanner.floor_surface,
                                                    x_margin_max_, t_margin_, z_margin_, N_sample, N_sample_ineq,
                                                    degree)
            if self.SIMULATION:
                self.pybEnvironment3D = PybEnvironment3D(params, self.gait, self.statePlanner, self.footstepPlanner,
                                                         self.footTrajectoryGenerator)
        else:
            self.statePlanner = lqrw.StatePlanner()
            self.statePlanner.initialize(params)

            self.footstepPlanner = lqrw.FootstepPlanner()
            self.footstepPlanner.initialize(params, self.gait)

            self.footTrajectoryGenerator = lqrw.FootTrajectoryGenerator()
            self.footTrajectoryGenerator.initialize(params, self.gait)

        self.envID = params.envID
        self.velID = params.velID
        self.dt_wbc = params.dt_wbc
        self.dt_mpc = params.dt_mpc
        self.k_mpc = int(params.dt_mpc / params.dt_wbc)
        self.t = t
        self.N_SIMULATION = params.N_SIMULATION
        self.type_MPC = params.type_MPC
        self.use_flat_plane = params.use_flat_plane
        self.predefined_vel = params.predefined_vel
        self.enable_pyb_GUI = params.enable_pyb_GUI
        self.enable_corba_viewer = params.enable_corba_viewer
        self.Kp_main = params.Kp_main
        self.Kd_main = params.Kd_main
        self.Kff_main = params.Kff_main

        self.k = 0

        self.qmes12 = np.zeros((19, 1))
        self.vmes12 = np.zeros((18, 1))

        self.q_display = np.zeros((19, 1))
        self.v_ref = np.zeros((18, 1))
        self.a_ref = np.zeros((18, 1))
        self.h_v = np.zeros((18, 1))
        self.h_v_windowed = np.zeros((6, 1))
        self.yaw_estim = 0.0
        self.RPY_filt = np.zeros(3)

        self.feet_a_cmd = np.zeros((3, 4))
        self.feet_v_cmd = np.zeros((3, 4))
        self.feet_p_cmd = np.zeros((3, 4))

        self.error = False  # True if something wrong happens in the controller

        self.q_filter = np.zeros((18, 1))
        self.h_v_filt_mpc = np.zeros((6, 1))
        self.vref_filt_mpc = np.zeros((6, 1))
        self.filter_mpc_q = lqrw.Filter()
        self.filter_mpc_q.initialize(params)
        self.filter_mpc_v = lqrw.Filter()
        self.filter_mpc_v.initialize(params)
        self.filter_mpc_vref = lqrw.Filter()
        self.filter_mpc_vref.initialize(params)

        self.nle = np.zeros((6, 1))

        self.p_ref = np.zeros((6, 1))
        self.treshold_static = False

        # Interface with the PD+ on the control board
        self.result = Result()

        # Run the control loop once with a dummy device for initialization
        dDevice = dummyDevice()
        dDevice.joints.positions = q_init
        self.compute(dDevice)

    def compute(self, device, qc=None):
        """Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """
        t_start = time.time()

        self.joystick.update_v_ref(self.k, self.velID, self.gait.getIsStatic())

        q_perfect = np.zeros(6)
        b_baseVel_perfect = np.zeros(3)
        if self.solo3D and qc == None:
            q_perfect[:3] = device.dummyPos
            q_perfect[3:] = device.imu.attitude_euler  # Yaw only used for solo3D
            b_baseVel_perfect = device.b_baseVel
        elif self.solo3D and qc != None:
            if self.k <= 1:
                # self.initial_pos = qc.getPosition()
                # self.initial_pos[2] = self.initial_pos[2] - 0.22
                self.initial_pos = [0.419, 0.009, -0.047]
                # self.initial_rot = quaternionToRPY(qc.getOrientationQuat())
                # self.initial_matrix = pin.rpy.rpyToMatrix(0., 0., self.initial_rot[2, 0]).transpose()
                self.initial_matrix = pin.rpy.rpyToMatrix(0., 0., 0.).transpose()

            q_perfect[:3] = self.initial_matrix @ (qc.getPosition() - self.initial_pos)
            q_perfect[3:] = quaternionToRPY(qc.getOrientationQuat())
            b_baseVel_perfect[:] = (qc.getOrientationMat9().reshape((3, 3)).transpose() @ qc.getVelocity().reshape((3, 1))).ravel()

        if np.isnan(np.sum(q_perfect)):
            print("Error: nan values in perfect position of the robot")
            q_perfect = self.q[6:]
            self.error = True
        elif np.isnan(np.sum(b_baseVel_perfect)):
            print("Error: nan values in perfect velocity of the robot")
            b_baseVel_perfect = self.v[6:]
            self.error = True

        self.estimator.run_filter(self.gait.getCurrentGait(),
                                  self.footTrajectoryGenerator.getFootPosition(),
                                  device.imu.linear_acceleration,
                                  device.imu.gyroscope,
                                  device.imu.attitude_euler,
                                  device.joints.positions,
                                  device.joints.velocities,
                                  q_perfect, b_baseVel_perfect)

        # Update state vectors of the robot (q and v) + transformation matrices between world and horizontal frames
        self.estimator.updateState(self.joystick.getVRef(), self.gait)
        oRh = self.estimator.getoRh()
        hRb = self.estimator.gethRb()
        oTh = self.estimator.getoTh().reshape((3, 1))
        self.a_ref[:6, 0] = self.estimator.getARef()
        self.v_ref[:6, 0] = self.estimator.getVRef()
        self.h_v[:6, 0] = self.estimator.getHV()
        self.h_v_windowed[:6, 0] = self.estimator.getHVWindowed()
        if self.solo3D:
            self.q[:3, 0] = self.estimator.getQFilt()[:3]
            self.q[6:, 0] = self.estimator.getQFilt()[7:]
            self.q[3:6] = quaternionToRPY(self.estimator.getQFilt()[3:7])
        else:
            self.q[:, 0] = self.estimator.getQUpdated()
        self.v[:, 0] = self.estimator.getVUpdated()
        self.yaw_estim = self.estimator.getYawEstim()

        # Quantities go through a 1st order low pass filter with fc = 15 Hz (avoid >25Hz foldback)
        self.q_filter[:6, 0] = self.filter_mpc_q.filter(self.q[:6, 0], True)
        self.q_filter[6:, 0] = self.q[6:, 0].copy()
        self.h_v_filt_mpc[:, 0] = self.filter_mpc_v.filter(self.h_v[:6, 0], False)
        self.vref_filt_mpc[:, 0] = self.filter_mpc_vref.filter(self.v_ref[:6, 0], False)

        if self.solo3D:
            oTh_3d = np.zeros((3, 1))
            oTh_3d[:2, 0] = self.q_filter[:2, 0]
            oRh_3d = pin.rpy.rpyToMatrix(0., 0., self.q_filter[5, 0])

        t_filter = time.time()

        self.gait.updateGait(self.k, self.k_mpc, self.joystick.getJoystickCode())

        update_mip = self.k % self.k_mpc == 0 and self.gait.isNewPhase()
        if self.solo3D:
            if update_mip:
                self.statePlanner.updateSurface(self.q_filter[:6, :1], self.vref_filt_mpc[:6, :1])
                if self.surfacePlanner.initialized:
                    self.error = self.surfacePlanner.get_latest_results()
                    if not self.enable_multiprocessing_mip and self.SIMULATION:
                        self.pybEnvironment3D.update_target_SL1M(self.surfacePlanner.all_feet_pos)

            self.footstepPlanner.updateSurfaces(self.surfacePlanner.potential_surfaces, self.surfacePlanner.selected_surfaces,
                                                self.surfacePlanner.mip_success, self.surfacePlanner.mip_iteration)

        self.o_targetFootstep = self.footstepPlanner.updateFootsteps(self.k % self.k_mpc == 0 and self.k != 0,
                                                                     int(self.k_mpc - self.k % self.k_mpc),
                                                                     self.q_filter[:, 0], self.h_v_windowed[:6, :1].copy(),
                                                                     self.v_ref[:6, :1])

        self.statePlanner.computeReferenceStates(self.q_filter[:6, :1], self.h_v_filt_mpc[:6, :1].copy(), self.vref_filt_mpc[:6, :1])

        xref = self.statePlanner.getReferenceStates()
        fsteps = self.footstepPlanner.getFootsteps()
        cgait = self.gait.getCurrentGait()

        if update_mip and self.solo3D:
            configs = self.statePlanner.getConfigurations().transpose()
            self.surfacePlanner.run(configs, cgait, self.o_targetFootstep, self.vref_filt_mpc[:3, 0].copy())
            self.surfacePlanner.initialized = True

        t_planner = time.time()

        # Solve MPC
        if (self.k % self.k_mpc) == 0:
            try:
                self.mpc_wrapper.solve(self.k, xref, fsteps, cgait, np.zeros((3, 4)))
            except ValueError:
                print("MPC Problem")
        self.x_f_mpc = self.mpc_wrapper.get_latest_result()

        t_mpc = time.time()

        # Update pos, vel and acc references for feet
        if self.solo3D:
            self.footTrajectoryGenerator.update(self.k, self.o_targetFootstep, self.surfacePlanner.selected_surfaces, self.q_filter)
        else:
            self.footTrajectoryGenerator.update(self.k, self.o_targetFootstep)

        if not self.error and not self.joystick.getStop():
            if self.DEMONSTRATION and self.gait.getIsStatic():
                hRb = np.eye(3)

            # Desired position, orientation and velocities of the base
            self.xgoals[:6, 0] = np.zeros((6,))
            if self.DEMONSTRATION and self.joystick.getL1() and self.gait.getIsStatic():
                self.p_ref[:, 0] = self.joystick.getPRef()
                self.xgoals[[3, 4], 0] = self.p_ref[[3, 4], 0]
                self.h_ref = self.p_ref[2, 0]
                hRb = pin.rpy.rpyToMatrix(0.0, 0.0, self.p_ref[5, 0])
            else:
                self.xgoals[[3, 4], 0] = xref[[3, 4], 1]
                self.h_ref = self.h_ref_mem

            # If the four feet are in contact then we do not listen to MPC (default contact forces instead)
            if self.DEMONSTRATION and self.gait.getIsStatic():
                self.x_f_mpc[12:24, 0] = [0.0, 0.0, 9.81 * 2.5 / 4.0] * 4

            # Update configuration vector for wbc with filtered roll and pitch and reference angular positions of previous loop
            self.q_wbc[3:5, 0] = self.q_filter[3:5, 0]
            self.q_wbc[6:, 0] = self.wbcWrapper.qdes[:]

            # Update velocity vector for wbc
            self.dq_wbc[:6, 0] = self.estimator.getVFilt()[:6]  #  Velocities in base frame (not horizontal frame!)
            self.dq_wbc[6:, 0] = self.wbcWrapper.vdes[:]  # with reference angular velocities of previous loop

            # Feet command position, velocity and acceleration in base frame
            if self.solo3D:  # Use estimated base frame
                self.feet_a_cmd = self.footTrajectoryGenerator.getFootAccelerationBaseFrame(
                    oRh_3d.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
                self.feet_v_cmd = self.footTrajectoryGenerator.getFootVelocityBaseFrame(
                    oRh_3d.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
                self.feet_p_cmd = self.footTrajectoryGenerator.getFootPositionBaseFrame(
                    oRh_3d.transpose(), oTh_3d + np.array([[0.0], [0.0], [xref[2, 1]]]))
            else:  # Use ideal base frame
                self.feet_a_cmd = self.footTrajectoryGenerator.getFootAccelerationBaseFrame(
                    hRb @ oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
                self.feet_v_cmd = self.footTrajectoryGenerator.getFootVelocityBaseFrame(
                    hRb @ oRh.transpose(), np.zeros((3, 1)), np.zeros((3, 1)))
                self.feet_p_cmd = self.footTrajectoryGenerator.getFootPositionBaseFrame(
                    hRb @ oRh.transpose(), oTh + np.array([[0.0], [0.0], [self.h_ref]]))

            self.xgoals[6:, 0] = self.vref_filt_mpc[:, 0]  # Velocities (in horizontal frame!)

            # Run InvKin + WBC QP
            self.wbcWrapper.compute(self.q_wbc, self.dq_wbc,
                                    (self.x_f_mpc[12:24, 0:1]).copy(), np.array([cgait[0, :]]),
                                    self.feet_p_cmd,
                                    self.feet_v_cmd,
                                    self.feet_a_cmd,
                                    self.xgoals)
            # Quantities sent to the control board
            self.result.P = np.array(self.Kp_main.tolist() * 4)
            self.result.D = np.array(self.Kd_main.tolist() * 4)
            self.result.FF = self.Kff_main * np.ones(12)
            self.result.q_des[:] = self.wbcWrapper.qdes[:]
            self.result.v_des[:] = self.wbcWrapper.vdes[:]
            self.result.tau_ff[:] = self.wbcWrapper.tau_ff

            self.clamp_result(device)

            self.nle[:3, 0] = self.wbcWrapper.nle[:3]

            # Display robot in Gepetto corba viewer
            if self.enable_corba_viewer and (self.k % 5 == 0):
                self.q_display[:3, 0] = self.q_wbc[:3, 0]
                self.q_display[3:7, 0] = pin.Quaternion(pin.rpy.rpyToMatrix(self.q_wbc[3:6, 0])).coeffs()
                self.q_display[7:, 0] = self.q_wbc[6:, 0]
                self.solo.display(self.q_display)

        t_wbc = time.time()

        self.security_check()
        if self.error or self.joystick.getStop():
            self.set_null_control()

        # Update PyBullet camera
        if not self.solo3D:
            self.pyb_camera(device, 0.0)
        else:  # Update 3D Environment
            if self.SIMULATION:
                self.pybEnvironment3D.update(self.k)

        # Update debug display (spheres, ...)
        self.pyb_debug(device, fsteps, cgait, xref)

        # Logs
        self.log_misc(t_start, t_filter, t_planner, t_mpc, t_wbc)

        self.k += 1

    def pyb_camera(self, device, yaw):
        """
           Update position of PyBullet camera on the robot position to do as if it was attached to the robot
        """
        if self.k > 10 and self.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                           cameraTargetPosition=[device.dummyHeight[0], device.dummyHeight[1], 0.0])

    def pyb_debug(self, device, fsteps, cgait, xref):

        if self.k > 1 and self.enable_pyb_GUI:

            # Display desired feet positions in WBC as green spheres
            oTh_pyb = device.dummyPos.reshape((-1, 1))
            oTh_pyb[2, 0] += 0.0
            oRh_pyb = pin.rpy.rpyToMatrix(0.0, 0.0, device.imu.attitude_euler[2])
            for i in range(4):
                if not self.solo3D:
                    pos = oRh_pyb @ self.feet_p_cmd[:, i:(i+1)] + oTh_pyb
                    pyb.resetBasePositionAndOrientation(device.pyb_sim.ftps_Ids_deb[i], pos[:, 0].tolist(), [0, 0, 0, 1])
                else:
                    pos = self.o_targetFootstep[:, i]
                    pyb.resetBasePositionAndOrientation(device.pyb_sim.ftps_Ids_deb[i], pos, [0, 0, 0, 1])

            # Display desired footstep positions as blue spheres
            for i in range(4):
                j = 0
                cpt = 1
                status = cgait[0, i]
                while cpt < cgait.shape[0] and j < device.pyb_sim.ftps_Ids.shape[1]:
                    while cpt < cgait.shape[0] and cgait[cpt, i] == status:
                        cpt += 1
                    if cpt < cgait.shape[0]:
                        status = cgait[cpt, i]
                        if status:
                            pos = oRh_pyb @ fsteps[cpt, (3*i):(3*(i+1))].reshape(
                                (-1, 1)) + oTh_pyb - np.array([[0.0], [0.0], [self.h_ref]])
                            pyb.resetBasePositionAndOrientation(
                                device.pyb_sim.ftps_Ids[i, j], pos[:, 0].tolist(), [0, 0, 0, 1])
                        else:
                            pyb.resetBasePositionAndOrientation(device.pyb_sim.ftps_Ids[i, j], [
                                                                0.0, 0.0, -0.1], [0, 0, 0, 1])
                        j += 1

                # Hide unused spheres underground
                for k in range(j, device.pyb_sim.ftps_Ids.shape[1]):
                    pyb.resetBasePositionAndOrientation(device.pyb_sim.ftps_Ids[i, k], [0.0, 0.0, -0.1], [0, 0, 0, 1])

            # Display reference trajectory
            xref_rot = np.zeros((3, xref.shape[1]))
            for i in range(xref.shape[1]):
                xref_rot[:, i:(i+1)] = oRh_pyb @ xref[:3, i:(i+1)] + oTh_pyb + np.array([[0.0], [0.0], [0.05 - self.h_ref]])

            if len(device.pyb_sim.lineId_red) == 0:
                for i in range(xref.shape[1]-1):
                    device.pyb_sim.lineId_red.append(pyb.addUserDebugLine(
                        xref_rot[:3, i].tolist(), xref_rot[:3, i+1].tolist(), lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8))
            else:
                for i in range(xref.shape[1]-1):
                    device.pyb_sim.lineId_red[i] = pyb.addUserDebugLine(xref_rot[:3, i].tolist(), xref_rot[:3, i+1].tolist(),
                                                                        lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8,
                                                                        replaceItemUniqueId=device.pyb_sim.lineId_red[i])

            # Display predicted trajectory
            x_f_mpc_rot = np.zeros((3, self.x_f_mpc.shape[1]))
            for i in range(self.x_f_mpc.shape[1]):
                x_f_mpc_rot[:, i:(i+1)] = oRh_pyb @ self.x_f_mpc[:3, i:(i+1)] + oTh_pyb + np.array([[0.0], [0.0], [0.05 - self.h_ref]])

            if len(device.pyb_sim.lineId_blue) == 0:
                for i in range(self.x_f_mpc.shape[1]-1):
                    device.pyb_sim.lineId_blue.append(pyb.addUserDebugLine(
                        x_f_mpc_rot[:3, i].tolist(), x_f_mpc_rot[:3, i+1].tolist(), lineColorRGB=[0.0, 0.0, 1.0], lineWidth=8))
            else:
                for i in range(self.x_f_mpc.shape[1]-1):
                    device.pyb_sim.lineId_blue[i] = pyb.addUserDebugLine(x_f_mpc_rot[:3, i].tolist(), x_f_mpc_rot[:3, i+1].tolist(),
                                                                         lineColorRGB=[0.0, 0.0, 1.0], lineWidth=8,
                                                                         replaceItemUniqueId=device.pyb_sim.lineId_blue[i])

    def security_check(self):
        """
        Check if the command is fine and set the command to zero in case of error
        """
        if not (self.error or self.joystick.getStop()):
            error_flag = self.estimator.security_check(self.wbcWrapper.tau_ff)
            if (error_flag != 0):
                self.error = True
                if (error_flag == 1):
                    print("-- POSITION LIMIT ERROR --")
                    print(self.estimator.getQFilt()[7:])
                elif (error_flag == 2):
                    print("-- VELOCITY TOO HIGH ERROR --")
                    print(self.estimator.getVSecu())
                else:
                    print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
                    print(self.wbcWrapper.tau_ff)

    def clamp(self, num, min_value=None, max_value=None):
        clamped = False
        if min_value is not None and num <= min_value:
            num = min_value
            clamped = True
        if max_value is not None and num >= max_value:
            num = max_value
            clamped = True
        return clamped

    def clamp_result(self, device, set_error=False):
        """
        Clamp the result
        """
        hip_max = 120. * np.pi / 180.
        knee_min = 5. * np.pi / 180.
        for i in range(4):
            if self.clamp(self.result.q_des[3 * i + 1], -hip_max, hip_max):
                print("Clamping hip n " + str(i))
                self.error = False
            if self.q_init[3 * i + 2] >= 0 and self.clamp(self.result.q_des[3 * i + 2], knee_min):
                print("Clamping knee n " + str(i))
                self.error = False
            elif self.clamp(self.result.q_des[3 * i + 2], max_value=-knee_min):
                print("Clamping knee n " + str(i))
                self.error = False

        for i in range(12):
            if self.clamp(self.result.q_des[i], device.joints.positions[i] - 1., device.joints.positions[i] + 1.):
                print("Clamping position difference of motor n " + str(i))
                self.error = False

            if self.clamp(self.result.v_des[i], device.joints.velocities[i] - 80., device.joints.velocities[i] + 80.):
                print("Clamping velocity of motor n " + str(i))
                self.error = False

            if self.clamp(self.result.tau_ff[i], -8., 8.):
                print("Clamping torque of motor n " + str(i))
                self.error = False

    def set_null_control(self):
        """
        Send default null values to the robot
        """
        self.result.P = np.zeros(12)
        self.result.D = 0.1 * np.ones(12)
        self.result.q_des[:] = np.zeros(12)
        self.result.v_des[:] = np.zeros(12)
        self.result.FF = np.zeros(12)
        self.result.tau_ff[:] = np.zeros(12)

    def log_misc(self, tic, t_filter, t_planner, t_mpc, t_wbc):
        self.t_filter = t_filter - tic
        self.t_planner = t_planner - t_filter
        self.t_mpc = t_mpc - t_planner
        self.t_wbc = t_wbc - t_mpc
        self.t_loop = time.time() - tic
