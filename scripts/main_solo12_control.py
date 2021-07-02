# coding: utf8

import os
import sys

sys.path.insert(0, './solopython')


import threading
from Controller import Controller
import numpy as np
import argparse
from LoggerSensors import LoggerSensors
from LoggerControl import LoggerControl
import libquadruped_reactive_walking as lqrw

params = lqrw.Params()  # Object that holds all controller parameters

if params.SIMULATION:
    # from PyBulletSimulator import PyBulletSimulator
    from solo3D.PyBulletSimulator import PyBulletSimulator
else:
    # from pynput import keyboard
    from solopython.solo12 import Solo12
    from solopython.utils.qualisysClient import QualisysClient


key_pressed = False


def get_input():
    global key_pressed
    keystrk = input('Put the robot on the floor and press Enter \n')
    # thread doesn't continue until key is pressed
    key_pressed = True


def put_on_the_floor(device, q_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    global key_pressed
    key_pressed = False
    Kp_pos = 3.
    Kd_pos = 0.01
    imax = 3.0
    pos = np.zeros(device.nb_motors)
    for motor in range(device.nb_motors):
        pos[motor] = q_init[device.motorToUrdf[motor]] * device.gearRatioSigned[motor]
    i = threading.Thread(target=get_input)
    i.start()
    while not key_pressed:
        device.UpdateMeasurment()
        for motor in range(device.nb_motors):
            ref = Kp_pos*(pos[motor] - device.hardware.GetMotor(motor).GetPosition() -
                          Kd_pos*device.hardware.GetMotor(motor).GetVelocity())
            ref = min(imax, max(-imax, ref))
            device.hardware.GetMotor(motor).SetCurrentReference(ref)
        device.SendCommand(WaitEndOfCycle=True)

    print("Start the motion.")


def clone_movements(name_interface_clone, q_init, cloneP, cloneD, cloneQdes, cloneDQdes, cloneRunning, cloneResult):

    print("-- Launching clone interface --")

    print(name_interface_clone, params.dt_wbc)
    clone = Solo12(name_interface_clone, dt=params.dt_wbc)
    clone.Init(calibrateEncoders=True, q_init=q_init)

    while cloneRunning.value and not clone.hardware.IsTimeout():

        # print(cloneP[:], cloneD[:], cloneQdes[:], cloneDQdes[:], cloneRunning.value, cloneResult.value)
        if cloneResult.value:

            clone.SetDesiredJointPDgains(cloneP[:], cloneD[:])
            clone.SetDesiredJointPosition(cloneQdes[:])
            clone.SetDesiredJointVelocity(cloneDQdes[:])
            clone.SetDesiredJointTorque([0.0] * 12)

            clone.SendCommand(WaitEndOfCycle=True)

            cloneResult.value = False

    return 0


def control_loop(name_interface, name_interface_clone=None):
    """Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        name_interface (string): name of the interface that is used to communicate with the robot
        name_interface_clone (string): name of the interface that will mimic the movements of the first
    """

    # Check .yaml file for parameters of the controller

    # Enable or disable PyBullet GUI
    enable_pyb_GUI = params.enable_pyb_GUI
    if not params.SIMULATION:
        enable_pyb_GUI = False

    # Time variable to keep track of time
    t = 0.0

    # Default position after calibration
    q_init = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, -0.7, +1.4, -0.0, -0.7, +1.4])
    # q_init = np.array([0.0, 0.9, -1.6, -0.0, 0.9, -1.6, 0.0, 0.9, -1.6, -0.0, 0.9, -1.6])

    # q_init = np.array([0.0, 0.7, -1.4, -0.0, 0.7, -1.4, 0.0, 0.7, -1.4, -0.0, 0.7, -1.4])

    # Run a scenario and retrieve data thanks to the logger

    ####

    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = Solo12(name_interface, dt=params.dt_wbc)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    controller = Controller(q_init, params.envID, params.velID, params.dt_wbc, params.dt_mpc,
                            int(params.dt_mpc / params.dt_wbc), t, params.T_gait,
                            params.T_mpc, params.N_SIMULATION, params.type_MPC, params.use_flat_plane,
                            params.predefined_vel, enable_pyb_GUI, params.kf_enabled, params.N_gait,
                            params.SIMULATION, qc=qc)

    if name_interface_clone is not None:
        print("PASS")
        from multiprocessing import Process, Array, Value
        cloneP = Array('d', [0] * 12)
        cloneD = Array('d', [0] * 12)
        cloneQdes = Array('d', [0] * 12)
        cloneDQdes = Array('d', [0] * 12)
        cloneRunning = Value('b', True)
        cloneResult = Value('b', True)
        clone = Process(target=clone_movements, args=(name_interface_clone, q_init, cloneP,
                        cloneD, cloneQdes, cloneDQdes, cloneRunning, cloneResult))
        clone.start()
        print(cloneResult.value)

    if params.LOGGING or params.PLOTTING:
        loggerSensors = LoggerSensors(device, qualisys=qc, logSize=params.N_SIMULATION-3)
        loggerControl = LoggerControl(params.dt_wbc, params.N_gait, joystick=controller.joystick,
                                      estimator=controller.estimator, loop=controller,
                                      gait=controller.gait, statePlanner=controller.statePlanner,
                                      footstepPlanner=controller.footstepPlanner,
                                      footTrajectoryGenerator=controller.footTrajectoryGenerator,
                                      logSize=params.N_SIMULATION-3)

    # Number of motors
    nb_motors = device.nb_motors

    # Initiate communication with the device and calibrate encoders
    if params.SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=params.envID,
                    use_flat_plane=params.use_flat_plane, enable_pyb_GUI=enable_pyb_GUI, dt=params.dt_wbc)
    else:
        device.Init(calibrateEncoders=True, q_init=q_init)

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (params.N_SIMULATION-2) * params.dt_wbc

    while ((not device.hardware.IsTimeout()) and (t < t_max) and (not controller.myController.error)):

        # Update sensor data (IMU, encoders, Motion capture)
        device.UpdateMeasurment()

        # Desired torques
        controller.compute(device)

        # Check that the initial position of actuators is not too far from the
        # desired position of actuators to avoid breaking the robot
        if (t <= 10 * params.dt_wbc):
            if np.max(np.abs(controller.result.q_des - device.q_mes)) > 0.15:
                print("DIFFERENCE: ", controller.result.q_des - device.q_mes)
                print("q_des: ", controller.result.q_des)
                print("q_mes: ", device.q_mes)
                break

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(controller.result.P, controller.result.D)
        device.SetDesiredJointPosition(controller.result.q_des)
        device.SetDesiredJointVelocity(controller.result.v_des)
        device.SetDesiredJointTorque(controller.result.tau_ff.ravel())

        # Call logger
        if params.LOGGING or params.PLOTTING:
            loggerSensors.sample(device, qc)
            loggerControl.sample(controller.joystick, controller.estimator,
                                 controller, controller.gait, controller.statePlanner,
                                 controller.footstepPlanner, controller.footTrajectoryGenerator,
                                 controller.myController)

        # Send command to the robot
        for i in range(1):
            device.SendCommand(WaitEndOfCycle=True)
        """if ((device.cpt % 100) == 0):
            device.Print()"""

        # import os
        # from matplotlib import pyplot as plt
        # import pybullet as pyb
        # newpath = r'/home/thomas_cbrs/Desktop/edin/recording/images/'
        # if (t == 0.0):
        #     cpt_frames = 0
        #     step = 5
        # if (cpt_frames % step) == 0:
        #     if (cpt_frames % 1000):
        #         print(cpt_frames)
        #     img = pyb.getCameraImage(width=1920, height=1080, renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        #     if cpt_frames == 0:
                
        #         if not os.path.exists(newpath):
        #             os.makedirs(newpath)
        #     if (int(cpt_frames/step) < 10):
        #         plt.imsave(newpath + 'frame_000'+str(int(cpt_frames/step))+'.png', img[2])
        #     elif int(cpt_frames/step) < 100:
        #         plt.imsave(newpath + 'frame_00'+str(int(cpt_frames/step))+'.png', img[2])
        #     elif int(cpt_frames/step) < 1000:
        #         plt.imsave(newpath + 'frame_0'+str(int(cpt_frames/step))+'.png', img[2])
        #     else:
        #         plt.imsave(newpath + 'frame_'+str(int(cpt_frames/step))+'.png', img[2])

        # cpt_frames += 1

        t += params.dt_wbc  # Increment loop time

    # ***************************************************************
    print(not device.hardware.IsTimeout())
    print(t < t_max)
    print(not controller.myController.error)

    # Stop clone interface running in parallel process
    if not params.SIMULATION and name_interface_clone is not None:
        cloneResult.value = False

    # Stop MPC running in a parallel process
    if controller.enable_multiprocessing:
        print("Stopping parallel process")
        controller.mpc_wrapper.stop_parallel_loop()
        # controller.surfacePlanner.stop_parallel_loop()
    # controller.view.stop()  # Stop viewer

    # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
    t = 0.0
    t_max = 2.5
    while ((not device.hardware.IsTimeout()) and (t < t_max)):

        device.UpdateMeasurment()  # Retrieve data from IMU and Motion capture

        # Set desired quantities for the actuators
        device.SetDesiredJointPDgains(np.zeros(12), 0.1 * np.ones(12))
        device.SetDesiredJointPosition(np.zeros(12))
        device.SetDesiredJointVelocity(np.zeros(12))
        device.SetDesiredJointTorque(np.zeros(12))

        # Send command to the robot
        device.SendCommand(WaitEndOfCycle=True)
        if ((device.cpt % 1000) == 0):
            device.Print()

        t += params.dt_wbc

    # FINAL SHUTDOWN *************************************************

    # Whatever happened we send 0 torques to the motors.
    device.SetDesiredJointTorque([0]*nb_motors)
    device.SendCommand(WaitEndOfCycle=True)

    if device.hardware.IsTimeout():
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")
    device.hardware.Stop()  # Shut down the interface between the computer and the master board

    # Plot estimated computation time for each step for the control architecture
    """from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(controller.t_list_filter[1:], 'r+')
    plt.plot(controller.t_list_gait[1:], 'y+')
    plt.plot(controller.t_list_footstep[1:], 'r+')
    plt.plot(controller.t_list_state[1:], 'g+')
    plt.plot(controller.t_list_foottraj[1:], 'b+')
    plt.plot(controller.t_list_mpc[1:], 'b+')
    plt.plot(controller.t_list_wbc[1:], '+', color="violet")
    plt.plot(controller.t_list_loop[1:], 'k+')
    plt.plot(controller.t_list_InvKin[1:], 'o', color="darkgreen")
    plt.plot(controller.t_list_QPWBC[1:], 'o', color="royalblue")
    plt.legend(["Estimator", "gait", "footstep planner", "state planner", "trajectory planner", "MPC", "WBC", "Whole loop", "InvKin", "QP WBC"])
    plt.title("Loop time [s]")
    plt.show(block=True)"""

    # Plot recorded data
    if params.PLOTTING:
        loggerControl.plotAll(loggerSensors)

    # Save the logs of the Logger object
    if params.LOGGING:
        loggerControl.saveAll(loggerSensors)
        print("Log saved")

    if params.SIMULATION and enable_pyb_GUI:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

    if controller.myController.error:
        if (controller.error_flag == 1):
            print("-- POSITION LIMIT ERROR --")
        elif (controller.error_flag == 2):
            print("-- VELOCITY TOO HIGH ERROR --")
        elif (controller.error_flag == 3):
            print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
        print(controller.error_value)

    # Print timing from cprofile
    # output_folder = "/home/frisbourg/Bureau/"
    # controller.footstepPlanner.print_profile(output_folder + "footStepPlanner.prof")
    # controller.footTrajectoryGenerator.print_profile(output_folder + "bezier.prof")
    # controller.surfacePlanner.surfacePlanner.print_profile(output_folder + "surfacePlanner.prof")

    # Plot figure relative to planner values
    controller.loggerPlanner.plot_log_planner()

    print("End of script")
    quit()


def main():
    """Main function
    """

    parser = argparse.ArgumentParser(description='Playback trajectory to show the extent of solo12 workspace.')
    parser.add_argument('-i',
                        '--interface',
                        required=True,
                        help='Name of the interface (use ifconfig in a terminal), for instance "enp1s0"')
    parser.add_argument('-c',
                        '--clone',
                        required=False,
                        help='Name of the clone interface that will reproduce the movement of the first one \
                              (use ifconfig in a terminal), for instance "enp1s0"')

    control_loop(parser.parse_args().interface, parser.parse_args().clone)


if __name__ == "__main__":
    main()
