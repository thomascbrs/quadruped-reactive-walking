# coding: utf8

import numpy as np
import libquadruped_reactive_walking as MPC
from multiprocessing import Process, Value, Array
from utils_mpc import quaternionToRPY
import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl
import crocoddyl_class.MPC_crocoddyl_planner as MPC_crocoddyl_planner
import crocoddyl_class.MPC_crocoddyl_planner_time as MPC_crocoddyl_planner_time


class Dummy:
    """Dummy class to store variables"""

    def __init__(self):
        self.xref = None  # Desired trajectory
        self.footsteps = None  # Desired location of footsteps
        pass


class MPC_Wrapper:
    """Wrapper to run both types of MPC (OQSP or Crocoddyl) with the possibility to run OSQP in
    a parallel process

    Args:
        mpc_type (int): 0 for OSQP MPC, 1, 2, 3 for Crocoddyl MPCs
        dt (float): Time step of the MPC
        n_nodes (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
        q_init (array): the default position of the robot
        multiprocessing (bool): Enable/Disable running the MPC with another process
    """

    def __init__(self, params, q_init):
        self.initialized = False
        self.params = params

        self.k_mpc = int(params.dt_mpc/params.dt_wbc)  # Number of WBC steps for 1 step of the MPC

        self.dt = params.dt_mpc
        self.n_nodes = np.int(params.T_mpc/params.dt_mpc)
        self.T_gait = params.T_gait
        self.N_gait = params.N_gait
        self.gait_memory = np.zeros(4)

        self.mpc_type = params.type_MPC

        self.multiprocessing = params.enable_multiprocessing
        if self.multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', False)
            if self.mpc_type == 3:  # Need more space to store optimized footsteps and l_footsteps to stop the optimization around it
                self.dataIn = Array('d', [0.0] * (1 + (self.n_nodes + 1) * 12 + 12 * self.N_gait + 12))
                self.dataOut = Array('d', [0] * 32 * self.n_nodes)
            elif self.mpc_type == 4:  # Need more space to store optimized footsteps and l_footsteps to stop the optimization around it
                self.dataIn = Array('d', [0.0] * (1 + (self.n_nodes + 1) * 12 + 12 * self.N_gait + 12))
                self.dataOut = Array('d', [0] * 33 * self.n_nodes)
            else:
                self.dataIn = Array('d', [0.0] * (1 + (self.n_nodes + 1) * 12 + 12 * self.N_gait))
                self.dataOut = Array('d', [0] * 24 * self.n_nodes)
            self.footsteps_future = np.zeros((self.N_gait, 12))
            self.running = Value('b', True)
        else:
            if self.mpc_type == 0:  # OSQP MPC
                self.mpc = MPC.MPC(params)
            elif self.mpc_type == 1:  # Crocoddyl MPC Linear
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, linearModel=True)
            elif self.mpc_type == 2:  # Crocoddyl MPC Non-Linear
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, linearModel=False)
            elif self.mpc_type == 3:  # Crocoddyl MPC Non-Linear with footsteps optimization
                self.mpc = MPC_crocoddyl_planner.MPC_crocoddyl_planner(params, mu=0.9)
            elif self.mpc_type == 4:  # Crocoddyl MPC Non-Linear with footsteps and time optimization
                self.mpc = MPC_crocoddyl_planner_time.MPC_crocoddyl_planner_time(params, mu=0.9)
            else:
                print("Unknown MPC type, using crocoddyl linear")
                self.mpc_type = 1
                self.mpc = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, inner=False, linearModel=True)

        # Setup initial result for the first iteration of the main control loop
        x_init = np.zeros(12)
        x_init[0:3] = q_init[0:3, 0]
        x_init[3:6] = quaternionToRPY(q_init[3:7, 0]).ravel()
        if self.mpc_type == 3:
            self.last_available_result = np.zeros((32, self.n_nodes))
        elif self.mpc_type == 4:
            self.last_available_result = np.zeros((33, self.n_nodes))
        else:
            self.last_available_result = np.zeros((24, self.n_nodes))
        self.last_available_result[:24, 0] = np.hstack((x_init, np.array([0.0, 0.0, 8.0] * 4)))
        self.mpc_latest_result = np.zeros((12,))

    def solve(self, k, xref, footsteps, gait, l_target_footstep):
        """
        Call either the asynchronous MPC or the synchronous MPC depending on the value of multiprocessing during
        the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            footsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            gait (4xN array): Contact state of feet (gait matrix)
            l_target_footstep (3x4 array) : 4*[x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """
        if self.multiprocessing:
            self.run_MPC_asynchronous(k, xref, footsteps, l_target_footstep)
        else:
            self.run_MPC_synchronous(k, xref, footsteps, l_target_footstep)

        if k > 2:
            self.last_available_result[12:12 + self.n_nodes, :] = np.roll(self.last_available_result[12:12 + self.n_nodes, :], -1, axis=1)

        j = 0
        while np.any(gait[j + 1, :]):
            j += 1
        if k > 2 and not np.array_equal(gait[0, :], gait[j, :]):
            # TODO: grab from URDF
            mass = 2.5  
            n_contacts = np.sum(gait[j, :])
            F = 9.81 * mass / n_contacts
            self.last_available_result[12:24, self.n_nodes - 1] = np.zeros(12)
            for i in range(4):
                if (gait[j, i] == 1):
                    self.last_available_result[12 + 3 * i + 2, self.n_nodes - 1] = F

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration of the MPC
        If a new result is available, return the new result. Otherwise return the old result again.
        """
        if self.initialized:
            if not self.multiprocessing:
                return self.mpc_latest_result
            elif self.new_result.value:
                self.new_result.value = False
                self.last_available_result = self.convert_dataOut()
            return self.last_available_result
        else:
            self.initialized = True
            return self.last_available_result

    def run_MPC_synchronous(self, k, xref, footsteps, l_target_footstep):
        """
        Run the MPC (synchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            footsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            l_target_footstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """
        if self.mpc_type == 3 or self.mpc_type == 4:
            self.mpc.solve(k, xref.copy(), footsteps.copy(), l_target_footstep)
        else:
            self.mpc.solve(k, xref.copy(), footsteps.copy())

        self.mpc_latest_result = self.mpc.get_latest_result()

    def run_MPC_asynchronous(self, k, xref, footsteps, l_target_footstep):
        """
        Run the MPC (asynchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            xref (12xN): Desired state vector for the whole prediction horizon
            footsteps (12xN array): the [x, y, z]^T desired position of each foot for each time step of the horizon
            params (object): stores parameters
            l_target_footstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """
        if k == 0:
            p = Process(target=self.create_MPC_asynchronous, args=(self.new_data, self.new_result, self.dataIn, self.dataOut, self.running))
            p.start()

        self.compress_dataIn(k, xref, footsteps, l_target_footstep)
        self.newData.value = True

    def create_MPC_asynchronous(self, new_data, new_result, dataIn, running):
        """
        Parallel process with an infinite loop that run the asynchronous MPC

        Args:
            new_data (Value): shared variable that is true if new data is available, false otherwise
            new_result (Value): shared variable that is true if a new result is available, false otherwise
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
            running (Value): shared variable to stop the infinite loop when set to False
        """

        # print("Entering infinite loop")
        while running.value:
            # Checking if new data is available to trigger the asynchronous MPC
            if new_data.value:
                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                new_data.value = False

                # Retrieve data thanks to the decompression function and reshape it
                if self.mpc_type == 3 or self.mpc_type == 4:
                    kf, xref_1dim, footsteps_1dim, l_target_1dim = self.decompress_dataIn(dataIn)
                    l_target = np.reshape(l_target_1dim, (3, 4))
                else:
                    kf, xref_1dim, footsteps_1dim = self.decompress_dataIn(dataIn)

                # Reshaping 1-dimensional data
                k = int(kf[0])
                xref = np.reshape(xref_1dim, (12, self.n_nodes + 1))
                footsteps = np.reshape(footsteps_1dim, (self.N_gait, 12))

                # Create the MPC object of the parallel process during the first iteration
                if k == 0:
                    if self.mpc_type == 0:
                        loop_mpc = MPC.MPC(self.params)
                    elif self.mpc_type == 1:  # Crocoddyl MPC Linear
                        loop_mpc = MPC_crocoddyl.MPC_crocoddyl(self.params, mu=0.9, linearModel=True)
                    elif self.mpc_type == 2:  # Crocoddyl MPC Non-Linear
                        loop_mpc = MPC_crocoddyl.MPC_crocoddyl(self.params, mu=0.9, linearModel=False)
                    elif self.mpc_type == 3:  # Crocoddyl MPC Non-Linear
                        loop_mpc = MPC_crocoddyl_planner.MPC_crocoddyl_planner(self.params, mu=0.9)
                    else:  # Crocoddyl MPC Non-Linear with footsteps optimization
                        loop_mpc = MPC_crocoddyl_planner_time.MPC_crocoddyl_planner_time(self.params, mu=0.9)

                # Run the asynchronous MPC with the data that as been retrieved
                if self.mpc_type == 0:
                    footsteps[np.isnan(footsteps)] = 0.0
                    loop_mpc.run(np.int(k), xref, footsteps)
                elif self.mpc_type == 3 or self.mpc_type == 4:
                    loop_mpc.solve(k, xref.copy(), footsteps.copy(), l_target.copy())
                else:
                    loop_mpc.solve(k, xref.copy(), footsteps.copy())

                self.dataOut[:] = loop_mpc.get_latest_result().ravel(order='F')

                # Set shared variable to true to signal that a new result is available
                new_result.value = True

    def compress_dataIn(self, k, xref, footsteps, l_target_footstep):
        """
        Compress data in a single C-type array that belongs to the shared memory to send data from the main control
        loop to the asynchronous MPC

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
            l_target_footstep (3x4 array) : [x, y, z]^T target position in local frame, to stop the optimisation of the feet location around it
        """

        # Replace NaN values by 0.0 to be stored in C-type arrays
        footsteps[np.isnan(footsteps)] = 0.0

        # Compress data in the shared input array
        if self.mpc_type == 3 or self.mpc_type == 4:
            self.dataIn[:] = np.concatenate([[(k/self.k_mpc)], xref.ravel(), footsteps.ravel(), l_target_footstep.ravel()], axis=0)
        else:
            self.dataIn[:] = np.concatenate([[(k/self.k_mpc)], xref.ravel(), footsteps.ravel()], axis=0)

    def decompress_dataIn(self, dataIn):
        """
        Decompress data from a single C-type array that belongs to the shared memory to retrieve data from the main control
        loop in the asynchronous MPC

        Args:
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
        """

        # Sizes of the different variables that are stored in the C-type array
        if self.mpc_type == 3 or self.mpc_type == 4:
            sizes = [0, 1, (self.n_nodes + 1) * 12, 12 * self.N_gait, 12]
        else:
            sizes = [0, 1, (self.n_nodes + 1) * 12, 12 * self.N_gait]
        csizes = np.cumsum(sizes)

        # Return decompressed variables in a list
        return [dataIn[csizes[i]:csizes[i+1]] for i in range(len(sizes)-1)]

    def convert_dataOut(self):
        """
        Return the result of the asynchronous MPC (desired contact forces) that is stored in the shared memory
        """

        if self.mpc_type == 3:  # Need more space to store optimized footsteps
            return np.array(self.dataOut[:]).reshape((32, -1), order='F')
        else:
            return np.array(self.dataOut[:]).reshape((24, -1), order='F')

    def roll_asynchronous(self, footsteps):
        """
        Move one step further in the gait cycle. Since the output of the asynchronous MPC is retrieved by
        TSID during the next call to the MPC, it should not work with the current state of the gait but with the
        gait on step into the future. That way, when TSID retrieves the result, it is consistent with the current
        state of the gait.

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion).
        Simplification: instead of creating a new phase if required (see roll function of FootstepPlanner) we always
        increase the last one by 1 step. That way we don't need to call other functions to predict the position of
        footstep when a new phase is created.

        Args:
            footsteps (13xN_gait array): the remaining number of steps of each phase of the gait (first column)
            and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns)
        """
        self.footsteps_future = footsteps.copy()

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.footsteps_future[:, 0]) if val == 0.0), 0.0)[0]

        # Create a new phase if needed or increase the last one by 1 step
        self.footsteps_future[index-1, 0] += 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.footsteps_future[0, 0] > 1.0:
            self.footsteps_future[0, 0] -= 1.0
        else:
            self.footsteps_future = np.roll(self.footsteps_future, -1, axis=0)
            self.footsteps_future[-1, :] = np.zeros((13, ))

    def stop_parallel_loop(self):
        """
        Stop the infinite loop in the parallel process to properly close the simulation
        """
        self.running.value = False
