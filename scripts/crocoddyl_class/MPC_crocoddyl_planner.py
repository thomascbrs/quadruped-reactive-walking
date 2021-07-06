# coding: utf8

import crocoddyl
import numpy as np
import quadruped_walkgen as quadruped_walkgen
import pinocchio as pin


class MPC_crocoddyl_planner():
    """Wrapper class for the MPC problem to call the ddp solver and
    retrieve the results.

    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
    """

    def __init__(self, params,  mu=1, warm_start=False, min_fz=0.0):
        self.T_mpc = params.T_mpc  # Period of the MPC
        self.dt = params.dt_mpc    # Time step of the solver
        self.mass = 2.50000279     # Mass of the robot
        self.mu = mu               # Friction coefficient
        self.max_iteration = 10    # Max iteration ddp solver

        # Inertia matrix of the robot in body frame
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])

        # Weights
        self.stateWeights = np.zeros(12)
        self.stateWeights[:6] = [0.3, 0.3, 2, 0.9, 1., 0.4]
        self.stateWeights[6:] = [1.5, 2, 1, 0.05, 0.07, 0.05] * np.sqrt(self.stateWeights[:6])

        self.forceWeights = np.array(4*[0.01, 0.01, 0.01])  # Weight Vector : Force Norm
        self.frictionWeights = 0.5                          # Weight Vector : Friction cone cost
        self.heuristicWeights = np.array(4*[0.3, 0.4])      # Weights on the heuristic term
        self.stepWeights = np.full(8, 0.05)                 # Weight on the step command (distance between steps)
        self.stopWeights = np.ones(8)                       # Weights to stop the optimisation at the end of the flying phase
        self.shoulderContactWeight = 5                      # Weight for shoulder-to-contact penalty
        self.shoulder_hlim = 0.225

        # TODO : create a proper warm-start with the previous optimisation
        self.warm_start = warm_start

        # Minimum normal force(N) and reference force vector bool
        self.min_fz = min_fz

        # Gait matrix
        self.gait = np.zeros((params.N_gait, 4))
        self.gait_old = np.zeros(4)

        # Position of the feet in local frame
        self.fsteps = np.zeros((params.N_gait, 12))

        # List to generate the problem
        self.action_models = []
        self.x_init = []
        self.u_init = []

        self.problem = None   # Shooting problem
        self.ddp = None  # ddp solver
        self.Xs = np.zeros((20, int(self.T_mpc/self.dt)))  # Xs results without the actionStepModel

        # Initial foot location (local frame, X,Y plan)
        self.shoulders = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946,   0.14695, -0.1946,  -0.14695]

        # Index to stop the feet optimisation
        self.index_lock_time = int(params.lock_time / params.dt_mpc)  # Row index in the gait matrix when the optimisation of the feet should be stopped
        self.index_stop_optimisation = []  # List of index to reset the stopWeights to 0 after optimisation

        self.initialize_models(params)

    def initialize_models(self, params):
        ''' Reset the two lists of augmented and step-by-step models, to avoid recreating them at each loop.
        Not all models here will necessarily be used.  

        Args : 
            - params : object containing the parameters of the simulation
        '''
        self.models_augmented = []
        self.models_step = []

        for _ in range(params.N_gait):
            model = quadruped_walkgen.ActionModelQuadrupedAugmented()
            self.update_model_augmented(model)
            self.models_augmented.append(model)

        for _ in range(4 * int(params.T_gait / params.T_mpc)):
            model = quadruped_walkgen.ActionModelQuadrupedStep()
            self.update_model_step(model)
            self.models_step.append(model)

        # Terminal node
        self.terminal_model = quadruped_walkgen.ActionModelQuadrupedAugmented()
        self.update_model_augmented(self.terminal_model, terminal=True)

    def solve(self, k, xref, footsteps, l_stop):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : the desired state vector
            footsteps : current position of the feet (given by planner)
            l_stop : current and target position of the feet (given by footstepTragectory generator)
        """
        self.updateProblem(k, xref, footsteps, l_stop)
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Reset to 0 the stopWeights for next optimisation
        for index_stopped in self.index_stop_optimisation:
            self.models_augmented[index_stopped].stopWeights = np.zeros(8)

    def updateProblem(self, k, xref, footsteps, l_stop):
        """
        Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state.

        Args:
        """
        self.compute_gait_matrix(footsteps)

        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.shoulders \
            + np.repeat(self.gait[0, :], 2) * footsteps[0, [0, 1, 3, 4, 6, 7, 9, 10]]

        self.x_init.clear()
        self.u_init.clear()
        self.action_models.clear()
        self.index_stop_optimisation.clear()

        index_step = 0
        index_augmented = 0
        j = 0

        while np.any(self.gait[j, :]):
            if j == 0:
                if np.any(self.gait[0, :] - self.gait_old):
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                             xref[:, j+1], self.gait[0, :] - self.gait_old)
                    self.action_models.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                       l_stop, xref[:, j+1], self.gait[j, :])

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time:
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)

                    self.action_models.append(self.models_augmented[index_augmented])

                    index_step += 1
                    index_augmented += 1
                    
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.zeros(8))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

                else:
                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                       l_stop, xref[:, j+1], self.gait[j, :])
                    self.action_models.append(self.models_augmented[index_augmented])

                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

            else:
                if np.any(self.gait[j, :] - self.gait[j-1, :]):
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                             xref[:, j+1], self.gait[j, :] - self.gait[j-1, :])
                    self.action_models.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                       l_stop, xref[:, j+1], self.gait[j, :])

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time:
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)

                    self.action_models.append(self.models_augmented[index_augmented])

                    index_step += 1
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.zeros(8))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

                else:
                    self.models_augmented[index_augmented].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                                       l_stop, xref[:, j+1], self.gait[j, :])
                    self.action_models.append(self.models_augmented[index_augmented])

                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0]))
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])]))

            # Update row matrix
            j += 1

        # Update terminal model
        self.terminal_model.updateModel(np.reshape(footsteps[j-1, :], (3, 4), order='F'), l_stop, xref[:, -1], self.gait[j-1, :])
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, j-1], p0]))

        self.problem = crocoddyl.ShootingProblem(np.zeros(20),  self.action_models, self.terminal_model)
        self.problem.x0 = np.concatenate([xref[:, 0], p0])

        self.ddp = crocoddyl.SolverDDP(self.problem)

    def get_latest_result(self):
        """ 
        Return the desired contact forces that have been computed by the last iteration of the MPC
        """
        index = 0
        N = int(self.T_mpc / self.dt)
        result = np.zeros((32, N))
        for i in range(len(self.action_models)):
            if self.action_models[i].__class__.__name__ != "ActionModelQuadrupedStep":
                if index >= N:
                    raise ValueError("Too many action model considering the current MPC prediction horizon")
                result[:12, index] = self.ddp.xs[i][:12]
                result[12:24, index] = self.ddp.us[i]
                result[24:, index] = self.ddp.xs[i][12:]
                if i > 0 and self.action_models[i-1].__class__.__name__ == "ActionModelQuadrupedStep":
                    pass
                index += 1

        return result

    def update_model_augmented(self, model, terminal=False):
        """ 
        Set intern parameters for augmented model type
        """
        # Model parameters
        model.dt = self.dt
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.relative_forces = True
        model.shoulderContactWeight = self.shoulderContactWeight

        # Weights vectors
        model.stateWeights = self.stateWeights
        model.stopWeights = np.zeros(8)

        if terminal:
            self.terminal_model.forceWeights = np.zeros(12)
            self.terminal_model.frictionWeights = 0.
            self.terminal_model.heuristicWeights = np.zeros(8)
        else:
            model.frictionWeights = self.frictionWeights
            model.forceWeights = self.forceWeights
            model.heuristicWeights = self.heuristicWeights

    def update_model_step(self, model):
        """
        Set intern parameters for step model type
        """
        model.heuristicWeights = np.zeros(8)
        model.stateWeights = self.stateWeights
        model.stepWeights = self.stepWeights

    def compute_gait_matrix(self, footsteps):
        """ 
        Recontruct the gait based on the computed footstepsC
        Args:
            footsteps : current and predicted position of the feet
        """

        self.gait_old = self.gait[0, :].copy()

        j = 0
        self.gait = np.zeros(np.shape(self.gait))
        while np.any(footsteps[j, :]):
            self.gait[j, :] = (footsteps[j, ::3] != 0.0).astype(int)
            j += 1
