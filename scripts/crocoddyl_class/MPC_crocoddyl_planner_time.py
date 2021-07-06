import crocoddyl
import numpy as np
import quadruped_walkgen
import pinocchio as pin


class MPC_crocoddyl_planner_time():
    """Wrapper class for the MPC problem to call the ddp solver and
    retrieve the results.
    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
    """

    def __init__(self, params,  mu=1, warm_start=False, min_fz=0.0):
        self.dt = params.dt_mpc                   # Time step of the solver
        self.T_mpc = params.T_mpc                 # Period of the MPC
        self.n_nodes = int(self.T_mpc/self.dt)    # Number of nodes
        self.mass = 2.50000279                    # Mass of the robot
        self.mu = mu                              # Friction coefficient
        self.max_iteration = 100                  # Max iteration ddp solver

        self.centrifugal_term = True              # symmetry term in foot position heuristic
        self.symmetry_term = True                 # centrifugal term in foot position heuristic

        self.warm_start = warm_start              # Warm Start for the solver
        self.x_init = []                          # Inital x
        self.u_init = []                          # initial u
        self.action_models = []                   # list of actions

        self.gait = np.zeros((20, 4))             # Gait matrix
        self.gait_old = np.zeros((20, 5))         # Last gait matrix

        # Inertia matrix of the robot in body frame
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])

        self.stateWeights = np.zeros(12)
        self.stateWeights[:6] = [0.3, 0.3, 2, 0.9, 1., 0.4]
        self.stateWeights[6:] = [1.5, 2, 1, 0.05, 0.07, 0.05] * np.sqrt(self.stateWeights[:6])

        self.forceWeights = 0.01 * np.ones(12)     # Weight Vector : Force Norm
        self.frictionWeights = 10.                 # Weight Vector : Friction cone cost
        self.stepWeights = 0.4 * np.ones(4)        # Weight on the step command
        self.lastPositionWeights = 2. * np.ones(8)  # Weights on the previous position predicted
        self.heuristicWeights = np.zeros(8)        # weight on the heuristic

        self.min_fz = min_fz  # Minimum normal force (N)
        self.max_fz = 25      # Maximum normal force (N)

        self.fsteps = np.zeros((20, 13))  # Position of the feet in local frame

        self.dt_ref = 0.02
        self.dt_weight = 0.0  # Weight on the step command for dt

        self.T_gait_min = 0.12
        self.T_gait_max = 0.46     # 460ms for half period
        self.dt_min = self.T_gait_min / self.n_nodes / 2
        self.dt_max = self.T_gait_max / self.n_nodes / 2
        self.dt_weight_bound = 0

        self.speedWeight = 0
        self.nb_nodes = self.T_mpc/self.dt / 2 - 1

        self.stop_optim = 0.1
        self.index_stop = int((1 - self.stop_optim)*(int(0.5*self.T_mpc/self.dt) - 1))

        # Index of the control cycle to start the "stopping optimisation"
        self.start_stop_optim = 20

        self.l_fsteps = np.zeros((3, 4))
        self.o_fsteps = np.zeros((3, 4))

        self.problem = None  # Shooting problem
        self.ddp = None      # ddp solver

        self.Xs = np.zeros((21, self.n_nodes))

        self.shoulders = np.array([0.1946, 0.15005, 0.1946, -0.15005, -0.1946,   0.15005, -0.1946,  -0.15005])

        # Weights for dt model :
        self.dt_weight_bound_cmd = 1000000.  # Upper/lower bound
        self.dt_weight_cmd = 0.  # ||U-dt||^2

        self.relative_forces = True

        # Param for gait
        self.nb_nodes_horizon = 8
        self.T_min = 0.32
        self.T_max = 0.82
        self.node_init = self.T_min/(2*self.dt) - 1
        self.dt_init = self.dt
        self.terminal_factor = 5
        self.o_feet = np.zeros((3, 4))
        self.xref = None
        self.results_dt = np.zeros(2)   # dt1 and dt2 --dt0 always to 0.01

        # Weight on the shoulder term :
        self.shoulderWeights = 0.1
        self.shoulder_hlim = 0.225
        self.initialSpeedWeight = 1.
        self.speedWeight = 5

        self.feet_param = np.zeros((3, 4))
        self.first_step = True

        # Weight for period optim
        self.vlim = 1.5

    def solve(self, k, xref, footsteps):
        """ Solve the MPC problem 
        Args:
            k : Iteration 
            xref : the desired state vector
            l_feet : current position of the feet
        """
        self.updateProblem(k, xref, footsteps)
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # for i in range(len(self.ddp.problem.runningModels)):
        #     if self.ddp.problem.runningModels[i].nu == 4:
        #         if self.ddp.problem.runningModels[i].first_step == True:
        #             self.ddp.problem.runningModels[i].speedWeight = self.initialSpeedWeight
        #         else:
        #             self.ddp.problem.runningModels[i].speedWeight = self.speedWeight
        #         self.ddp.problem.runningModels[i].stepWeights = self.stepWeights
        #     if self.ddp.problem.runningModels[i].nu == 1:
        #         self.ddp.problem.runningModels[i].dt_weight_cmd = 0.

        # self.ddp.solve(self.ddp.xs, self.ddp.us, 100, isFeasible=True)
        # self.get_fsteps()

    def updateProblem(self, k, xref, footsteps):
        """Update the dynamic of the model list according to the predicted position of the feet, 
        and the desired state. 
        self.o_feet = np.zeros((3, 4))  # position of feet in world frame
        Args:
        """
        self.gait_old = self.gait[0, :].copy()

        self.compute_gait_matrix(footsteps)

        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.shoulders \
            + np.repeat(self.gait[0, :], 2) * footsteps[0, [0, 1, 3, 4, 6, 7, 9, 10]]

        if k == 0:
            self.create_List_model()
            self.gait_old = self.gait[0, :].copy()
        else:
            self.roll_models()

        self.update_action_models(k, xref, footsteps)

        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.action_models, self.terminal_model)
        self.problem.x0 = np.concatenate([xref[:, 0], p0, [self.dt_init]])

        self.ddp = crocoddyl.SolverDDP(self.problem)

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """
        index = 0
        result = np.zeros((33, self.n_nodes))
        for i in range(len(self.action_models)):
            if self.action_models[i].__class__.__name__ == "ActionModelQuadrupedAugmentedTime":
                if index >= self.n_nodes:
                    raise ValueError("Too many action model considering the current MPC prediction horizon")
                result[:12, index] = self.ddp.xs[i][:12]
                result[12:24, index] = self.ddp.us[i]
                result[24:, index] = self.ddp.xs[i][12:]
                index += 1

        return result

    def update_model_augmented(self, model):
        """
        Set intern parameters for augmented model type
        """
        # Model parameters
        model.dt = self.dt
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.max_fz = self.max_fz

        # Weights vectors
        model.stateWeights = self.stateWeights
        model.forceWeights = self.forceWeights
        model.frictionWeights = self.frictionWeights
        model.heuristicWeights = self.heuristicWeights

        # Feet positions parameters
        model.lastPositionWeights = np.zeros(8)
        model.shoulderWeights = self.shoulderWeights
        model.shoulder_hlim = self.shoulder_hlim
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term

        # Time parameters
        model.dt_ref = self.dt_ref
        model.dt_weight = 0.
        model.dt_weight_bound = self.dt_weight_bound
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max
        model.shoulders = self.shoulders

        model.relative_forces = self.relative_forces

    def update_model_step(self, model, optim_period=True):
        """ 
        Set intern parameters for step model type
        """
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term

        model.vlim = self.vlim
        model.nb_nodes = self.nb_nodes

        if optim_period:
            model.stepWeights = self.stepWeights
            model.speedWeight = self.speedWeight
            model.heuristicWeights = np.zeros(8)
            model.stateWeights = np.zeros(12)
        else:
            model.stepWeights = self.stepWeights
            model.speedWeight = 0.
            model.heuristicWeights = self.heuristicWeights
            model.stateWeights = self.stateWeights

    def update_model_time(self, model, optim_period=True):
        """ Set intern parameters for step model type
        """
        model.dt_ref = self.dt_ref
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term
        model.T_gait = self.T_mpc

        model.dt_weight_bound_cmd = self.dt_weight_bound_cmd
        model.dt_weight_cmd = 0.
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max

        if optim_period:
            model.heuristicWeights = np.zeros(8)
            model.stateWeights = np.zeros(12)
        else:
            model.heuristicWeights = self.heuristicWeights
            model.stateWeights = self.stateWeights

        model.dt_min = self.dt_min
        model.dt_max = self.dt_max
        model.dt_weight_cmd = 1000
        model.dt_ref = self.dt_ref

        # modelTime.dt_min = self.dt
        # modelTime.dt_max = self.T_max/(2*(self.nb_nodes + 1))
        # modelTime.dt_weight_cmd = 1000
        # modelTime.dt_ref = modelTime.dt_min

    def create_List_model(self):
        """
        Create the list of action models and initialize them
        Start with a time action model, then add the first step action models, and then a step 
        followed by a time before the second step, and so on.
        """
        j = 0

        model = quadruped_walkgen.ActionModelQuadrupedTime()
        self.update_model_time(model, True)
        self.action_models.append(model)

        while np.any(self.gait[j, :]):
            model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
            self.update_model_augmented(model)
            self.action_models.append(model)

            if np.any(self.gait[j+1, :]) and not np.array_equal(self.gait[j, :], self.gait[j+1, :]):
                model = quadruped_walkgen.ActionModelQuadrupedStepTime()
                self.update_model_step(model)
                self.action_models.append(model)

                model = quadruped_walkgen.ActionModelQuadrupedTime()
                self.update_model_time(model, True)
                self.action_models.append(model)

            j += 1

        # Model parameters of terminal node
        self.terminal_model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
        self.update_model_augmented(self.terminal_model)

        # Weights vectors of terminal node
        self.terminal_model.forceWeights = np.zeros(12)
        self.terminal_model.frictionWeights = 0.
        self.terminal_model.heuristicWeights = np.zeros(8)
        self.terminal_model.lastPositionWeights = np.zeros(8)
        self.terminal_model.stateWeights = 5 * self.terminal_model.stateWeights

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.action_models, self.terminal_model)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)



    def update_action_models(self, k, xref, footsteps):
        """
        Update the list of action models
        """
        x0 = np.zeros(21)
        x0[2] = 0.2027
        x0[-1] = self.dt_init
        u0 = np.array([0.1, 0.1, 8, 0.1, 0.1, 8, 0.1, 0.1, 8, 0.1, 0.1, 8])
        self.x_init = []
        self.u_init = []

        j = 0  # phase of te gait
        next_step_flag = True
        next_step_index = 0  # Get index of incoming step for updatePositionWeights

        self.action_models[0].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'), xref[:, j], self.gait[0, :] - self.gait_old)
        self.x_init.append(x0)
        self.u_init.append(np.array([self.dt_init]))

        index = 1  # index of action model in self.action_models
        while np.any(self.gait[j, :]):
            if self.action_models[index].__class__.__name__ == "ActionModelQuadrupedStepTime":
                if next_step_flag:
                    next_step_index = index
                    next_step_flag = False

                if j == 0:
                    self.action_models[index].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                          xref[:, j], self.gait[0, :] - self.gait_old)
                else:
                    self.action_models[index].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                          xref[:, j], self.gait[j, :] - self.gait[j-1, :])
                self.x_init.append(x0)
                self.u_init.append(np.zeros(4))

                self.action_models[index+1].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'),
                                                        xref[:, j], self.gait[j, :])
                self.x_init.append(x0)
                self.u_init.append(np.array([self.dt_init]))

                self.action_models[index+2].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'), xref[:, j], self.gait[j, :])
                self.x_init.append(x0)
                self.u_init.append(np.repeat(self.gait[j, :], 3) * u0)

                index += 2

            else:
                self.action_models[index].updateModel(np.reshape(footsteps[j, :], (3, 4), order='F'), xref[:, j], self.gait[j, :])
                self.x_init.append(x0)
                self.u_init.append(np.repeat(self.gait[j, :], 3) * u0)

            j += 1
            index += 1

        # if k > self.start_stop_optim:
        #     self.updatePositionWeights(next_step_index)  # Update the lastPositionweight

        self.terminal_model.updateModel(np.reshape(footsteps[j-1, :], (3, 4), order='F'), xref[:, -1], self.gait[j-1, :])
        self.x_init.append(x0)

    def get_fsteps(self):
        """Create the matrices fstep, the position of the feet predicted during the control cycle.
        To be used after the solve function.
        """
        ##################################################
        # Get command vector without actionModelStep node
        ##################################################

        Us = self.ddp.us
        Liste = [x for x in Us if (x.size != 4 and x.size != 1)]
        self.Us = np.array(Liste)[:, :].transpose()

        ################################################
        # Get state vector without actionModelStep node
        ################################################

        Xs = [self.ddp.xs[i] for i in range(len(self.ddp.us)) if (self.ddp.us[i].size != 4 and self.ddp.us[i].size != 1)]
        Xs.append(self.ddp.xs[-1])  # terminal node
        self.Xs = np.array(Xs).transpose()

        # Dt optimised
        results_dt = [self.ddp.xs[i+1][-1] for i in range(len(self.ddp.us)) if self.ddp.us[i].size == 1]

        if len(results_dt) == 2:
            self.results_dt[:2] = results_dt[:2]
        else:
            self.results_dt[:2] = results_dt[1:3]

        ########################################
        # Compute fheuristicWeights steps using the state vector
        ################################16 ########

        j = 0
        k_cum = 0

        self.ListPeriod = []
        self.gait_new = np.zeros((20, 5))

        # Iterate over all phases of the gait
        while (self.gait[j, 0] != 0):

            self.fsteps[j, 1:] = np.repeat(self.gait[j, 1:], 3)*np.concatenate([self.Xs[12:14, k_cum], [0.], self.Xs[14:16, k_cum], [0.],
                                                                                self.Xs[16:18, k_cum], [0.], self.Xs[18:20, k_cum], [0.]])
            if int(self.gait[0, 0]) > 1:
                if j == 0:
                    # k_cum + 1 --> 1st is at dt, optim after the first node
                    self.fsteps[j, 0] = np.around(1 + ((self.gait[j, 0] - 1)*self.Xs[20, k_cum + 1] / self.dt), decimals=0)
                    self.gait_new[j, 0] = np.around(1 + ((self.gait[j, 0] - 1)*self.Xs[20, k_cum + 1] / self.dt), decimals=0)
                elif j == 1 and int(np.sum(self.gait[1, 1:])) == 4:
                    self.fsteps[j, 0] = self.gait[j, 0]
                    self.gait_new[j, 0] = self.gait[j, 0]
                else:
                    self.fsteps[j, 0] = self.gait[j, 0]
                    self.gait_new[j, 0] = self.gait[j, 0]
            else:
                if j == 1 and np.sum(self.gait[1, 1:]) == 4:
                    self.fsteps[j, 0] = np.around((self.gait[j, 0]*self.Xs[20, k_cum] / self.dt), decimals=0)
                    self.gait_new[j, 0] = np.around((self.gait[j, 0]*self.Xs[20, k_cum] / self.dt), decimals=0)
                else:
                    self.fsteps[j, 0] = self.gait[j, 0]
                    self.gait_new[j, 0] = self.gait[j, 0]

            self.gait_new[j, 1:] = self.gait[j, 1:]

            k_cum += np.int(self.gait[j, 0])
            j += 1

        self.gait = self.gait_new

        ####################################################
        # Compute the current position of feet in contact
        # and the position of desired feet in flying phase
        # in local frame
        #####################################################

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(self.fsteps[:, 3*i+1]) if ((not (val == 0)) and (not np.isnan(val)))), [-1])[0]
            pos_tmp = np.array([self.fsteps[index, (1+i*3):(4+i*3)]]).transpose()
            self.l_fsteps[:2, i] = pos_tmp[:2]

            pos_tmp = self.oMl * pos_tmp.copy()
            self.o_fsteps[:2, i] = pos_tmp[:2]

        return self.fsteps

    def updatePositionWeights(self):
        """Update the parameters in the action_models to keep the next foot position at the same position computed by the 
         previous control cycle and avoid re-optimization at the end of the flying phase
        """

        if self.gait[0, 0] == self.index_stop:
            self.action_models[int(self.gait[0, 0]) + 1].lastPositionWeights = np.repeat((np.array([1, 1, 1, 1]) - self.gait[0, 1:]), 2) * self.lastPositionWeights

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column is deleted as it corresponds
        initial state vector
        Args:
        """
        return np.array(self.ddp.xs)[1:, :].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon, 
        Args:
        """
        return np.array(self.ddp.us)[:, :].transpose()[:, :]

    def compute_gait_matrix(self, footsteps):
        """ Compute the gait matrix
        Args:
            footsteps : current and predicted position of the feet
        """
        j = 0
        self.gait = np.zeros(np.shape(self.gait))
        while np.any(footsteps[j, :]):
            self.gait[j, :] = (footsteps[j, ::3] != 0.0).astype(int)  # Recontruct the gait based on the computed footsteps
            j += 1

    def roll_models(self):
        """
        Move one step further in the gait cycle
        Add and remove corresponding model in action_models
        """
        step_first = False
        if self.action_models[0].__class__.__name__ == "ActionModelQuadrupedStepTime":
            self.action_models.pop(0)
            step_first = True
        model = self.action_models.pop(0)  # Remove first model

        if step_first:  # Add last model & step model if needed
            modelStep = quadruped_walkgen.ActionModelQuadrupedStep()
            self.update_model_step(modelStep)
            self.action_models.append(modelStep)

        model.lastPositionWeights = np.zeros(8)  # reset to 0 the weight lastPosition
        self.action_models.append(model)
