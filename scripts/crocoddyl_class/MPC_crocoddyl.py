import crocoddyl
import numpy as np
import quadruped_walkgen as quadruped_walkgen

class MPC_crocoddyl:
    """Wrapper class for the MPC problem to call the ddp solver and
    retrieve the results.

    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
        linearModel(bool) : Approximation in the cross product by using desired state
    """

    def __init__(self, params,  mu=1, linearModel=True):
        self.dt = params.dt_mpc                   # Time step of the solver
        self.T_mpc = params.T_mpc                 # Period of the MPC
        self.n_nodes = int(self.T_mpc/self.dt)    # Number of nodes
        self.mass = 2.50000279                    # Mass of the robot
        self.mu = mu                              # Friction coefficient
        self.max_iteration = 25                   # Max iteration ddp solver
                
        self.warm_start = True                    # Warm Start for the solver
        self.x_init = []                          # Inital x
        self.u_init = []                          # initial u

        self.gait = np.zeros((params.N_gait, 4))  # Gait matrix
        self.index = 0                            # Index of the gait matrix

        self.implicit_integration = False         # Integration scheme

        # Inertia matrix of the robot in body frame
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])

        # Gains
        # self.stateWeights = np.zeros(12)
        # self.stateWeights[:6] = [0.5, 0.5, 2., 0.11, 0.11, 0.11]
        # self.stateWeights[6:] = [2., 2., 2., 0.05, 0.05, 0.05] * np.sqrt(self.stateWeights[:6])
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 2 * 10
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.4
        self.w_vx = 1.5*np.sqrt(self.w_x)
        self.w_vy = 2*np.sqrt(self.w_y)
        self.w_vz = 1*np.sqrt(self.w_z)
        self.w_vroll = 0.05*np.sqrt(self.w_roll)
        self.w_vpitch = 0.07*np.sqrt(self.w_pitch)
        self.w_vyaw = 0.05*np.sqrt(self.w_yaw)
        self.stateWeights = np.array([self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
                                      self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])

        self.forceWeights = 0.02 * np.ones(12)  # Weight Vector : Force Norm
        self.frictionWeights = 1.0  # Weight Vector : Friction cone cost

        self.min_fz = 0.2  # Minimum normal force (N)
        self.max_fz = 25  # Maximum normal force (N)

        self.shoulderWeights = 1.  # Weight on the shoulder term
        self.shoulder_hlim = 0.23  # shoulder maximum height

        self.fsteps = np.full((params.N_gait, 12), np.nan)  # Position of the feet

        # Action models
        if linearModel:
            self.ListAction = [quadruped_walkgen.ActionModelQuadruped() for _ in range(self.n_nodes)]
            self.terminalModel = quadruped_walkgen.ActionModelQuadruped()
        else:
            self.ListAction = [quadruped_walkgen.ActionModelQuadrupedNonLinear() for _ in range(self.n_nodes)]
            self.terminalModel = quadruped_walkgen.ActionModelQuadrupedNonLinear()
        self.updateActionModels()

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(12),  self.ListAction, self.terminalModel)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

    def updateProblem(self, fsteps, xref):
        """Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state

        Args:
            fsteps (6x13): Position of the feet in local frame
            xref (12x17): Desired state vector for the whole gait cycle
            (the initial state is the first column)
        """
        # Update position of the feet
        self.fsteps[:, :] = fsteps[:, :]

        # Update initial state of the problem
        self.problem.x0 = xref[:, 0]

        # Construction of the gait matrix representing the feet in contact with the ground
        self.index = 0
        while (np.any(self.fsteps[self.index, :])):
            self.index += 1
        self.gait[:self.index, :] = 1.0 - (self.fsteps[:self.index, 0::3] == 0.0)
        self.gait[self.index:, :] = 0.0

        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state
        for j in range(self.index):
            # Update model
            self.ListAction[j].updateModel(np.reshape(self.fsteps[j, :], (3, 4), order='F'),
                                           xref[:, j+1], self.gait[j, :])

        # Update model of the terminal model
        self.terminalModel.updateModel(np.reshape(
            self.fsteps[self.index-1, :], (3, 4), order='F'), xref[:, -1], self.gait[self.index-1, :])

    def solve(self, k, xref, fsteps):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : desired state vector
            fsteps : feet predicted positions
        """
        # Update the dynamic depending on the predicted feet position
        self.updateProblem(fsteps, xref)

        self.x_init.clear()
        self.u_init.clear()

        # Warm start : set candidate state and input vector
        if self.warm_start and k != 0:
            self.u_init = self.ddp.us[1:]
            self.u_init.append(np.repeat(self.gait[self.index-1, :], 3)*np.array(4*[0.5, 0.5, 5.]))

            self.x_init = self.ddp.xs[2:]
            self.x_init.insert(0, xref[:, 0])
            self.x_init.append(self.ddp.xs[-1])

        self.ddp.solve(self.x_init,  self.u_init, self.max_iteration)

    def get_latest_result(self):
        """Returns the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """
        output = np.zeros((24, self.n_nodes))
        for i in range(self.n_nodes):
            output[:12, i] = np.asarray(self.ddp.xs[i+1])
            output[12:, i] = np.asarray(self.ddp.us[i])
        return output

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column
        is deleted as it corresponds initial state vector
        Args:
        """
        return np.array(self.ddp.xs)[1:, :].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon,
        Args:
        """
        return np.array(self.ddp.us)[:, :].transpose()[:, :]

    def initializeActionModel(self, model, terminal=False):
        """ Initialize an action model with the parameters"""
        # Model parameters
        model.dt = self.dt
        model.mass = self.mass
        model.gI = self.gI
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.max_fz = self.max_fz

        # Weights vectors
        model.stateWeights = self.stateWeights
        if terminal:
            model.forceWeights = np.zeros(12)
            model.frictionWeights = 0.
        else:
            model.max_fz = self.max_fz
            model.forceWeights = self.forceWeights
            model.frictionWeights = self.frictionWeights

        # shoulder term :
        model.shoulderWeights = self.shoulderWeights
        model.shoulder_hlim = self.shoulder_hlim

        # integration scheme
        model.implicit_integration = self.implicit_integration

    def updateActionModels(self):
        """Update the quadruped model with the new weights or model parameters.
        Useful to try new weights without modify this class
        """
        for model in self.ListAction:
            self.initializeActionModel(model)

        self.initializeActionModel(self.terminalModel, terminal=True)
