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
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5], 
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])   # Inertia matrix of the robot in body frame       

        # Not used
        # self.centrifugal_term = True              # symmetry term in foot position heuristic
        # self.symmetry_term = True                 # centrifugal term in foot position heuristic

        self.warm_start = warm_start              # Warm Start for the solver
        self.x_init = []                          # Inital x
        self.u_init = []                          # initial u
        self.action_models = []                   # list of actions

        self.gait = np.zeros((params.N_gait, 4))             # Gait matrix
        self.gait_old = np.zeros((1, 4))        # Last gait matrix
        
        # Weights and parameters generals (augmented model)
        # self.stateWeights = np.zeros(12)
        # self.stateWeights[:6] = [0.3, 0.3, 2, 0.9, 1., 0.4]
        # self.stateWeights[6:] = [1.5, 2, 1, 0.05, 0.07, 0.05] * np.sqrt(self.stateWeights[:6])
        self.stateWeights = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 10.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3]) # fit osqp gains

        self.forceWeights = 0.01 * np.ones(12)     # Weight Vector : Force Norm
        self.frictionWeights = 1.                 # Weight Vector : Friction cone cost
        self.stopWeights = 0. * np.ones(8)  # Weights on the previous position predicted
        self.heuristicWeights = 0.5*np.ones(8)        # weight on the heuristic

        # Weight on the shoulder term :
        self.shoulderContactWeight = 0.
        self.shoulder_hlim = 0.225

        self.relative_forces = False
        self.min_fz = min_fz  # Minimum normal force (N)
        self.max_fz = 25      # Maximum normal force (N)      

        self.T_gait_min = 0.12
        self.T_gait_max = 0.46     # 460ms for half period
        self.dt_min = 2 * self.T_gait_min / self.n_nodes 
        self.dt_max = 2 * self.T_gait_max / self.n_nodes 

        # Weights for augmented models
        self.dt_weight_bound = 0

        # Weights for time model :
        self.dt_weight_bound_cmd = 1000000.  # Upper/lower bound
        self.dt_weight_cmd = 100000.  # ||U-dt_ref||^2    
        self.dt_ref = 0.02

        # Weights for step model
        self.stepWeights = 0.4 * np.ones(8)        # Weight on the step command
        self.speedWeight = 0.
        self.nb_nodes_nominal = self.T_mpc/self.dt / 2 - 1
        self.vlim = 1.5

        # DDP problem
        self.problem = None  # Shooting problem
        self.ddp = None      # ddp solver
        self.max_iteration = 100    # Max iteration ddp solver
        self.terminal_factor = 1   # Weight on the terminal node
        
        # Param for warm start
        self.dt_init = self.dt        

        # Initial foot location (local frame, X,Y plan)
        self.p0 = [0.1946, 0.14695, 0.1946, -0.14695, -0.1946,   0.14695, -0.1946,  -0.14695]
        # Index to stop the feet optimisation 
        self.index_lock_time = int(params.lock_time / params.dt_mpc ) # Row index in the gait matrix when the optimisation of the feet should be stopped
        self.index_stop_optimisation = [] # List of index to reset the stopWeights to 0 after optimisation

        self.initializeModels(params)

    def initializeModels(self, params):
        ''' Reset the 3 lists of augmented, step and step-time models, to avoid recreating them at each loop.
        Not all models here will necessarily be used.  

        Args : 
            - params : object containing the parameters of the simulation
        '''
        self.models_augmented = []
        self.models_step = []
        self.models_time = []

        for j in range(params.N_gait) :
            model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()            
            
            self.update_model_augmented(model)
            self.models_augmented.append(model)

        for j in range(4 * int(params.T_gait / params.T_mpc) ) :
            model = quadruped_walkgen.ActionModelQuadrupedStepTime()  
            model_time = quadruped_walkgen.ActionModelQuadrupedTime()   
            
            self.update_model_step(model)
            self.models_step.append(model)
            self.update_model_time(model_time)
            self.models_time.append(model_time)
        
        # Terminal node
        self.terminal_model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime() 
        self.update_model_augmented(self.terminal_model)
        # Weights vectors of terminal node
        self.terminal_model.forceWeights = np.zeros(12)
        self.terminal_model.frictionWeights = 0.
        self.terminal_model.heuristicWeights = np.zeros(8)
        self.terminal_model.stopWeights = np.zeros(8)
        self.terminal_model.stateWeights = self.terminal_factor * self.terminal_model.stateWeights

        return 0

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
        model.stopWeights = np.zeros(8)
        model.shoulderContactWeight = self.shoulderContactWeight
        model.shoulder_hlim = self.shoulder_hlim

        # Not used
        # model.symmetry_term = self.symmetry_term
        # model.centrifugal_term = self.centrifugal_term

        # Time parameters
        model.dt_ref = self.dt_ref
        model.dt_weight_bound = self.dt_weight_bound
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max

        model.relative_forces = self.relative_forces

    def update_model_step(self, model, optim_period=True):
        """ 
        Set intern parameters for step model type
        """
        # Not used
        # model.symmetry_term = self.symmetry_term
        # model.centrifugal_term = self.centrifugal_term

        model.stepWeights = self.stepWeights        
        model.stateWeights = self.stateWeights

        # The Gait matrix given corresponds to the flying feet
        # and not the ones already on the ground, where this cost is usefull
        model.heuristicWeights = np.zeros(8)

        model.vlim = self.vlim
        model.nb_nodes = self.nb_nodes_nominal

        if optim_period:            
            model.speedWeight = self.speedWeight
        else:
            model.speedWeight = 0.
            

    def update_model_time(self, model):
        """ Set intern parameters for step model type
        """
        # Not used
        # model.symmetry_term = self.symmetry_term
        # model.centrifugal_term = self.centrifugal_term
        # model.T_gait = self.T_mpc

        model.dt_ref = self.dt_ref
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max
        
        # Weights command 
        model.dt_weight_bound_cmd = self.dt_weight_bound_cmd
        model.dt_weight_cmd = self.dt_weight_cmd

        # Weights on states
        model.heuristicWeights = np.zeros(8) # Not used to simplify for now
        model.stateWeights = self.stateWeights
       
    def solve(self, k, xref, l_feet, l_stop, velocity, acceleration):
        """ Solve the MPC problem

        Args:
            k : Iteration
            xref : the desired state vector
            l_feet (Array 3x4): current position of the feet (given by planner)
            l_stop (Array 3x4): current and target position of the feet (given by footstepTragectory generator)
            velocity (Array 3x4): Current velocity of the flying feet
            acceleration (Array 3x4): Current acceleration of the flying feet
        """

        # Update the dynamic depending on the predicted feet position
        self.updateProblem(k, xref, l_feet, l_stop, velocity, acceleration)

        # Solve problem
        self.ddp.solve(self.x_init, self.u_init, self.max_iteration)

        # Reset to 0 the stopWeights for next optimisation
        for index_stopped in self.index_stop_optimisation :
            self.models_augmented[index_stopped].stopWeights = np.zeros(8)

        return 0

    def updateProblem(self, k, xref, l_feet, l_stop, velocity, acceleration):
        """Update the dynamic of the model list according to the predicted position of the feet,
        and the desired state.

        Args:
        """
        # Save previous gait state before updating the gait
        self.gait_old[0, :] = self.gait[0, :].copy()        

        # Recontruct the gait based on the computed footsteps
        a = 0
        while np.any(l_feet[a, :]):
            self.gait[a, :] = (l_feet[a, ::3] != 0.0).astype(int)
            a += 1
        self.gait[a:, :] = 0.0

        # On swing phase before --> initialised below shoulder
        p0 = (1.0 - np.repeat(self.gait[0, :], 2)) * self.p0
        # On the ground before -->  initialised with the current feet position
        p0 += np.repeat(self.gait[0, :], 2) * l_feet[0, [0, 1, 3, 4, 6, 7, 9, 10]]  # (x, y) of each foot       
        
        self.x_init.clear()
        self.u_init.clear()
        self.action_models.clear()
        self.index_stop_optimisation.clear()

        index_step = 0
        index_time = 0
        index_augmented = 0   
        j = 0   
        flying_foot = np.where( self.gait[0, :] == 0 )[0]
        nodes_flying_foot = []
        for foot_id in flying_foot :
            a = 0
            while self.gait[a,foot_id] == 0. :
                a += 1
            nodes_flying_foot.append(a)

        # Iterate over all phases of the gait
        while np.any(self.gait[j, :]):
            if j == 0 : # First row, need to take into account previous gait
                if np.any(self.gait[0,:] - self.gait_old[0,:]) : # No optimisation of the time, end of current flying phase here
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                                        velocity, acceleration,
                                                                        xref[:, j+1], self.gait[0, :] - self.gait_old[0, :])
                    # if first_step cost is activated, here no optimisation of the time before
                    self.models_step[index_step].speedWeight = 0.
                    self.models_step[index_step].first_step = False # Avoid computing the samples for cost of 0. weights
                    self.action_models.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'), l_stop,
                                                    xref[:, j+1], self.gait[j, :])                  

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time :
                        # print("l_stop j == 0 : " , l_stop)
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)
                    
                    self.action_models.append(self.models_augmented[index_augmented])

                    index_step += 1
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))   
                    self.u_init.append(np.zeros(4))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))      
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))
                
                else : # optimisation of the flying phase
                    # Step time model 
                    self.models_time[index_time].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                                        xref[:, j+1], self.gait[j, :])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'), l_stop,
                                                     xref[:, j+1], self.gait[j, :])
                    self.action_models.append(self.models_time[index_time])
                    self.action_models.append(self.models_augmented[index_augmented])
                    
                    index_augmented += 1
                    index_time += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))   
                    self.u_init.append(np.array([self.dt_init]) ) 
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))   
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))
            
            else : 
                if np.any(self.gait[j,:] - self.gait[j-1,:]) :
                    # Step model
                    self.models_step[index_step].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                       velocity, acceleration,
                                                       xref[:, j+1], self.gait[j, :] - self.gait[j-1, :])

                    # if first_step cost is activated, can use a different speedWeights ..etc
                    
                    # Check if the moving foot corresponds to the current flying foot to adapt the number of nodes 
                    # --> Modify period of flight, usefull for the cost
                    moving_foot = np.where( (self.gait[j,:] - self.gait[j-1,:]) == 1. )[0]
                    is_flying = False
                    nb_nodes = 0
                    for moving_id in moving_foot :
                        if moving_id in flying_foot :
                            is_flying = True
                            nb_nodes = nodes_flying_foot[int(np.where(flying_foot == moving_id)[0]) ]

                    if is_flying :
                        self.models_step[index_step].nb_nodes = nb_nodes
                        self.models_step[index_step].speedWeight = self.speedWeight # Maybe adapt with new gain
                        self.models_step[index_step].first_step = True
                    else :
                        self.models_step[index_step].nb_nodes = self.nb_nodes_nominal
                        self.models_step[index_step].speedWeight = self.speedWeight # Maybe adapt with new gain
                        self.models_step[index_step].first_step = False

                    self.action_models.append(self.models_step[index_step])

                    # Augmented model
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'), l_stop,
                                                     xref[:, j+1], self.gait[j, :])

                    # Activation of the cost to stop the optimisation around l_stop (position locked by the footstepGenerator)
                    if j < self.index_lock_time :
                        # print("l_stop j != 10 : " , l_stop)
                        self.models_augmented[index_augmented].stopWeights = self.stopWeights
                        self.index_stop_optimisation.append(index_augmented)
                    
                    self.action_models.append(self.models_augmented[index_augmented])

                    index_step += 1
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))   
                    self.u_init.append(np.zeros(8))
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))    
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))   

                    # End of a flying phase, starting of a new one 
                    if np.any(self.gait[j+1,:]) : # not the end of the gait matrix
                        self.models_time[index_time].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'),
                                                                        xref[:, j+1], self.gait[j, :])
                        self.action_models.append(self.models_time[index_time])
                        self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))   
                        self.u_init.append(np.array([self.dt_init]) ) 

                        index_time += 1
                                    
                
                else :
                    self.models_augmented[index_augmented].updateModel(np.reshape(l_feet[j, :], (3, 4), order='F'), l_stop,
                                                     xref[:, j+1], self.gait[j, :])
                    self.action_models.append(self.models_augmented[index_augmented])
                    
                    index_augmented += 1
                    # Warm-start
                    self.x_init.append(np.concatenate([xref[:, j+1], p0, np.array([self.dt_init])]))    
                    self.u_init.append(np.repeat(self.gait[j, :], 3) * np.array(4*[0., 0., 2.5*9.81/np.sum(self.gait[j, :])] ))

            # Update row matrix
            j += 1

        # Update terminal model 
        self.terminal_model.updateModel(np.reshape(l_feet[j-1, :], (3, 4), order='F'), l_stop,xref[:, -1], self.gait[j-1, :])
        # Warm-start
        self.x_init.append(np.concatenate([xref[:, j-1], p0, np.array([self.dt_init])]))   

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.action_models, self.terminal_model)

        self.problem.x0 = np.concatenate([xref[:, 0], p0, np.array([self.dt_init]) ])

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

        return 0

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

 