import numpy as np 
import utils_mpc
import math

class FootStepPlanner:
    ''' Python class to compute the target foot step and the ref state. Equivalent to the actual c++ class.
    '''

    def __init__(self , dt , T_gait , h_ref, k_feedback , g , L , on_solo8 , k_mpc) :

        # Time step of the contact sequence
        self.dt = dt

        self.k_mpc = k_mpc

        # Gait duration
        self.T_gait = T_gait

        # Number of time steps in the prediction horizon
        self.n_steps = np.int(self.T_gait/self.dt)

        self.dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((12, 1 + self.n_steps))

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)
        self.is_static = False  # Flag for static gait
        self.q_static = np.zeros((19, 1))
        self.RPY_static = np.zeros((3, 1))
        self.RPY = np.zeros((3, 1))

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])
        
        # Feet matrix
        self.o_feet_contact = self.shoulders.ravel(order='F').copy()

        # To store the result of the compute_next_footstep function
        self.next_footstep = np.zeros((3, 4))

        self.h_ref = h_ref

        self.flag_rotation_command = int(0)

        # Predefined matrices for compute_footstep function
        self.R = np.zeros((3, 3, self.gait.shape[0]))
        self.R[2, 2, :] = 1.0

        self.b_v_cur = np.zeros((3,1))
        self.b_v_ref = np.zeros((3,1))

        self.k_feedback = k_feedback
        self.g = g
        self.L = L
        self.on_solo8 = on_solo8


        

    def getRefStates(self, q, v, vref, z_average ):
        """Compute the reference trajectory of the CoM for each time step of the
        predition horizon. The ouput is a matrix of size 12 by (N+1) with N the number
        of time steps in the gait cycle (T_gait/dt) and 12 the position, orientation,
        linear velocity and angular velocity vertically stacked. The first column contains
        the current state while the remaining N columns contains the desired future states.

        Args:
            T_gait (float): duration of one period of gait
            q (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            vref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """
        # Get the reference velocity in world frame (given in base frame)
        self.RPY = utils_mpc.quaternionToRPY(q[3:7, 0])

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # dt_vector = np.linspace(self.dt, self.T_gait, self.n_steps)
        # yaw = np.linspace(0, self.T_gait-self.dt, self.n_steps) * vref[5, 0]

        # Update yaw and yaw velocity
        # dt_vector = np.linspace(self.dt, T_gait, self.n_steps)
        self.xref[5, 1:] = vref[5, 0] * self.dt_vector
        self.xref[11, 1:] = vref[5, 0]

        # Update x and y velocities taking into account the rotation of the base over the prediction horizon
        # yaw = np.linspace(0, T_gait-self.dt, self.n_steps) * v_ref[5, 0]
        self.xref[6, 1:] = vref[0, 0] * np.cos(self.xref[5, 1:]) - vref[1, 0] * np.sin(self.xref[5, 1:])
        self.xref[7, 1:] = vref[0, 0] * np.sin(self.xref[5, 1:]) + vref[1, 0] * np.cos(self.xref[5, 1:])

        # Update x and y depending on x and y velocities (cumulative sum)
        if vref[5, 0] != 0:
            self.xref[0, 1:] = (vref[0, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) +
                                vref[1, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
            self.xref[1, 1:] = (vref[1, 0] * np.sin(vref[5, 0] * self.dt_vector[:]) -
                                vref[0, 0] * (np.cos(vref[5, 0] * self.dt_vector[:]) - 1)) / vref[5, 0]
        else:
            self.xref[0, 1:] = vref[0, 0] * self.dt_vector[:]
            self.xref[1, 1:] = vref[1, 0] * self.dt_vector[:]
        # self.xref[0, 1:] = self.dx  # dt_vector * self.xref[6, 1:]
        # self.xref[1, 1:] = self.dy  # dt_vector * self.xref[7, 1:]

        # Start from position of the CoM in local frame
        #self.xref[0, 1:] += q[0, 0]
        #self.xref[1, 1:] += q[1, 0]

        self.xref[5, 1:] += self.RPY[2, 0]

        # Desired height is supposed constant
        self.xref[2, 1:] = self.h_ref + z_average
        self.xref[8, 1:] = 0.0

        # No need to update Z velocity as the reference is always 0
        # No need to update roll and roll velocity as the reference is always 0 for those
        # No need to update pitch and pitch velocity as the reference is always 0 for those

        # Update the current state
        self.xref[0:3, 0:1] = q[0:3, 0:1]
        self.xref[3:6, 0:1] = self.RPY
        self.xref[6:9, 0:1] = v[0:3, 0:1]
        self.xref[9:12, 0:1] = v[3:6, 0:1]

        # Time steps [0, dt, 2*dt, ...]
        # to = np.linspace(0, self.T_gait-self.dt, self.n_steps)

        # Threshold for gamepad command (since even if you do not touch the joystick it's not 0.0)
        step = 0.05

        # Detect if command is above threshold
        if (np.abs(vref[2, 0]) > step) and (self.flag_rotation_command != 1):
            self.flag_rotation_command = 1

        """if True:  # If using joystick
            # State machine
            if (np.abs(vref[2, 0]) > step) and (self.flag_rotation_command == 1):  # Command with joystick
                self.h_rotation_command += vref[2, 0] * self.dt
                self.xref[2, 1:] = self.h_rotation_command
                self.xref[8, 1:] = vref[2, 0]

                self.flag_rotation_command = 1
            elif (np.abs(vref[2, 0]) < step) and (self.flag_rotation_command == 1):  # No command with joystick
                self.xref[8, 1:] = 0.0
                self.xref[9, 1:] = 0.0
                self.xref[10, 1:] = 0.0
                self.flag_rotation_command = 2
            elif self.flag_rotation_command == 0:  # Starting state of state machine
                self.xref[2, 1:] = self.h_ref
                self.xref[8, 1:] = 0.0

            if self.flag_rotation_command != 0:
                # Applying command to pitch and roll components
                self.xref[3, 1:] = self.xref[3, 0].copy() + vref[3, 0].copy() * to
                self.xref[4, 1:] = self.xref[4, 0].copy() + vref[4, 0].copy() * to
                self.xref[9, 1:] = vref[3, 0].copy()
                self.xref[10, 1:] = vref[4, 0].copy()
        else:
            self.xref[2, 1:] = self.h_ref
            self.xref[8, 1:] = 0.0"""

        # Current state vector of the robot
        # self.x0 = self.xref[:, 0:1]

        """self.xref[0:3, 1:2] = self.xref[0:3, 0:1] + self.xref[6:9, 0:1] * self.dt
        self.xref[3:6, 1:2] = self.xref[3:6, 0:1] + self.xref[9:12, 0:1] * self.dt"""

        self.xref[0, 1:] += self.xref[0, 0]
        self.xref[1, 1:] += self.xref[1, 0]

        if self.is_static:
            self.xref[0:3, 1:] = self.q_static[0:3, 0:1]
            self.xref[3:6, 1:] = (utils_mpc.quaternionToRPY(self.q_static[3:7, 0])).reshape((3, 1))

        """if v[0, 0] > 0.02:
            from IPython import embed
            embed()"""

        return self.xref
    
    
    
    def compute_footsteps(self, k ,q_cur, v_cur, v_ref, reduced  , gaitPlanner):
        """Compute the desired location of footsteps over the prediction horizon

        Compute a X by 13 matrix containing the remaining number of steps of each phase of the gait (first column)
        and the [x, y, z]^T desired position of each foot for each phase of the gait (12 other columns).
        For feet currently touching the ground the desired position is where they currently are.

        Args:
            q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
            v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
            v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """
        # Assume xref, RPY updated
        # Update intern gait matrix
        self.gait = gaitPlanner.getCurrentGait().astype(int) 

        self.fsteps[:, 0] = self.gait[:, 0]
        self.fsteps[:, 1:] = 0.

        i = 1

        rpt_gait = np.repeat(self.gait[:, 1:] == 1, 3, axis=1)
       

        # Set current position of feet for feet in stance phase
        (self.fsteps[0, 1:])[rpt_gait[0, :]] = (gaitPlanner.o_feet_contact)[rpt_gait[0, :]]
      

        # Get future desired position of footsteps
        self.compute_next_footstep(q_cur, v_cur, v_ref)

        """if reduced:  # Reduce size of support polygon
            self.next_footstep[0:2, :] -= np.array([[0.00, 0.00, -0.00, -0.00],
                                                    [0.04, -0.04, 0.04, -0.04]])"""

        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        dt_cum = np.cumsum(self.gait[:, 0]) * self.dt

        # Get future yaw angle compared to current position
        angle = v_ref[5, 0] * dt_cum + self.RPY[2, 0]
        c = np.cos(angle)
        s = np.sin(angle)
        self.R[0:2, 0:2, :] = np.array([[c, -s], [s, c]])

        # Displacement following the reference velocity compared to current position
        if v_ref[5, 0] != 0:
            dx = (v_cur[0, 0] * np.sin(v_ref[5, 0] * dt_cum) +
                  v_cur[1, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
            dy = (v_cur[1, 0] * np.sin(v_ref[5, 0] * dt_cum) -
                  v_cur[0, 0] * (np.cos(v_ref[5, 0] * dt_cum) - 1)) / v_ref[5, 0]
        else:
            dx = v_cur[0, 0] * dt_cum
            dy = v_cur[1, 0] * dt_cum

        """print(v_cur.ravel())
        print(v_ref.ravel())
        print(dt_cum.ravel())"""

        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        while (self.gait[i, 0] != 0):

            # Feet that were in stance phase and are still in stance phase do not move
            A = rpt_gait[i-1, :] & rpt_gait[i, :]
            if np.any(rpt_gait[i-1, :] & rpt_gait[i, :]):
                (self.fsteps[i, 1:])[A] = (self.fsteps[i-1, 1:])[A]

            # Feet that are in swing phase are NaN whether they were in stance phase previously or not
            # Commented as self.fsteps is already filled by np.nan by default
            """if np.any(rpt_gait[i, :] == False):
                (self.fsteps[i, 1:])[rpt_gait[i, :] == False] = np.nan * np.ones((12,))[rpt_gait[i, :] == False]"""

            # Feet that were in swing phase and are now in stance phase need to be updated
            A = np.logical_not(rpt_gait[i-1, :]) & rpt_gait[i, :]
            q_tmp = np.array([[q_cur[0, 0]], [q_cur[1, 0]], [0.0]])  # current position without height
            if np.any(A):
                # self.compute_next_footstep(i, q_cur, v_cur, v_ref)

                # Get desired position of footstep compared to current position
                next_ft = (np.dot(self.R[:, :, i-1], self.next_footstep) + q_tmp +
                           np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')
                # next_ft = (self.next_footstep + q_tmp + np.array([[dx[i-1]], [dy[i-1]], [0.0]])).ravel(order='F')

                # next_ft = (self.next_footstep).ravel(order='F')

                # Assignement only to feet that have been in swing phase
                (self.fsteps[i, 1:])[A] = next_ft[A]

            i += 1

        # print(self.fsteps[0:2, 2::3])
        return self.fsteps


    def compute_next_footstep(self, q_cur, v_cur, v_ref):
        """Compute the desired location of footsteps for a given pair of current/reference velocities

    Compute a 3 by 1 matrix containing the desired location of each feet considering the current velocity of the
    robot and the reference velocity

    Args:
        q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
        v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
        v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)
        """

        c, s = math.cos(self.RPY[2, 0]), math.sin(self.RPY[2, 0])
        R = np.array([[c, -s, 0], [s, c, 0], [0.0, 0.0, 0.0]])
        self.b_v_cur = R.transpose() @ v_cur[0:3, 0:1]
        self.b_v_ref = R.transpose() @ v_ref[0:3, 0:1]

        # TODO: Automatic detection of t_stance to handle arbitrary gaits
        t_stance = self.T_gait * 0.5
        # self.t_stance[:] = np.sum(self.gait[:, 0:1] * self.gait[:, 1:], axis=0) * self.dt
        """for j in range(4):
            i = 0
            t_stance = 0.0
            while self.gait[i, 1+j] == 1:
                t_stance += self.gait[i, 0]
                i += 1
            if i > 0:
                self.t_stance[j] = t_stance * self.dt"""

        # for j in range(4):
        """if self.gait[i, 1+j] == 1:
            t_stance = 0.0
            while self.gait[i, 1+j] == 1:
                t_stance += self.gait[i, 0]
                i += 1
            i_end = self.gait.shape[0] - 1
            while self.gait[i_end, 0] == 0:
                i_end -= 1
            if i_end == (i - 1):
                i = 0
                while self.gait[i, 1+j] == 1:
                    t_stance += self.gait[i, 0]
                    i += 1
            t_stance *= self.dt"""

        # Order of feet: FL, FR, HL, HR

        # self.next_footstep = np.zeros((3, 4))
        #print("Py computing ", j)

        # Add symmetry term
        self.next_footstep[0:2, :] = t_stance * 0.5 * self.b_v_cur[0:2, 0:1]  # + q_cur[0:2, 0:1]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Add feedback term
        self.next_footstep[0:2, :] += self.k_feedback * (self.b_v_cur[0:2, 0:1] - self.b_v_ref[0:2, 0:1])

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Add centrifugal term
        #print("b_v_cur: ", self.b_v_cur[0:3, 0].ravel())
        #print("v_ref: ", v_ref[3:6, 0])
        cross = self.cross3(np.array(self.b_v_cur[0:3, 0]), v_ref[3:6, 0])
        # cross = np.cross(v_cur[0:3, 0:1], v_ref[3:6, 0:1], 0, 0).T

        self.next_footstep[0:2, :] += 0.5 * math.sqrt(self.h_ref/self.g) * cross[0:2, 0:1]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        # Legs have a limited length so the deviation has to be limited
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) > self.L] = self.L
        (self.next_footstep[0:2, :])[(self.next_footstep[0:2, :]) < (-self.L)] = -self.L

        # solo8: no degree of freedom along Y for footsteps
        if self.on_solo8:
            # self.next_footstep[1, :] = 0.0
            # TODO: Adapt behaviour for world frame
            pass

        # Add shoulders
        self.next_footstep[0:2, :] += self.shoulders[0:2, :]

        #print(self.next_footstep[0:2, j:(j+1)].ravel())

        return 0
    
    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])
