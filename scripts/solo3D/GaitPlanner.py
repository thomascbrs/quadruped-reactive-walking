import numpy as np


class GaitPlanner:

    def __init__(self, T_gait, dt, T_mpc, N_gait=100):

        # Gait matrix
        self.current_gait = np.zeros((N_gait, 4))
        self.desired_gait = np.zeros((N_gait, 4))
        self.past_gait = np.zeros((N_gait, 4))
        # self.fsteps = np.full((N_gait, 12), np.nan)

        # Gait duration
        self.T_gait = T_gait
        self.T_mpc = T_mpc
        self.dt = dt
        self.n_steps = round(T_mpc / dt)

        self.newPhase = False

        # # Position of shoulders in local frame
        # self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
        #                            [0.14695, -0.14695, 0.14695, -0.14695],
        #                            [0.0, 0.0, 0.0, 0.0]])

        # # Feet matrix
        # # [px1,py1,pz1 , px2 ....    px4,py4,pz4] x12
        # self.o_feet_contact = self.shoulders.ravel(order='F').copy()

        self.remaining_time = 0.

        # Initialize matrix
        # self.create_walk()
        self.create_trot()
        self.create_gait_f()

    def updateGait(self, k, k_mpc):
        if k % k_mpc == 0:
            self.rollGait()
        return 0

    def rollGait(self):

        # Transfer current gait into past gait
        for m in range(self.n_steps, 0, -1):
            self.past_gait[[m, m-1]] = self.past_gait[[m - 1, m]]

        self.past_gait[0, :] = self.current_gait[0, :].copy()

        # Entering new contact phase, store positions of feet that are now in contact
        if (self.current_gait[0, :] - self.current_gait[1, :]).any():
            self.newPhase = True
        else:
            self.newPhase = False

        # Age current gait
        index = 1
        while self.current_gait[index, :].any():
            self.current_gait[[index - 1, index]] = self.current_gait[[index, index - 1]]
            index += 1

        # Insert a new line from desired gait into current gait
        self.current_gait[index-1, :] = self.desired_gait[0, :].copy()

        # Age desired gait
        index = 1
        while self.desired_gait[index, :].any():
            self.desired_gait[[index - 1, index]] = self.desired_gait[[index, index - 1]]
            index += 1

        return 0

    def getPhaseDuration(self, i,  j,  value):

        t_phase = 1
        i = int(i)
        a = i

        # / Looking for the end of the swing/stance phase in currentGait_
        while (self.current_gait[i+1, :].any() and self.current_gait[i+1, j] == value):
            i += 1
            t_phase += 1

        # If we reach the end of currentGait_ we continue looking for the end of the swing/stance phase in desiredGait_
        if (not self.current_gait[i+1, :].any()):
            k = 0
            while self.desired_gait[k, :].any() and self.desired_gait[k, j] == value:
                k += 1
                t_phase += 1

        # We suppose that we found the end of the swing/stance phase either in currentGait_ or desiredGait_
        self.remaining_time = t_phase

        # 'Looking for the beginning of the swing/stance phase in currentGait_
        while (a > 0 and self.current_gait[a-1, j] == value):
            a -= 1
            t_phase += 1

        # If we reach the end of currentGait_ we continue looking for the beginning of the swing/stance phase in pastGait_
        if a == 0:
            while self.past_gait[a, :].any() and self.past_gait[a, j] == value:
                a += 1
                t_phase += 1

        # We suppose that we found the beginning of the swing/stance phase either in currentGait_ or pastGait_

        return t_phase * self.dt   # Take into account time step value

    def getRemainingTime(self):

        return self.remaining_time

    def getCurrentGait(self):

        return self.current_gait

    def isNewPhase(self):
        return self.newPhase

    def create_walk(self):

        # Number of timesteps in 1/4th period of gait
        N = round(0.25 * self.T_gait / self.dt)

        sequence = np.array([0., 1., 1., 1.])
        self.desired_gait[:N, :] = sequence

        sequence = np.array([1., 0., 1., 1.])
        self.desired_gait[N:2*N, :] = sequence

        sequence = np.array([1., 1., 0., 1.])
        self.desired_gait[2*N:3*N, :] = sequence

        sequence = np.array([1., 1., 1., 0.])
        self.desired_gait[3*N:4*N, :] = sequence

        return 0

    def create_trot(self):

        # Number of timesteps in a half period of gait
        N = round(0.5 * self.T_gait / self.dt)

        sequence = np.array([1., 0., 0., 1.])
        self.desired_gait[:N, :] = sequence

        sequence = np.array([0., 1., 1., 0.])
        self.desired_gait[N:2*N, :] = sequence

        return 0

    def create_gait_f(self):

        i = 0

        # Fill currrent gait matrix
        for j in range(self.n_steps):
            self.current_gait[j, :] = self.desired_gait[i, :]
            i += 1
            if not self.desired_gait[i, :].any():
                i = 0  # Loop back if T_mpc_ longer than gait duration

        # Get index of first empty line
        index = 1
        while self.desired_gait[index, :].any():
            index += 1

        # Age desired gait to take into account what has been put in the current gait matrix
        for k in range(i):
            for m in range(index - 1):
                self.desired_gait[[m, m+1]] = self.desired_gait[[m + 1, m]]

        return 0
