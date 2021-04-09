import numpy as np

class GaitPlanner:

    def __init__(self , T_gait , dt , T_mpc ):

        # Gait matrix
        self.N0_gait = 20
        self.gait = np.zeros((20, 5))
        self.desired_gait = np.zeros((20, 5))
        self.past_gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)       

        # Gait duration
        self.T_gait = T_gait
        self.T_mpc = T_mpc
        self.dt = dt        

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        # Feet matrix
        # [px1,py1,pz1 , px2 ....    px4,py4,pz4] x12
        self.o_feet_contact = self.shoulders.ravel(order='F').copy()

        self.remaining_time = 0.

        # Initialize matrix
        self.create_trot()
        self.create_gait_f()




    def roll(self, k  , fsteps):

        # Transfer current gait into past gait
        #self.past_gait[:,:] = self.gait[:,:]

        # Transfer current gait into past gait
        # If current gait is the same than the first line of past gait we just increment the counter
        if np.all(self.gait[0 , 1:] == self.past_gait[0,1:]) :
            self.past_gait[0,0] += 1
        else :
            tmp = self.past_gait[:-1,:]
            self.past_gait[1:,:] = tmp
            self.past_gait[0,:] = self.gait[0,:]
            self.past_gait[0,0] = 1

      
             
        # Age future gait
        if self.gait[0,0] == 1 :
            self.gait[:-1,:] = self.gait[1: , :]
            # Entering new contact phase, store positions of feet that are now in contact 
            if k != 0 :
                for i in range(4) :
                    if self.gait[0,i+1] == 1 :
                        self.o_feet_contact[(3*i):(3*(i+1))] = fsteps[1, (3*i+1):(3*(i+1)+1)]
        else :
            self.gait[0,0] -= 1 

        # Index of the first empty line
        i = 1
        while self.gait[i,0] > 0 :
            i += 1 

        # Increment last gait line or insert a new line
        if np.all(self.gait[i - 1 , 1:] == self.desired_gait[0,1:]) :
            self.gait[i - 1 , 0] += 1
        else :
            self.gait[i , :] = self.desired_gait[0 , :]
            self.gait[i , 0] = 1 


        # Age future desired gait
        # Index of the first empty line
        j = 1
        while self.desired_gait[j,0] > 0 :
            j += 1 

        #' Increment last gait line or insert a new line
        if np.all(self.desired_gait[0 , 1:] == self.desired_gait[j-1 , 1:]) :
            self.desired_gait[j-1,0] += 1 
        else :
            self.desired_gait[j,:] = self.desired_gait[0,:]
            self.desired_gait[j,0] = 1 
        
        if self.desired_gait[0,0 ] == 1 :
            self.desired_gait[:-1,:] = self.desired_gait[1:,:]
        else :
            self.desired_gait[0,0] -= 1

        return self.gait


    def getPhaseDuration(self , i , j , value): 

        t_phase = self.gait[i,0]
        a = int(i)

        # Looking for the end of the swing/stance phase in currentGait_
        while ( self.gait[i+1 , 0] > 0 and  self.gait[i+1 , 1 + j] == value  ) :
            i += 1
            t_phase += self.gait[i,0]

        # If we reach the end of currentGait_ we continue looking for the end of the swing/stance phase in desiredGait_
        if (self.gait[i + 1, 0] == 0.0) :
            k = 0
            while (self.desired_gait[k,0] > 0 and self.desired_gait[k,1+j] == value) :
                t_phase += self.desired_gait[k,0]
                k += 1
        #  We suppose that we found the end of the swing/stance phase either in currentGait_ or desiredGait_
        self.remaining_time = t_phase

        # Looking for the beginning of the swing/stance phase in currentGait_
        while a > 0 and self.gait[a-1 , 1 + j] == value :
            a -= 1
            t_phase += self.gait[a, 0]
        
        #  If we reach the end of currentGait_ we continue looking for the beginning of the swing/stance phase in pastGait
        if a == 0 :
            while self.past_gait[a,0] > 0 and self.past_gait[a, 1 + j] == value : 
                t_phase += self.past_gait[a,0]
                a +=1
        # We suppose that we found the beginning of the swing/stance phase either in currentGait_ or pastGait_

        return t_phase*self.dt # Take into account time step value

    def getRemainingTime(self) :

        return self.remaining_time
    
    def getCurrentGait(self) :

        return self.gait


    
    def create_static(self):
        """Create the matrices used to handle the gait and initialize them to keep the 4 feet in contact

        self.gait and self.fsteps matrices contains information about the gait
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        self.gait[0:4, 0] = np.array([2*N, 0, 0, 0])
        self.fsteps[0:4, 0] = self.gait[0:4, 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[0, 1:] = np.ones((4,))

        return 0

    def static_gait(self):
        """
        For a static gait (4 feet on the ground)
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([2*N, 0, 0, 0])
        new_desired_gait[0:4, 1:] = np.ones((4, 4))

        return new_desired_gait

    def create_walking_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a walking trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        i = 1
        self.gait[(4*i):(4*(i+1)), 0] = np.array([4, N-4, 4, N-4])
        self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[4*i+0, 1:] = np.ones((4,))
        self.gait[4*i+1, [1, 4]] = np.ones((2,))
        self.gait[4*i+2, 1:] = np.ones((4,))
        self.gait[4*i+3, [2, 3]] = np.ones((2,))

        return 0

    def create_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        self.desired_gait = np.zeros((20, 5))
        self.desired_gait[0:2, 0] = np.array([N, N])

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.desired_gait[0, [1, 4]] = np.ones((2,))
        self.desired_gait[1, [2, 3]] = np.ones((2,))
        
        return 0

    def one_swing_gait(self):
        """
        For a gait with only one leg in swing phase at a given time
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([N/2, N/2, N/2, N/2])
        new_desired_gait[0, 1] = 1
        new_desired_gait[1, 4] = 1
        new_desired_gait[2, 2] = 1
        new_desired_gait[3, 3] = 1

        return new_desired_gait

    def trot_gait(self):
        """
        For a walking trot gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:2, 0] = np.array([N, N])
        new_desired_gait[0, [1, 4]] = np.ones((2,))
        new_desired_gait[1, [2, 3]] = np.ones((2,))

        return new_desired_gait

    def pacing_gait(self):
        """
        For a pacing gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        new_desired_gait[0, 1:] = np.ones((4,))
        new_desired_gait[1, [1, 3]] = np.ones((2,))
        new_desired_gait[2, 1:] = np.ones((4,))
        new_desired_gait[3, [2, 4]] = np.ones((2,))

        return new_desired_gait

    def bounding_gait(self):
        """
        For a bounding gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:4, 0] = np.array([1, N-1, 1, N-1])
        new_desired_gait[0, 1:] = np.ones((4,))
        new_desired_gait[1, [1, 2]] = np.ones((2,))
        new_desired_gait[2, 1:] = np.ones((4,))
        new_desired_gait[3, [3, 4]] = np.ones((2,))

        return new_desired_gait

    def pronking_gait(self):
        """
        For a pronking gait
        Set stance and swing phases and their duration
        Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Gait matrix
        new_desired_gait = np.zeros((self.fsteps.shape[0], 5))
        new_desired_gait[0:2, 0] = np.array([N-1, N+1])
        new_desired_gait[0, 1:] = np.zeros((4,))
        new_desired_gait[1, 1:] = np.ones((4,))

        return new_desired_gait

    def create_custom(self):
        """Create the matrices used to handle the gait and initialize them to perform a trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_gait/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        i = 1
        self.gait[(4*i):(4*(i+1)), 0] = np.array([N/2, N/2, N/2, N/2])
        self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

        # Set stance and swing phases
        # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
        # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
        self.gait[4*i+0, [2, 3, 4]] = np.ones((3,))
        self.gait[4*i+1, [1, 3, 4]] = np.ones((3,))
        self.gait[4*i+2, [1, 2, 4]] = np.ones((3,))
        self.gait[4*i+3, [1, 2, 3]] = np.ones((3,))

        return 0

    def create_gait_f(self):

        sum_ = 0.
        offset = 0.
        i = 0
        j = 0

        # Fill future gait matrix
        while (sum_ < (self.T_mpc / self.dt)) :
            self.gait[j,:] = self.desired_gait[i,:]
            sum_ += self.desired_gait[i,0]
            offset += self.desired_gait[i,0]
            i += 1 
            j += 1
            if self.desired_gait[i,0] == 0 :
                i = 0
                offset = 0.0   # Loop back if T_mpc_ longer than gait duration

        # Remove excess time steps
        self.gait[j - 1, 0] -= sum_ - (self.T_mpc / self.dt)
        offset -= sum_ - (self.T_mpc / self.dt)


        # Age future desired gait to take into account what has been put in the future gait matrix
        j = 1
        while self.desired_gait[j, 0] > 0.0 :
            j += 1 
        
        k = 0
        while k < offset :
            k += 1

            if self.desired_gait[0 ,1:] == self.desired_gait[j-1 ,1:] : 
                self.desired_gait[j-1,0] += 1
            else :
                self.desired_gait[j,:] = self.desired_gait[0,:]
                self.desired_gait[j,0] = 1
                j += 1

            if  self.desired_gait[0 ,0] == 1 : 
                self.desired_gait[:-1,:] = self.desired_gait[1:,:]
                j -= 1
            else :
                self.desired_gait[j-1,0] -= 1
        return 0

                
        






        
