import numpy as np

class FootTrajectoryGenerator:

    def __init__(self , T_gait , dt_tsid , k_mpc , fsteps_init):

        self.T_gait = T_gait
        self.dt_wbc = dt_tsid
        self.k_mpc = k_mpc

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        # Foot trajectory generator
        self.max_height_feet = 0.05
        self.t_lock_before_touchdown = 0.07

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)

        # Variables for foot trajectory generator
        self.i_end_gait = -1
        self.t_stance = np.zeros((4, ))  # Total duration of current stance phase for each foot
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot
        self.footsteps_target = (self.shoulders[0:3, :]).copy()
        self.goals = fsteps_init.copy()  # Store 3D target position for feet
        self.vgoals = np.zeros((3, 4))  # Store 3D target velocity for feet
        self.agoals = np.zeros((3, 4))  # Store 3D target acceleration for feet
        self.acc = np.zeros((3, 4))  # Storage variable for the trajectory generator
        self.vel = np.zeros((3, 4))  # Storage variable for the trajectory generator
        self.pos = self.shoulders  # Storage variable for the trajectory generator

        self.feet = []
        self.t0s = np.zeros((4, ))

        self.Ax = np.zeros((6,4))
        self.Ay = np.zeros((6,4))

        self.t_remaining = 0.


    def updateFootPosition(self, k , i_foot ):
        """Update the 3D desired position for feet in swing phase by using a 5-th order polynomial that lead them
        to the desired position on the ground (computed by the footstep planner)

        Args:
            k (int): number of time steps since the start of the simulation
            """

        if self.t0s[i_foot] == 0 or k == 0 :
            ddx0 = 0.
            ddy0 = 0.
            dx0 = 0.
            dy0 = 0.
            x0 = self.pos[0,i_foot]
            y0 = self.pos[1,i_foot]

        else :            
            ddx0 = self.acc[0,i_foot]
            ddy0 = self.acc[1,i_foot]
            dx0 = self.vel[0,i_foot]
            dy0 = self.vel[1,i_foot]
            x0 = self.pos[0,i_foot]
            y0 = self.pos[1,i_foot]

        t0 = self.t0s[i_foot]
        t1 = self.t_swing[i_foot]
        dt = self.dt_wbc

        x1 = self.footsteps_target[0,i_foot]
        y1 = self.footsteps_target[1,i_foot]

        h = self.max_height_feet

        

        if t0 < t1 - self.t_lock_before_touchdown :

            # compute polynoms coefficients for x and y
            self.Ax[5,i_foot] = (ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12 *
                             x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ax[4,i_foot]  = (30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 +
                             2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ax[3,i_foot]  = (t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0 *
                             t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ax[2,i_foot]  = -(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1 **
                                2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ax[1,i_foot]  = -(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2 *
                                    t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ax[0,i_foot]  = (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0 **
                                    2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

            self.Ay[5,i_foot] = (ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12 *
                            y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ay[4,i_foot]  = (30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 +
                               2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ay[3,i_foot]  = (t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0 *
                         t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ay[2,i_foot]  = -(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1 **
                          2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ay[1,i_foot]  = -(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2 *
                             t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            self.Ay[0,i_foot]  = (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0 **
                            2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

        # coefficients for z (deterministic)
        Az6 = -h/((t1/2)**3*(t1 - t1/2)**3)
        Az5 = (3*t1*h)/((t1/2)**3*(t1 - t1/2)**3)
        Az4 = -(3*t1**2*h)/((t1/2)**3*(t1 - t1/2)**3)
        Az3 = (t1**3*h)/((t1/2)**3*(t1 - t1/2)**3)

        # Get the next point
        ev = t0 + dt
        evz = ev

        if t0 < 0. or t0 > t1 :
            
            self.acc[0,i_foot] = 0.
            self.acc[1,i_foot] = 0.
            self.vel[0,i_foot] = 0.
            self.vel[1,i_foot] = 0.
            self.pos[0,i_foot] = x0
            self.pos[1,i_foot] = y0
        
        else : 
            self.acc[0,i_foot] = 2*self.Ax[2,i_foot] + 3*2*self.Ax[3,i_foot]*ev + 4*3*self.Ax[4,i_foot]*ev**2 + 5*4*self.Ax[5,i_foot]*ev**3
            self.acc[1,i_foot] = 2*self.Ay[2,i_foot] + 3*2*self.Ay[3,i_foot]*ev + 4*3*self.Ay[4,i_foot]*ev**2 + 5*4*self.Ay[5,i_foot]*ev**3
            self.vel[0,i_foot] = self.Ax[1,i_foot] + 2*self.Ax[2,i_foot]*ev + 3*self.Ax[3,i_foot]*ev**2 + 4*self.Ax[4,i_foot]*ev**3 + 5*self.Ax[5,i_foot]*ev**4
            self.vel[1,i_foot] = self.Ay[1,i_foot] + 2*self.Ay[2,i_foot]*ev + 3*self.Ay[3,i_foot]*ev**2 + 4*self.Ay[4,i_foot]*ev**3 + 5*self.Ay[5,i_foot]*ev**4
            self.pos[0,i_foot] = self.Ax[0,i_foot] + self.Ax[1,i_foot]*ev + self.Ax[2,i_foot]*ev**2 + self.Ax[3,i_foot]*ev**3 + self.Ax[4,i_foot]*ev**4 + self.Ax[5,i_foot]*ev**5
            self.pos[1,i_foot] = self.Ay[0,i_foot] + self.Ay[1,i_foot]*ev + self.Ay[2,i_foot]*ev**2 + self.Ay[3,i_foot]*ev**3 + self.Ay[4,i_foot]*ev**4 + self.Ay[5,i_foot]*ev**5
        
        self.acc[2,i_foot] = 2*3*Az3*evz + 3*4*Az4*evz**2 + 4*5*Az5*evz**3 + 5*6*Az6*evz**4
        self.vel[2,i_foot] =  3*Az3*evz**2 + 4*Az4*evz**3 + 5*Az5*evz**4 + 6*Az6*evz**5
        self.pos[2,i_foot] = Az3*evz**3 + Az4*evz**4 + Az5*evz**5 + Az6*evz**6
 

        # Store desired position, velocity and acceleration for later call to this function
        self.goals[:, i_foot] = self.pos[:,i_foot] # + np.array([0.0, 0.0, q[2, 0] - self.h_ref])
        self.vgoals[:, i_foot] = self.vel[:,i_foot]
        self.agoals[:, i_foot] = self.acc[:,i_foot]
          
        return 0


    def update_foot_trajectory(self, k , fsteps , gaitPlanner):

        self.gait = gaitPlanner.getCurrentGait() 
        

        # Update foot target 
        self.footsteps_target = np.zeros((3, 4))

        # for i in range(4):
        #     index = next((idx for idx, val in np.ndenumerate(
        #         fsteps[:, 3*i+1]) if ((not (val == 0)) and (not np.isnan(val)))), [-1])[0]
        #     self.footsteps_target[:, i] = fsteps[index, (1+i*3):(4+i*3)]
        #     self.footsteps_target[:, i] = fsteps[index, (1+i*3):(4+i*3)]

        for i in range(4) :
            index = 0
            while fsteps[index, 1 + 3*i] == 0. :
                index += 1 
            self.footsteps_target[:, i] = fsteps[index, (1+i*3):(4+i*3)]



        if (k % self.k_mpc) == 0 :
            self.feet = []
            self.t0s = np.zeros((4, ))

            # Indexes of feet in swing phase
            self.feet = np.where(self.gait[0, 1:] == 0)[0]
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0
            
            # For each foot in swing phase get remaining duration of the swing phase
            for i in self.feet :              
                self.t_swing[i] = gaitPlanner.getPhaseDuration(0,i,0.)   # 0. for swing phase   
                self.remainingTime = gaitPlanner.getRemainingTime()
                value =  self.t_swing[i] - (self.remainingTime * self.k_mpc - ((k+1)%self.k_mpc) )*self.dt_wbc - self.dt_wbc
                self.t0s[i] =   np.max( [value , 0.] ) 
                

        else :
            # If no foot in swing phase
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # Increment of one time step for feet in swing phase
            for i in self.feet :
                self.t0s[i] = np.max(  [self.t0s[i] + self.dt_wbc, 0.0])
               
        
        print("\n")
        print("t_swing : " , self.t_swing)
        print("t_remaining : " , self.remainingTime)
        print("t0s : " , self.t0s)
        print("\n")
        
        
        for i in self.feet :
            self.updateFootPosition(k,i)


        return self.goals , self.vgoals , self.agoals

    