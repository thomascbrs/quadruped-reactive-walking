import numpy as np
import pybullet as pyb 



class LoggerPlanner:
    """Log the desired parameters of the planner class to analyses it
    """

    def __init__(self , dt , N_SIMULATION ):

        self.dt = dt 
        self.N_SIMULATION = int(N_SIMULATION)
        N = self.N_SIMULATION - 1

        self.feet_pos = np.zeros((N , 3 , 4)) # feet position from pybullet
        self.feet_vel = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_pos_des = np.zeros((N , 3 , 4)) # feet position from pybullet
        self.feet_vel_des = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_acc_des = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_pos_target = np.zeros((N, 3 , 4)) # feet position from pybullet

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])


    def log_feet(self, k , device , goals , vgoals, agoals, fsteps  ):
        """ Log feet 
                                            
        :param device: device interface to retrieve real feet pos/vel from pybullet
                goal : pos feet desired
                vgoal : vel feet desired
                vgoal : acc feet desired
                fsteps : end position foot desired                    
        """

        linkId = [3, 7 ,11 ,15]  # = [ FL , FR , HL , HR]

        if k == 0 :  # dummyDevice has no attriobute pyb_sym for 1st iteration
            self.feet_pos[k,:,:] = self.shoulders
            self.feet_vel[k,:,:] = np.zeros((3 , 4))

        else:
            links = pyb.getLinkStates(device.pyb_sim.robotId, linkId , computeForwardKinematics=True , computeLinkVelocity=True )

            for j in range(4) :
                self.feet_pos[k,:,j] = np.array(links[j][4])   # pos frame world for feet
                self.feet_vel[k,:,j] = np.array(links[j][6])   # vel frame world for feet

        self.feet_pos_des[k,:,:] = goals
        self.feet_vel_des[k,:,:] = vgoals
        self.feet_acc_des[k,:,:] = agoals

        return 0 
    
    def plot_feet(self):
        from matplotlib import pyplot as plt

        plt.figure()
        index = [1,2,3,4,5,6,7,8,9]
        lgd = ["pos x" , "pos y " , "pos z" , "vel x " , "vel y " , "vel z" , "acc x" , "acc y" , "acc z"]
        i_foot = 1
        

        

        time = np.linspace(0, self.dt*self.N_SIMULATION , self.N_SIMULATION - 1)

        for i in range(9):
            plt.subplot(3, 3, index[i])
            if i < 3 :
                plt.plot( self.feet_pos_des[: , i , i_foot] , 'x--' , color='r' )
                plt.plot( self.feet_pos[: , i , i_foot] , 'x--' , color='g')
                plt.legend(["pos des" , "pos pyb"])
                plt.ylabel(lgd[i])
            elif i < 6 :
                plt.plot( self.feet_vel_des[: , i - 3 , i_foot] , 'x--' , color = 'r')
                plt.plot( self.feet_vel[: , i - 3 , i_foot] , 'x--' , color = 'g')
                plt.legend(["vel des" , " vel pyb"])
                plt.ylabel(lgd[i])
            else :
                plt.plot(time , self.feet_acc_des[: , i - 6 , i_foot] , 'x--' )
                plt.legend("acc des")
                plt.ylabel(lgd[i])


        plt.suptitle("FR foot trajectory ")

        # Display graphs
        plt.show(block=True)


