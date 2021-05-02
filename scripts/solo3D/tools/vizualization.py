import numpy as np
import pybullet as pyb


class PybVisualizationTraj():
    ''' Class used to vizualise the feet trajectory on the pybullet simulation 
    '''

    def __init__(self , footTrajectoryGenerator):
        
        # Bezier traj class to vizualize the traj
        self.footTrajectoryGenerator = footTrajectoryGenerator

        # Solo3D python class 
        self.gait = np.zeros((20, 5))
        self.fsteps = np.full((self.gait.shape[0], 13), np.nan)



    def vizuFootTraj(self , k ,fsteps , gait, device) :
        ''' Fonction to vizualize the trajectories of the feet in pyb
        Args : 
        fsteps 13xN : Matrice containing current and next feet position
        device : Pyb object
        '''

        if k > 1 :
            goals = self.footTrajectoryGenerator.getFootPosition()
            # for i_foot in range(4) :
            #     if self.gait[1,i_foot + 1] == 1 :
            #         ftps_Ids = device.pyb_sim.ftps_Ids_deb  
            #         # Display the goal position of the feet as green sphere in PyBullet
            #         pyb.resetBasePositionAndOrientation(ftps_Ids[i_foot],
            #                                             posObj=goals[:,i_foot],
            #                                             ornObj=np.array([0.0, 0.0, 0.0, 1.0]) )


            for i_foot in range(4) :
                if gait[1,i_foot + 1] == 1 :
                    ftps_Ids_deb = device.pyb_sim.ftps_Ids_deb
                    # Display the goal position of the feet as green sphere in PyBullet
                    goals = fsteps[1,3*i_foot +1 : 3*i_foot + 4]
                    pyb.resetBasePositionAndOrientation(ftps_Ids_deb[i_foot],
                                                        posObj=goals,
                                                        ornObj=np.array([0.0, 0.0, 0.0, 1.0]))


        return 0

