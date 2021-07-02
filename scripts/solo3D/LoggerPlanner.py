import numpy as np
# import pybullet as pyb 
import pinocchio as pin
from example_robot_data import load



class LoggerPlanner:
    """Log the desired parameters of the planner class to analyses it
    """

    def __init__(self , dt , N_SIMULATION , T_gait  , k_mpc):

        self.dt = dt # dt mpc
        self.dt_wbc = 0.002
        self.N_SIMULATION = int(N_SIMULATION)
        N = self.N_SIMULATION 

        self.feet_pos = np.zeros((N , 3 , 4)) # feet position from pybullet
        self.feet_vel = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_pos_des = np.zeros((N , 3 , 4)) # feet position from pybullet
        self.feet_vel_des = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_acc_des = np.zeros((N, 3 , 4)) # feet velocity from pybullet
        self.feet_pos_target = np.zeros((N, 3 , 4)) # feet position from pybullet

        self.feet_pos_pio = np.zeros((N , 3 , 4)) # feet position from pybullet
        self.feet_vel_pio = np.zeros((N, 3 , 4)) # feet velocity from pybullet

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        
        # Number of time steps in the prediction horizon
        self.n_steps = np.int(T_gait/dt)
        self.k_mpc = k_mpc

        # Reference trajectory matrix of size 12 by (1 + N)  with the current state of
        # the robot in column 0 and the N steps of the prediction horizon in the others
        self.xref = np.zeros((int(N_SIMULATION/k_mpc) , 12, 1 + self.n_steps))
        

        # q_cur (7x1 array): current position vector of the flying base in world frame (linear and angular stacked)
        self.q_cur = np.zeros( (self.N_SIMULATION , 7 ))
        self.RPY = np.zeros( (self.N_SIMULATION , 3 ))

        # v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
        self.v_cur = np.zeros( (self.N_SIMULATION , 6 ))
        self.b_v_cur = np.zeros( (self.N_SIMULATION , 6 ))


        # v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)        
        self.vref = np.zeros( (self.N_SIMULATION , 6 ))  # From joystick
        self.b_vref = np.zeros( (self.N_SIMULATION , 6 ))
        

        # Model Predictive Control
        # output vector of the MPC (next state + reference contact force)
        self.mpc_x_f = np.zeros([int(N_SIMULATION/k_mpc), 24, self.n_steps])   

        
        self.footsteps_target = np.zeros( (self.N_SIMULATION , 3,4)  )


        # Load the URDF model to get Pinocchio data and model structures
        robot = load('solo12')
        self.data = robot.data.copy()  # for velocity estimation (forward kinematics)
        self.model = robot.model.copy()  # for velocity estimation (forward kinematics)


        # Load timings
        self.timings_qp = np.zeros( (self.N_SIMULATION , 4 ))
        self.timings_bezier = np.zeros( (self.N_SIMULATION , 4 ))
        
     


    def log_feet(self, k , device , goals , vgoals, agoals, targetFootstep , q_filt ,  v_filt ):
        """ Log feet 
                                            
        :param device: device interface to retrieve real feet pos/vel from pybullet
                goal : pos feet desired
                vgoal : vel feet desired
                vgoal : acc feet desired
                fsteps : end position foot desired                    
        """
        contactFrameId = [ 10 , 18 , 26 , 34 ]   # = [ FL , FR , HL , HR]

        pin.forwardKinematics(self.model, self.data, q_filt, v_filt)

        for i_foot in range(4) : 
            framePlacement = pin.updateFramePlacement( self.model, self.data, contactFrameId[i_foot])    # = solo.data.oMf[18].translation
            frameVelocity = pin.getFrameVelocity(self.model, self.data, contactFrameId[i_foot], pin.ReferenceFrame.LOCAL)
            
            self.feet_pos_pio[k,:,i_foot] = framePlacement.translation
            self.feet_vel_pio[k,:,i_foot] = frameVelocity.linear

        linkId = [3, 7 ,11 ,15]  # = [ FL , FR , HL , HR]

        if k == 0 :  # dummyDevice has no attriobute pyb_sym for 1st iteration
            self.feet_pos[k,:,:] = self.shoulders
            self.feet_vel[k,:,:] = np.zeros((3 , 4))

        else:
            links = pyb.getLinkStates(device.pyb_sim.robotId, linkId , computeForwardKinematics=True , computeLinkVelocity=True )

            for j in range(4) :
                self.feet_pos[k,:,j] = np.array(links[j][4])   # pos frame world for feet
                self.feet_pos[k,2,j] -= 0.01702
                self.feet_vel[k,:,j] = np.array(links[j][6])   # vel frame world for feet

        self.feet_pos_des[k,:,:] = goals
        self.feet_vel_des[k,:,:] = vgoals
        self.feet_acc_des[k,:,:] = agoals

        # Update foot target 
        self.footsteps_target[k,:, :] = targetFootstep

        return 0 

    def log_state(self , k , q, v, v_ref , b_vref , RPY ,  xref ) :
        
        if (k % self.k_mpc) == 0 : 
            self.xref[int(k/self.k_mpc), : , :] = xref
        
        self.q_cur[k,:] = q[:,0]
        self.RPY[k,:] = RPY[:,0]

        # v_cur (6x1 array): current velocity vector of the flying base in world frame (linear and angular stacked)
        self.v_cur[k,:] = v[:,0]
        # self.b_v_cur[k,:] = b_vref

        # v_ref (6x1 array): desired velocity vector of the flying base in world frame (linear and angular stacked)        
        self.vref[k,:] = v_ref[:,0]
        self.b_vref[k,:] = b_vref[:,0]
        
        return 0
    
    def log_timing(self ,k, timings_qp , timings_bezier) :
        """ Log the timings of the FoostepPlanner and curve planner
        Args :
        - timings_qp (array) : Array containing the results for [whole loop , convert_inequalities , res_qp , update new fstep ]
        - timings_bezier (array) : Array containing the results for [ ]
        """

        self.timings_qp[k,:] = timings_qp
        self.timings_qp[k,:] = timings_qp

        return 0


    def log_mpc(self, k , x_f_mpc) :
        
        if (k % self.k_mpc) == 0 : 
            self.mpc_x_f[int(k/self.k_mpc), : , :] = x_f_mpc

        return 0

    def plot_log_planner(self) :
        ''' Call the function to plot
        '''
        from matplotlib import pyplot as plt

        # self.plot_ref_state()
        # self.plot_MPC_pred()
        # self.plot_feet()
        self.plot_timings()


        # Display graphs
        plt.show(block=True)

        return 0
    
    def plot_feet(self):
        from matplotlib import pyplot as plt

        plt.figure()
        index = [1,2,3,4,5,6,7,8,9]
        lgd = ["pos x" , "pos y " , "pos z" , "vel x " , "vel y " , "vel z" , "acc x" , "acc y" , "acc z"]
        i_foot = 0     
        
        time = np.linspace(0, self.dt*self.N_SIMULATION , self.N_SIMULATION )

        for i in range(9):
            plt.subplot(3, 3, index[i])
            if i < 3 :
                plt.plot( self.feet_pos_des[: , i , i_foot] , 'x--' , color='r' )
                plt.plot( self.feet_pos[: , i , i_foot] , 'x--' , color='g')
                plt.plot( self.feet_pos_pio[: , i , i_foot] , 'x--' , color='k')
                plt.plot( self.footsteps_target[: , i , i_foot] , 'x--' , color='b')
                plt.legend(["pos des" ,  "pos pyb" , "pio pos" , "target"])
                plt.ylabel(lgd[i])
            elif i < 6 :
                plt.plot( self.feet_vel_des[: , i - 3 , i_foot] , 'x--' , color = 'r')
                plt.plot( self.feet_vel[: , i - 3 , i_foot] , 'x--' , color = 'g')
                plt.plot( self.feet_vel_pio[: , i - 3 , i_foot] , 'x--' , color='k')
                plt.legend(["vel des" , " vel pyb" , "vel pio"])
                plt.ylabel(lgd[i])
            else :
                plt.plot(time , self.feet_acc_des[: , i - 6 , i_foot] , 'x--' )
                plt.legend("acc des")
                plt.ylabel(lgd[i])

        plt.suptitle("FR foot trajectory ")
        
        return 0

    def plot_ref_state(self):
        from matplotlib import pyplot as plt
        
        plt.figure() 
        index = [1,5,9,2,6,10,3,7,11,4,8,12]
        lgd = [" x" , " y " , " z" , "roll " , "pitch" , "yaw" , "vx" , "vy" , "vz" , "vroll" , "vpitch" , "vyae"]

        for i in range(12) :
            plt.subplot(3,4,index[i])
            if i < 3 : # plot x,y,z ref
                plt.plot(self.xref[:,i,0] , "x--" , color = 'b')
                plt.plot(self.xref[:,i,1] , "x--" , color = 'r')
                plt.legend(["cur" , "ref"])
                plt.ylabel(lgd[i])
            if i >= 3 and i < 6 : # plot x,y,z ref
                plt.plot(self.xref[:,i,0] , "x--" , color = 'b')
                plt.plot(self.xref[:,i,1] , "x--" , color = 'r')
                plt.ylabel(lgd[i])
                plt.legend(["cur" , "ref"])
            if i >= 6 and i < 9 : # plot x,y,z ref
                plt.plot(self.xref[:,i,0] , "x--" , color = 'b')
                plt.plot(self.xref[:,i,1] , "x--" , color = 'r')
                plt.ylabel(lgd[i])
                plt.legend(["cur" , "ref"])
            if i >= 9 and i < 12 : # plot x,y,z ref
                plt.plot(self.xref[:,i,0] , "x--" , color = 'b')
                plt.plot(self.xref[:,i,1] , "x--" , color = 'r')
                plt.ylabel(lgd[i])
                plt.legend(["cur" , "ref"])    

        plt.suptitle("REF band cur traj")

        return 0

    def plot_MPC_pred(self) : 
        from matplotlib import pyplot as plt

        titles = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        index = [1,3,5,2,4,6]
       

        # Evolution of predicted trajectory along time
        log_t_pred = np.array([k*self.dt for k in range(self.mpc_x_f.shape[0])])
        log_t_current = np.array([k*self.dt_wbc for k in range(self.q_cur.shape[0])])

        plt.figure()

        for j in range(6):
            plt.subplot(3, 2, index[j])
        
            h1, = plt.plot(log_t_pred, self.mpc_x_f[:, j, 0], "b", linewidth=2, color='k')

            h2, = plt.plot(log_t_pred, self.xref[:, j, 1], linestyle="--", marker='x', color="g", linewidth=2)

            if j < 3 :
                h4, = plt.plot(log_t_current , self.q_cur[:, j], linestyle=None, marker='x', color="r", linewidth=1)
            else : 
                h4, = plt.plot(log_t_current , self.RPY[:, j-3], linestyle=None, marker='x', color="r", linewidth=1)

            h3, = plt.plot(log_t_pred , self.xref[:, j, 0], linestyle=None, marker='o', color="r", linewidth=1)
                
            
            plt.xlabel("Time [s]")
            plt.legend([h1, h2, h3, h4], ["Output trajectory of MPC",
                                      "Input trajectory of planner", "Actual robot trajectory [xref]" , "qcur"])
            plt.title("Predicted trajectory for " + titles[j])
        plt.suptitle("Analysis of trajectories in position and orientation computed by the MPC")

        plt.figure()

        for j in range(6):
            plt.subplot(3, 2, index[j])
        
            h1, = plt.plot(log_t_pred, self.mpc_x_f[:, j+6, 0], "b", linewidth=2, color='k')

            h2, = plt.plot(log_t_pred, self.xref[:, j+6, 1], linestyle="--", marker='x', color="g", linewidth=2)


            h4, = plt.plot(log_t_current , self.v_cur[:, j], linestyle=None, marker='x', color="r", linewidth=1)
            

            h3, = plt.plot(log_t_pred , self.xref[:, j+6, 0], linestyle=None, marker='o', color="r", linewidth=1)

            plt.xlabel("Time [s]")
            plt.legend([h1, h2, h3,h4], ["Output trajectory of MPC",
                                      "Input trajectory of planner", "Actual robot trajectory[ref]" , "Actual vcur"])
            plt.title("Predicted trajectory for velocity in " + titles[j])
        plt.suptitle("Analysis of trajectories of linear and angular velocities computed by the MPC")



        return 0
    
    def plot_timings(self) :


        from matplotlib import pyplot as plt

        plt.figure()
        h1, = plt.plot( self.timings_qp[:,0], "x", linewidth=2, color='k')

        h2, = plt.plot( self.timings_qp[:,1], "x", linewidth=2, color='b')

        h3, = plt.plot( self.timings_qp[:,2], "x", linewidth=2, color='r')

        h4, = plt.plot( self.timings_qp[:,3], "x", linewidth=2, color='g')


        plt.legend([h1, h2, h3,h4], ["Whole QP loop",
                                      "convert_pb_to_ineq", "res qp" , "update fsteps"])


        return 0


