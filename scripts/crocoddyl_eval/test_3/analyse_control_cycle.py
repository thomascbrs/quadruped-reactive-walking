# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
import libquadruped_reactive_walking as lqrw
import crocoddyl_class.MPC_crocoddyl_planner as MPC_crocoddyl_planner
import crocoddyl_class.MPC_crocoddyl as MPC_crocoddyl
import time 
import pinocchio as pin

##############
#  Parameters
##############
iteration_mpc = 560 # Control cycle
Relaunch_DDP = False # Compare a third MPC with != parameters
linear_mpc = True
params = lqrw.Params()  # Object that holds all controller parameters

######################
# Recover Logged data 
######################
file_name = "crocoddyl_eval/logs/data_2021_07_21_12_28.npz"
logs = np.load(file_name)
planner_gait = logs.get("planner_gait")
planner_xref = logs.get("planner_xref")
planner_fsteps = logs.get("planner_fsteps")
planner_goals = logs.get("planner_goals")
mpc_x_f = logs.get("mpc_x_f")
baseOrientation = logs.get("baseOrientation")
loop_o_q_int = logs.get("loop_o_q_int")

k = int( iteration_mpc * (params.dt_mpc / params.dt_wbc) ) # simulation iteration corresponding
k_previous = int( (iteration_mpc - 1) * (params.dt_mpc / params.dt_wbc) ) 

############
# OSQP MPC
###########
mpc_osqp = lqrw.MPC(params)
mpc_osqp.run(0, planner_xref[0] , planner_fsteps[0]) # Initialization of the matrix
# mpc_osqp.run(1, planner_xref[1] , planner_fsteps[1])
mpc_osqp.run(k, planner_xref[k] , planner_fsteps[k])

osqp_xs = mpc_osqp.get_latest_result()[:12,:] # States computed over the whole predicted horizon
osqp_xs = np.vstack([planner_xref[k,:,0] , osqp_xs.transpose()]).transpose() # Add current state 
osqp_us = mpc_osqp.get_latest_result()[12:,:] # Forces computed over the whole predicted horizon


########
# DDP lin
#######
mpc_ddp_lin = MPC_crocoddyl.MPC_crocoddyl(params, mu=0.9, linearModel=False)
# mpc_ddp_lin.shoulderWeights = 10.
# mpc_ddp_lin.max_fz = 50.
# Without warm-start :
# mpc_ddp.warm_start = False
# mpc_ddp.solve(k, planner_xref[k] , planner_fsteps[k] ) # Without warm-start

# Using warm-start logged :
#( Exactly the function solve from MPC_crocoddyl)
# But cannot be used because there areis no previous value inside ddp.xs and ddp.us
mpc_ddp_lin.updateProblem( planner_fsteps[k], planner_xref[k]) # Update dynamics

x_init = []
u_init = [] 
mpc_ddp_lin.ddp.solve(x_init,  u_init, mpc_ddp_lin.max_iteration)

k_previous = int( (iteration_mpc - 1) * (params.dt_mpc / params.dt_wbc) )

i = 0
while planner_gait[k,i,:].any() :
    i +=1

for j in range(mpc_x_f[0,:,:].shape[1] - 1 ) : 
    u_init.append(mpc_x_f[k_previous,12:24,j+1]) # Drag through an iteration (remove first)
u_init.append(np.repeat(planner_gait[k,i-1,:], 3)*np.array(4*[0.5, 0.5, 5.])) # Add last node with average values, depending on the gait

for j in range(mpc_x_f[0,:,:].shape[1] - 1 ) : 
    x_init.append(mpc_x_f[k_previous,:12,j+1]) # Drag through an iteration (remove first)
x_init.append(mpc_x_f[k_previous,:12,-1]) # repeat last term 
x_init.insert(0, planner_xref[k,:, 0]) # With ddp, first value is the initial value 

# Solve problem
# mpc_ddp_lin.ddp.solve(x_init,  u_init, mpc_ddp_lin.max_iteration)

ddp_xs_lin = mpc_ddp_lin.get_latest_result()[:12,:] # States computed over the whole predicted horizon 
ddp_xs_lin = np.vstack([planner_xref[k,:,0] , ddp_xs_lin.transpose()]).transpose() # Add current state 
ddp_us_lin = mpc_ddp_lin.get_latest_result()[12:,:] # Forces computed over the whole predicted horizon

###########
# DDP MPC 
##########
mpc_ddp = MPC_crocoddyl_planner.MPC_crocoddyl_planner(params, mu=0.9)

# Tune the weights 
# mpc_ddp.heuristicWeights = 100*np.array(4*[0.2, 0.6])
mpc_ddp.shoulderContactWeight = 10.
mpc_ddp.shoulder_hlim = 0.225
mpc_ddp.min_fz = 2.
# mpc_ddp.stepWeights = np.full(8, 0.005)
# mpc_ddp.stateWeights = np.sqrt([2.0, 2.0, 20.0, 0.25, 0.25, 10.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.3]) # fit osqp gains
# mpc_ddp.forceWeights = np.array(4*[0.02,0.02,0.02])
# mpc_ddp.stopWeights = np.zeros(8)
mpc_ddp.initialize_models(params) # re-initialize the model list with the new gains

mpc_ddp.gait = planner_gait[k_previous].copy() # gait_old will be initialised with that

rpy = pin.rpy.matrixToRpy(pin.Quaternion(loop_o_q_int[k][3:7]).toRotationMatrix())
rpy[0] = 0.
rpy[1] = 0.
oRh = pin.rpy.rpyToMatrix(rpy)
Rz = oRh
l_stop = oRh.transpose() @ planner_goals[k] 
for foot in range(4):
    l_stop[:,foot] += - loop_o_q_int[k,:3]

mpc_ddp.updateProblem(k , planner_xref[k] , planner_fsteps[k], l_stop)

mpc_ddp.ddp.solve(mpc_ddp.x_init,  mpc_ddp.u_init, mpc_ddp.max_iteration)

oTh = loop_o_q_int[k,:3].reshape((3,1))

ddp_xs = mpc_ddp.get_latest_result(oRh , oTh)[:12,:] # States computed over the whole predicted horizon 
ddp_xs = np.vstack([planner_xref[k,:,0] , ddp_xs.transpose()]).transpose() # Add current state 
ddp_us = mpc_ddp.get_latest_result(oRh , oTh)[12:,:] # Forces computed over the whole predicted horizon
ddp_fsteps = mpc_ddp.get_latest_result(oRh , oTh)[24:,:]
ddp_fsteps = np.vstack([planner_fsteps[k,0,:][[0,1,3,4,6,7,9,10]] , ddp_fsteps.transpose()]).transpose() # Add current state 


print("Gait : " , mpc_ddp.gait)
print("k : ", k)

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(0., params.T_gait, np.int(params.T_gait/params.dt_mpc)+1)
# l_str = ["X_osqp", "Y_osqp", "Z_osqp", "Roll_osqp", "Pitch_osqp", "Yaw_osqp", "Vx_osqp", "Vy_osqp", "Vz_osqp", "VRoll_osqp", "VPitch_osqp", "VYaw_osqp"]
l_str = ["X_lin", "Y_lin", "Z_lin", "Roll_lin", "Pitch_lin", "Yaw_lin", "Vx_lin", "Vy_lin", "Vz_lin", "VRoll_lin", "VPitch_lin", "VYaw_lin"]
l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    
    pl1, = plt.plot(l_t, ddp_xs[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, ddp_xs_lin[i,:], linewidth=2, marker='x')

    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, ddp_xs_relaunch[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2,pl3] , [l_str2[i] , l_str[i], "ddp_redo" ])
    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])      

# Desired evolution of contact forces
l_t = np.linspace(params.dt_mpc, params.T_gait, np.int(params.T_gait/params.dt_mpc))
# l_str = ["FL_X_osqp", "FL_Y_osqp", "FL_Z_osqp", "FR_X_osqp", "FR_Y_osqp", "FR_Z_osqp", "HL_X_osqp", "HL_Y_osqp", "HL_Z_osqp", "HR_X_osqp", "HR_Y_osqp", "HR_Z_osqp"]
l_str = ["FL_X_lin", "FL_Y_lin", "FL_Z_lin", "FR_X_lin", "FR_Y_lin", "FR_Z_lin", "HL_X_lin", "HL_Y_lin", "HL_Z_lin", "HR_X_lin", "HR_Y_lin", "HR_Z_lin"]
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    pl1, = plt.plot(l_t, ddp_us[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, ddp_us_lin[i,:], linewidth=2, marker='x')
   
    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, ddp_us_relaunch[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2,pl3] , [l_str2[i] , l_str[i], "ddp_redo" ])
    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])

plt.figure()
# CoM position predicted 
l_t = np.linspace(0., params.T_gait, np.int(params.T_gait/params.dt_mpc)+1)
for j in range(len(l_t)) : 
    if j == 6 : # middle sizie of the cross, for the legend
        pl1, = plt.plot(ddp_xs[0,j],  ddp_xs[1,j] ,color = "k" , marker='x', markersize= int(20/np.sqrt(j+3)) )
        pl2, = plt.plot(planner_xref[k,0,j], planner_xref[k,1,j] ,color = "g" , marker='x', markersize= int(20/np.sqrt(j+3)) )
    else : 
        plt.plot(ddp_xs[0,j],  ddp_xs[1,j] ,color = "k" , marker='x', markersize= int(20/np.sqrt(j+3)) )
        plt.plot(planner_xref[k,0,j], planner_xref[k,1,j] ,color = "g" , marker='x', markersize= int(20/np.sqrt(j+3)) )

plt.legend([pl1,pl2] , ["CoM ddp" , "CoM ref" ])

# First foot on the ground using previous gait matrix : iteration - 1
# for j in range(4) :       
#     if planner_gait[k_previous,0,j] == 1 : 
#         pl3, = plt.plot(planner_fsteps[k_previous,0 , 3*j] , planner_fsteps[k_previous,0 , 3*j] ,  'ro', markersize= 8   )
        # plt.plot(mpc_planner.Xs[12+2*i , 0  ] , mpc_planner.Xs[12+2*i + 1 , 0  ] ,  'yo', markerSize= 8   )

# Foostep computed by ddp and planner fsteps
for j in range(len(l_t)) : 
    if j >  0:
        ddp_fsteps_local = (Rz.transpose()[:2,:2] @ ddp_fsteps[:,j].reshape((2,4) , order = "F") -  loop_o_q_int[k][:2].reshape((2,1)) ).reshape((8,) , order = "F")
    else : 
        ddp_fsteps_local = ddp_fsteps[:,j] # 1st correspond to fstep innloca    
    for foot in range(4): 
        if j == 6 : # middle sizie of the cross, for the legend
            pl4, = plt.plot(ddp_fsteps_local[2*foot], ddp_fsteps_local[2*foot+ 1] ,color = "k" , marker='o', markersize= int(20/np.sqrt(j+3)),markerfacecolor='none' )
            pl5, =  plt.plot(planner_fsteps[k,j , 3*foot] , planner_fsteps[k,j , 3*foot+1],color = "r" ,  marker='x', markersize= int(20/np.sqrt(j+3))  ,markerfacecolor='none' )
        else : 
            plt.plot(ddp_fsteps_local[2*foot], ddp_fsteps_local[2*foot+ 1] ,color = "k" , marker='o', markersize= int(20/np.sqrt(j+3)),markerfacecolor='none' )
            plt.plot(planner_fsteps[k,j , 3*foot] , planner_fsteps[k,j , 3*foot+1],color = "r" ,  marker='x', markersize= int(20/np.sqrt(j+3))  ,markerfacecolor='none' )

plt.legend([pl1,pl2] , ["CoM ddp" , "CoM ref"])
plt.grid()
plt.show(block=True)



