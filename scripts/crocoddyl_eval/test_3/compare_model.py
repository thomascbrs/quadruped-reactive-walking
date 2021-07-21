# coding: utf8

import numpy as np

import quadruped_walkgen as quadruped_walkgen
import libquadruped_reactive_walking as lqrw
import crocoddyl 

#####################
# Select MPC type   #
#####################
params = lqrw.Params()  # Object that holds all controller parameters

model_fsteps = quadruped_walkgen.ActionModelQuadrupedAugmented()
model_nl = quadruped_walkgen.ActionModelQuadrupedNonLinear()

# Model parameters
model_fsteps.dt = params.dt_mpc  
model_fsteps.mass =  2.50000279
model_fsteps.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])

model_fsteps.mu = 0.9
model_fsteps.min_fz = 0.2
model_fsteps.relative_forces = True
model_fsteps.shoulderContactWeight = 10.
model_fsteps.shoulder_hlim = 0.225

# Weights vectors
stateWeights =  np.zeros(12)
stateWeights[:6] = [0.3, 0.3, 2, 0.9, 1., 0.4]
stateWeights[6:] = [1.5, 2, 1, 0.05, 0.07, 0.05] * np.sqrt(stateWeights[:6])
model_fsteps.stateWeights = stateWeights
model_fsteps.stopWeights = np.zeros(8)
model_fsteps.frictionWeights = 0.5
model_fsteps.forceWeights = 0.01 * np.ones(12) 
model_fsteps.heuristicWeights = np.zeros(8)

###################

model_nl.dt = params.dt_mpc  
model_nl.mass = 2.50000279
model_nl.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                            [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                            [1.865287e-5, 1.245813e-4, 6.939757e-2]])
model_nl.mu = 0.9
model_nl.min_fz = 0.2
model_nl.max_fz = 60.
model_nl.relative_forces = True
model_nl.shoulderWeights = 10.
model_nl.shoulder_hlim = 0.225

# Weights vectors
model_nl.stateWeights = stateWeights
model_nl.frictionWeights = 0.5
model_nl.forceWeights = 0.01 * np.ones(12) 

# integration scheme
model_nl.implicit_integration =  False


# Update the dynamic of the model 

a = 1
b = -1 

x_fsteps = a + (b-a)*np.random.rand(20)
x_nl = x_fsteps[:12]
u = a + (b-a)*np.random.rand(12)



fstep = np.random.rand(12)
x_fsteps[12:] = fstep[[0,1,3,4,6,7,9,10]]
fstep = fstep.reshape((3,4), order = "F")
fstop = np.random.rand(12).reshape((3,4))
xref = np.random.rand(12)
gait = np.random.randint(2, size=4)
model_fsteps.updateModel(fstep, fstop, xref , gait)
model_nl.updateModel(fstep, xref , gait)

################################################
## CHECK DERIVATIVE WITH NUM_DIFF 
#################################################



# model_diff = crocoddyl.ActionModelNumDiff(model)
data_fsteps = model_fsteps.createData()
data_nl = model_nl.createData()

model_fsteps.calc(data_fsteps , x_fsteps, u)
model_nl.calc(data_nl , x_nl, u)

model_fsteps.calcDiff(data_fsteps , x_fsteps, u)
model_nl.calcDiff(data_nl , x_nl, u)



# # RUN CALC DIFF
# def run_calcDiff_numDiff(epsilon) :
#   Lx = 0
#   Lx_err = 0
#   Lu = 0
#   Lu_err = 0
#   Lxu = 0
#   Lxu_err = 0
#   Lxx = 0
#   Lxx_err = 0
#   Luu = 0
#   Luu_err = 0
#   Luu_noFri = 0
#   Luu_err_noFri = 0
#   Fx = 0
#   Fx_err = 0 
#   Fu = 0
#   Fu_err = 0    

#   for k in range(N_trial):    

#     x = a + (b-a)*np.random.rand(20)
#     u = a + (b-a)*np.random.rand(12)

#     fstep = np.random.rand(12).reshape((3,4))
#     fstop = np.random.rand(12).reshape((3,4))
#     xref = np.random.rand(12)
#     gait = np.random.randint(2, size=4)
#     model.updateModel(fstep, fstop, xref , gait)
#     model_diff = crocoddyl.ActionModelNumDiff(model)     
   
#     # Run calc & calcDiff function : numDiff     
#     model_diff.calc(data_diff , x , u )
#     model_diff.calcDiff(data_diff , x , u )
    
#     # Run calc & calcDiff function : c++ model
#     model.calc(data , x , u )
#     model.calcDiff(data , x , u )

#     Lx +=  np.sum( abs((data.Lx - data_diff.Lx )) >= epsilon  ) 
#     Lx_err += np.sum( abs((data.Lx - data_diff.Lx )) )  

#     Lu +=  np.sum( abs((data.Lu - data_diff.Lu )) >= epsilon  ) 
#     Lu_err += np.sum( abs((data.Lu - data_diff.Lu )) )  

#     Lxu +=  np.sum( abs((data.Lxu - data_diff.Lxu )) >= epsilon  ) 
#     Lxu_err += np.sum( abs((data.Lxu - data_diff.Lxu )) )  

#     Lxx +=  np.sum( abs((data.Lxx - data_diff.Lxx )) >= epsilon  ) 
#     Lxx_err += np.sum( abs((data.Lxx - data_diff.Lxx )) )  

#     Luu +=  np.sum( abs((data.Luu - data_diff.Luu )) >= epsilon  ) 
#     Luu_err += np.sum( abs((data.Luu - data_diff.Luu )) ) 

#     Fx +=  np.sum( abs((data.Fx - data_diff.Fx )) >= epsilon  ) 
#     Fx_err += np.sum( abs((data.Fx - data_diff.Fx )) )  

#     Fu +=  np.sum( abs((data.Fu - data_diff.Fu )) >= epsilon  ) 
#     Fu_err += np.sum( abs((data.Fu - data_diff.Fu )) )  
  