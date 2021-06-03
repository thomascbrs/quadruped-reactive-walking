import numpy as np
import pinocchio as pin
import utils_mpc
import math
from solo3D.tools.optimisation import genCost, quadprog_solve_qp, to_least_square


class FootStepPlannerQP:
    ''' Python class to compute the target foot step and the ref state using QP optimisation 
    to avoid edges of surface.
    '''

    def __init__(self, dt, dt_wbc, T_gait, h_ref, k_mpc, gait, N_gait, heightMap):

        # Time step of the contact sequence
        self.dt = dt  # dt mpc
        self.dt_wbc = dt_wbc

        self.k_mpc = k_mpc

        # Gait duration
        self.T_gait = T_gait

        # Position of shoulders in local frame
        self.shoulders = np.array([[0.1946, 0.1946, -0.1946, -0.1946],
                                   [0.14695, -0.14695, 0.14695, -0.14695],
                                   [0.0, 0.0, 0.0, 0.0]])

        self.h_ref = h_ref
        self.k_feedback = 0.03
        self.g = 9.81
        self.L = 0.155

        # heightMap
        self.heightMap = heightMap

        # Coefficients QP
        self.weight_vref = 0.06
        self.weight_alpha = 1.

        # Store surface selected for the current swing phase
        self.surface_selected = [None, None, None, None]
        self.feet = []
        self.t0s = np.zeros((4, ))
        self.t_remaining = 0.
        self.t_stance = np.zeros((4, ))  # Total duration of current stance phase for each foot
        self.t_swing = np.zeros((4, ))  # Total duration of current swing phase for each foot

        # gaitPLanner
        self.gait = gait

        self.N_gait = N_gait
        self.dt_cum = np.zeros(N_gait)
        self.yaws = np.zeros(N_gait)
        self.dx = np.zeros(N_gait)
        self.dy = np.zeros(N_gait)

        # Roll / Pitch / Yaw array
        self.RPY = np.zeros(3)
        self.RPY_b = np.zeros(3)  # in base frame (only yaw)

        # List of the fsteps
        self.footsteps = []
        for i in range(N_gait):
            self.footsteps.append(np.zeros((3, 4)))

        self.nextFootstep = np.zeros((3, 4))

        self.current_fstep = self.shoulders.copy()
        self.targetFootstep = self.shoulders.copy()
        self.fsteps = np.zeros((self.N_gait, 12))

    def run_optimisation(self, L, q, b_vlin, b_vref, o_vref):
        ''' Update ftseps with the optimised positions. 
        L is an array containing information on the feet positions to optimise.
        the surface associated and the nb of inequalities in the surface (usefull to determined the size of the inequalities matrix)
        Arg : 
        - L (nb_feetx4) : [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  
                          --> First variable in gait[1,0]  
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here x,y,z,w quaternion
        - b_vlin (3x) : current v linear in base frame 
        - b_vref (6x) : array, vref i
        - o_vref (6x1) : o_vref in world frame
        '''

        ineqMatrix = np.zeros((int(np.sum(L[:, 3])), 3*L.shape[0] + 2))
        ineqVector = np.zeros(int(np.sum(L[:, 3])))

        eqMatrix = np.zeros((L.shape[0], 3*L.shape[0] + 2))
        eqVector = np.zeros(L.shape[0])

        count = 0
        count_eq = 0

        # current position without height
        RPY_tmp = np.zeros(3)
        q_tmp = q[:3, 0].copy()
        q_tmp[2] = 0.

        for indice_L, [i, j, indice_s, nb_ineq] in enumerate(L):

            if indice_s == -99:  # z = 0, ground
                eqMatrix[count_eq, 3*indice_L + 2 + 2] = 1.
            else:
                # Offset to the future position
                q_dxdy = np.array([self.dx[i - 1], self.dy[i - 1], 0.0])

                # Get future desired position of footsteps without k_feedback
                nextFootstep = self.computeNextFootstep(i, j, b_vlin, b_vref, False)

                # Get desired position of footstep compared to current position
                RPY_tmp[2] = self.yaws[i-1]
                Rz = pin.rpy.rpyToMatrix(RPY_tmp)
                next_ft = Rz.dot(nextFootstep) + q_tmp + q_dxdy

                surface = self.heightMap.Surfaces[indice_s]

                # S * [Vrefx , Vrefy]
                ineqMatrix[count:count + nb_ineq, :2] = -self.k_feedback*surface.ineq_inner[:-1, :2]

                # S * [alphax  , aplhay  Pz]
                ineqMatrix[count:count+nb_ineq, 3*indice_L + 2: 3*indice_L + 2 + 2] = surface.ineq_inner[:-1, :-1]
                ineqMatrix[count:count+nb_ineq, 3*indice_L + 2 + 2] = surface.ineq_inner[:-1, -1]

                ineqVector[count:count+nb_ineq] = surface.ineq_vect_inner[:-1] - \
                    np.dot(surface.ineq_inner[:-1, :2], next_ft[:2])

                # S * [Vrefx , Vrefy]
                eqMatrix[count_eq, :2] = -self.k_feedback*surface.ineq_inner[-1, :2]

                # S * [alphax  , aplhay  Pz]
                eqMatrix[count_eq, 3*indice_L + 2: 3*indice_L + 2 + 2] = surface.ineq_inner[-1, :-1]
                eqMatrix[count_eq, 3*indice_L + 2 + 2] = surface.ineq_inner[-1, -1]

                eqVector[count_eq] = surface.ineq_vect_inner[-1] - np.dot(surface.ineq_inner[-1, :2], next_ft[:2])

            count_eq += 1
            count += nb_ineq

        P = np.identity(2 + L.shape[0]*3)
        q = np.zeros(2 + L.shape[0]*3)

        P[:2, :2] = self.weight_vref*np.identity(2)
        q[:2] = -self.weight_vref*np.array([o_vref[0, 0], o_vref[1, 0]])
        res = quadprog_solve_qp(P, q,  G=ineqMatrix, h=ineqVector, C=eqMatrix, d=eqVector)

        # Store the value for getRefState
        self.o_vref_qp = o_vref.copy()
        self.o_vref_qp[0, 0] = res[0]
        self.o_vref_qp[1, 0] = res[1]

        # Get new reference velocity in base frame to recompute the new footsteps

        Rz = pin.rpy.rpyToMatrix(self.RPY_b)
        b_vref_qp = np.zeros(6)
        b_vref_qp[:3] = Rz.transpose().dot(self.o_vref_qp[:3, 0])
        b_vref_qp[3:] = Rz.transpose().dot(self.o_vref_qp[3:, 0])

        for indice_L, [i, j, indice_s, nb_ineq] in enumerate(L):

            # Offset to the future position
            q_dxdy = np.array([self.dx[i - 1], self.dy[i - 1], 0.0])

            # Get future desired position of footsteps without k_feedback
            nextFootstep = self.computeNextFootstep(i, j, b_vlin, o_vref[:, 0], True)

            # Get desired position of footstep compared to current position
            RPY_tmp[2] = self.yaws[i-1]
            Rz = pin.rpy.rpyToMatrix(RPY_tmp)
            next_ft = Rz.dot(nextFootstep) + q_tmp + q_dxdy

            next_ft += np.array([res[2+3*indice_L], res[2+3*indice_L+1], res[2+3*indice_L+2]])
            # next_ft += np.array([0., 0. , res[2+3*indice_L+2] ])

            self.footsteps[i][:, j] = next_ft

        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        i = 1
        gait = self.gait.getCurrentGait()
        while (gait[i, :].any()):
            # Feet that were in stance phase and are still in stance phase do not move
            for j in range(4):
                if gait[i-1, j]*gait[i, j] > 0:
                    self.footsteps[i][:, j] = self.footsteps[i-1][:, j]
            i += 1

        return 0

    def update_remaining_time(self):

        if (self.k % self.k_mpc) == 0:
            self.feet = []
            self.t0s = np.zeros((4, ))

            # Indexes of feet in swing phase
            self.feet = np.where(self.gait.getCurrentGait()[0, :] == 0)[0]
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # For each foot in swing phase get remaining duration of the swing phase
            for i in self.feet:
                self.t_swing[i] = self.gait.getPhaseDuration(0, int(i), 0.)   # 0. for swing phase
                self.remainingTime = self.gait.getRemainingTime()
                value = self.t_swing[i] - (self.remainingTime * self.k_mpc - ((self.k+1) %
                                                                              self.k_mpc))*self.dt_wbc - self.dt_wbc
                self.t0s[i] = np.max([value, 0.])

        else:
            # If no foot in swing phase
            if len(self.feet) == 0:  # If no foot in swing phase
                return 0

            # Increment of one time step for feet in swing phase
            for i in self.feet:
                self.t0s[i] = np.max([self.t0s[i] + self.dt_wbc, 0.0])

        return 0

    def computeFootsteps(self, q, v, o_vref):
        ''' 
        Params : 
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here x,y,z,w quaternion
        - v (6x1) : v in world frame (lin + ang)
        - o_vref (6x1) : o_vref in world frame
        '''
        self.footsteps.clear()

        for i in range(self.N_gait):
            self.footsteps.append(np.zeros((3, 4)))

        gait = self.gait.getCurrentGait()

        # Set current position of feet for feet in stance phase
        for j in range(4):
            if gait[0, j] == 1.:
                self.footsteps[0][:, j] = self.current_fstep[:, j]

        # Cumulative time by adding the terms in the first column (remaining number of timesteps)
        # Get future yaw yaws compared to current position
        self.dt_cum[0] = self.dt  # dt mpc
        self.yaws[0] = o_vref[5, 0] * self.dt_cum[0] + self.RPY_b[2]
        for j in range(1, len(self.footsteps)):
            if gait[j, :].any():
                self.dt_cum[j] = self.dt_cum[j-1] + self.dt
            else:
                self.dt_cum[j] = self.dt_cum[j-1]

            self.yaws[j] = o_vref[5, 0]*self.dt_cum[j] + self.RPY_b[2]

        # Displacement following the reference velocity compared to current position
        if o_vref[5, 0] != 0:
            for j in range(1, len(self.footsteps)):
                self.dx[j] = (v[0] * np.sin(o_vref[5, 0] * self.dt_cum[j]) + v[1] *
                              (np.cos(o_vref[5, 0] * self.dt_cum[j]) - 1.)) / o_vref[5, 0]
                self.dy[j] = (v[1] * np.sin(o_vref[5, 0] * self.dt_cum[j]) - v[0] *
                              (np.cos(o_vref[5, 0] * self.dt_cum[j]) - 1.)) / o_vref[5, 0]
        else:
            for j in range(1, len(self.footsteps)):
                self.dx[j] = v[0]*self.dt_cum[j]
                self.dy[j] = v[1]*self.dt_cum[j]

        # Get current and reference velocities in base frame (rotated yaw)
        Rz = pin.rpy.rpyToMatrix(self.RPY_b)
        b_vlin = Rz.transpose().dot(v[:3, 0])   # linear velocity in base frame, array
        b_vref = np.zeros(6)
        b_vref[:3] = Rz.transpose().dot(o_vref[:3, 0])
        b_vref[3:] = Rz.transpose().dot(o_vref[3:, 0])

        RPY_tmp = np.zeros(3)

        # Update remaining time flying phase
        self.update_remaining_time()

        # L contains the indice of the moving feet
        # L = [  [x =1,y=0  ,  surface_id  , nb ineq in surface]  ]  --> First variable in gait[1,0]
        # surace_id = -99 if no surfaces
        L = []

        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        i = 1
        while (gait[i, :].any()):
            # Feet that were in stance phase and are still in stance phase do not move
            for j in range(4):
                if gait[i-1, j]*gait[i, j] > 0:
                    self.footsteps[i][:, j] = self.footsteps[i-1][:, j]

            # Current position without height
            q_tmp = q[:3, 0].copy()
            q_tmp[2] = 0.

            # Feet that were in swing phase and are now in stance phase need to be updated
            for j in range(4):
                if (1 - gait[i-1, j]) * gait[i, j] > 0:
                    # Offset to the future position
                    q_dxdy = np.array([self.dx[i - 1], self.dy[i - 1], 0.0])

                    # Get future desired position of footsteps
                    nextFootstep = self.computeNextFootstep(i, j, b_vlin, b_vref, True)

                    # Get desired position of footstep compared to current position
                    RPY_tmp[2] = self.yaws[i-1]
                    Rz = pin.rpy.rpyToMatrix(RPY_tmp)
                    next_ft = Rz.dot(nextFootstep) + q_tmp + q_dxdy

                    # self.footsteps[i][:,j] = next_ft

                    if j in self.feet:  # feet currently in flying phase

                        if self.t0s[j] < 10e-4 and (self.k % self.k_mpc) == 0:
                            #  beginning of flying phase, selection of surface

                            print("Next foot : ", next_ft)

                            id_surface = None
                            for id_sf, sf in enumerate(self.heightMap.Surfaces):
                                if sf.isInsideIneq(next_ft):
                                    id_surface = id_sf
                            # height , id_surface = self.heightMap.getHeight(next_ft[0] , next_ft[1])
                            self.surface_selected[j] = id_surface

                            print("Surface selected : ", id_surface)

                        id_surface = self.surface_selected[j]

                        if id_surface != None:

                            nb_ineq = self.heightMap.Surfaces[id_surface].ineq_vect.shape[0] - 1
                            L.append([i, j, id_surface, int(nb_ineq)])

                        else:
                            L.append([i, j, -99, 0])

                    else:
                        # The surface can be modified at each time step if not in flying phase
                        # height , id_surface = self.heightMap.getHeight(next_ft[0] , next_ft[1])
                        id_surface = None
                        for id_sf, sf in enumerate(self.heightMap.Surfaces):
                            if sf.isInsideIneq(next_ft):
                                id_surface = id_sf

                        if id_surface != None:

                            nb_ineq = self.heightMap.Surfaces[id_surface].ineq_vect.shape[0] - 1
                            L.append([i, j, id_surface, int(nb_ineq)])

                        else:
                            L.append([i, j, -99, 0])

            i += 1

        self.run_optimisation(np.array(L), q, b_vlin, b_vref, o_vref)

        return 0

    def computeNextFootstep(self, i, j, b_vlin, b_vref, feedback_term=True):
        ''' No vectors here, only array, easier for additions
        Params : 
        - i,j (int) : index in gait
        - b_vlin    (3x) : linear velocity in base frame (yaw rotated)
        - b_vref (6x) : b_vref in base frame  (yaw rotated)
        '''

        nextFootstep = np.zeros(3)

        t_stance = self.gait.getPhaseDuration(int(i), int(j), 1.0)  # 1.0 for stance phase

        # Add symmetry term
        nextFootstep = t_stance * 0.5 * b_vlin

        # Add feedback term
        if feedback_term:
            nextFootstep += self.k_feedback * (b_vlin - b_vref[:3])

        # Add centrifugal term
        cross = self.cross3(b_vlin, b_vref[3:6])
        nextFootstep += 0.5 * np.sqrt(self.h_ref / self.g) * cross[:, 0]

        # Legs have a limited length so the deviation has to be limited
        nextFootstep[0] = min(self.nextFootstep[0, j], self.L)
        nextFootstep[0] = max(self.nextFootstep[0, j], -self.L)
        nextFootstep[1] = min(self.nextFootstep[1, j], self.L)
        nextFootstep[1] = max(self.nextFootstep[1, j], -self.L)

        # Add shoulders
        # self.nextFootstep[:,j] += self.shoulders[:,j]

        # Taking into account Roll and Pitch (here base frame only takes yaw)
        RP = self.RPY.copy()
        RP[2] = 0.   # Remove Yaw, taking into account after
        nextFootstep += pin.rpy.rpyToMatrix(RP).dot(self.shoulders[:, j])

        # Remove Z component (working on flat ground)
        nextFootstep[2] = 0.

        return nextFootstep

    def computeTargetFootstep(self, k, q, v, b_vref):
        ''' 
        Params : 
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here [x,y,z,w] quaternion
        - v (6x1) : v in world frame (lin + ang)
        - b_vref (6x1) : b_vref in base frame (Yaw rotated)
        '''
        self.k = k

        # pio : (x,y,z,w) , pybullet same
        # c++ : quat_ = {q(6), q(3), q(4), q(5)};  // w, x, y, z ??
        quat = pin.Quaternion(q[3:7])
        self.RPY = pin.rpy.matrixToRpy(quat.toRotationMatrix())   # Array (3,)
        self.RPY_b = self.RPY.copy()

        print("RPY : ", self.RPY)

        # Only yaw rotation for the base frame
        self.RPY_b[0] = 0
        self.RPY_b[1] = 0

        # vref in world frame
        o_vref = np.zeros((6, 1))
        o_vref[:3, 0:1] = pin.rpy.rpyToMatrix(self.RPY_b).dot(b_vref[:3, 0:1])  # linear
        o_vref[3:, 0:1] = pin.rpy.rpyToMatrix(self.RPY_b).dot(b_vref[3:7, 0:1])   # angular

        # Compute the desired location of footsteps over the prediction horizon
        self.computeFootsteps(q, v, o_vref[:, 0:1])

        # Update desired location of footsteps on the ground
        self.updateTargetFootsteps()

        # Update self.fsteps, not a list, but array representation (Nx12)
        self.fsteps = np.zeros((self.N_gait, 12))

        for index, footstep in enumerate(self.footsteps):
            self.fsteps[index, :] = np.reshape(footstep, 12, order='F')

        return self.getTargetFootsteps()

    def cross3(self, left, right):
        """Numpy is inefficient for this"""
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])

    def updateTargetFootsteps(self):

        for i in range(4):
            index = 0
            while(self.footsteps[index][0, i] == 0.):
                index += 1
            self.targetFootstep[:, i] = self.footsteps[index][:, i]

        return 0

    def updateNewContact(self):

        for i in range(4):
            if self.gait.getCurrentGait()[0, i] == 1.0:
                self.current_fstep[:, i] = self.footsteps[1][:, i]

        return 0

    def getFootsteps(self):

        return self.fsteps

    def getFootstepsList(self):
        return self.footsteps

    def getTargetFootsteps(self):
        return self.targetFootstep
