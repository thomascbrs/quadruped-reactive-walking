import numpy as np
import pinocchio as pin
from solo3D.tools.optimisation import genCost, quadprog_solve_qp, to_least_square
from solo3D.tools.collision_tool import fclObj_trimesh, simple_object, gjk
import os


from sl1m.tools.obj_to_constraints import load_obj, as_inequalities, rotate_inequalities
from sl1m.constants_and_tools import default_transform_from_pos_normal

from scipy.spatial import Delaunay
import hppfcl

from time import perf_counter as clock
from solo3D.tools.ProfileWrapper import ProfileWrapper


# Store the results from cprofile
profileWrap = ProfileWrapper()

class FootStepPlannerQP_mip:
    ''' Python class to compute the target foot step and the ref state using QP optimisation 
    to avoid edges of surface.
    '''

    def __init__(self, dt, dt_wbc, T_gait, h_ref, k_mpc, gait, N_gait, surfacePlanner):

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
        
        # Coefficients QP
        self.weight_vref = 0.06
        self.weight_alpha = 1.

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

        self.o_vref_qp = np.zeros((6, 1))

        # solo-rbprm link to retrieve the kinematic constraints
        self.com_objects = []
        # Constraints inequalities size != depending of which foot is used
        self.ine_CoM_size = []
        n_effectors = 4
        # kinematic_constraints_path = "/home/thomas_cbrs/Library/solo-rbprm/data/com_inequalities/feet_quasi_flat/"
        # limbs_names = ["FLleg" , "FRleg" , "HLleg" , "HRleg"]

        kinematic_constraints_path = os.getcwd() + "/solo3D/objects/constraints_sphere/"

        limbs_names = ["FL", "FR", "HL", "HR"]
        for foot in range(n_effectors):
            foot_name = limbs_names[foot]
            # filekin = kinematic_constraints_path + "COM_constraints_in_" + \
            #     foot_name + "_effector_frame_quasi_static_reduced.obj"
            filekin = kinematic_constraints_path + "COM_constraints_" + \
                foot_name + "3.obj"
            self.com_objects.append(as_inequalities(load_obj(filekin)))
            self.ine_CoM_size.append(self.com_objects[foot].A.shape[0])

        # self.statePlanner = statePlanner

        self.res = None

        self.surfacePlanner = surfacePlanner

        # Store surface selected for the current swing phase
        self.surface_selected = [self.surfacePlanner.floor_surface] * 4

        # Store timings of the loop
        # [whole loop , convert_inequalities , res_qp , update new fstep ]
        self.timings = np.zeros(4)    

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

    def select_surface_fromXY(self, point, phase, moving_foot_index):
        ''' Given a X,Y position, select the appropriate surface by testing the potential surfaces

        Args : 
        - point Arrayx2 or Arrayx3 : 
        - phase : The potential surfaces are sorted according to the phase of the walk (cf sl1m for phase definition)
        - moving_foot_index : For each phase, potential surface are different for each foot 
        '''

        # Usefull if 2 surfaces overlaps
        h_selected = 0.
        reduced_surface_found = False

        if self.surfacePlanner.mip_iteration != 0:
            potential_surfaces = self.surfacePlanner.potential_surfaces[phase][moving_foot_index]
            for sf in potential_surfaces:
                # print("sf.vertices : " , sf.vertices)
                if sf.isInside_XY(point):
                    # Test if x,y point is inside surface
                    height_potential = sf.getHeight(point)

                    if height_potential > h_selected:
                        # Select higher surface if overlapping
                        h_selected = height_potential
                        surface_selected = sf
                        reduced_surface_found = True

            if not reduced_surface_found:
                # The X,Y point is outide of the reduced surfaces, keep closest surface
                obj1 = simple_object(np.array([[point[0], point[1], 0.]]),  [[0, 0, 0]])
                o1 = fclObj_trimesh(obj1)
                t1 = hppfcl.Transform3f()
                t2 = hppfcl.Transform3f()
                distance = 100

                for sf in potential_surfaces:
                    vert = np.zeros(sf.vertices.shape)
                    vert[:, :2] = sf.vertices[:, :2]
                    tri = Delaunay(vert[:, :2])
                    obj2 = simple_object(vert,  tri.simplices.tolist())
                    o2 = fclObj_trimesh(obj2)
                    gg = gjk(o1, o2, t1, t2)

                    if gg.distance <= distance:
                        surface_selected = sf
                        distance = gg.distance

        else:
            surface_selected = self.surfacePlanner.floor_surface

        return surface_selected

    def convert_pb_inequalities(self, L):
        ''' Convert the list of surface and next noving points into inequality matrix 
        that can be used for the QP
        Args : 
        - L : = [  [x =1 , y=0  ,  surfaceData  , next_ft_qp ] , ... ]  --> First variable in gait[1,0]
        '''
        # Total number of row in inequality matrix
        nrow = 0
        for l in L:
            nrow += l[2].A.shape[0]

        # Total number of optimized variable (Vx , Vy , rx1 , ry1 , rz1 , rx2 ... )
        ncol = 2 + 3*len(L)

        ineqMatrix = np.zeros((nrow, ncol))
        ineqVector = np.zeros(nrow)

        i_start = 0
        for index_L, [i, j, surface, next_ft] in enumerate(L):
            i_start = self.surface_inequality(surface, next_ft, ineqMatrix, ineqVector, i_start, index_L)

        return ineqMatrix, ineqVector

    def solve_qp(self, ineqMatrix, ineqVector, o_vref):
        ''' Solve the QP problem 
        X optimised : [Vxref , Vyref , alpha_x1 , alpha_y1 , pz1 , alpha_x2 ... , ]
        min (1/2)x' P x + q' x
        subject to  G x <= h
        '''
        P = np.identity(ineqMatrix.shape[1])
        q = np.zeros(ineqMatrix.shape[1])

        P[:2, :2] = self.weight_vref*np.identity(2)
        q[:2] = -self.weight_vref*np.array([o_vref[0, 0], o_vref[1, 0]])

        try:
            self.res = quadprog_solve_qp(P, q,  G=ineqMatrix, h=ineqVector)
        except:
            print("------------ ERROR QP FOOTSTEP ------------")
        #     res = self.res

        return self.res

    def surface_inequality(self, surface, next_ft, ineqMatrix, ineqVector, i_start, j):
        ''' Update the inequality matrix with surface inequalities : 
        MX <= n , with X = [Vx_ref , Vy_ref , alpha_x1 , alpha_y1 , p_z1 , alpha_x2 ...] 
        Args : 
        - surface : Ax <= b , A = surface[0] , b = surface[1]
        - next_ft : next feet position without the k_feedback
        - ineqMatrix : inequality matrix to fill
        - ineqVector : ineqVector vector to fill
        - i_start : indice of the first row
        - j : indice of the variable
        '''

        nrow = surface.A.shape[0]
        i_end = i_start + nrow
        j_start = 3*j + 2
        j_end = 3*j+5

        # surface.b[-2] += 0.001
        # surface.b[-1] += -0.001

        # S * [Vrefx , Vrefy]
        ineqMatrix[i_start:i_start + nrow, :2] = -self.k_feedback*surface.A[:, :2]

        # S * [alphax  , aplhay  Pz]
        ineqMatrix[i_start:i_end, j_start: j_end] = surface.A[:, :]

        ineqVector[i_start:i_end] = surface.b - np.dot(surface.A[:, :2], next_ft[:2])

        return i_end

    def integration_desired_velocity(self, o_vref, v, gait):
        ''' Integration OF desired velocity in world frame, to get
        dx,dy,dyaws in world frame for the time horizon
        Args : 
        - o_vref (6x1) : o_vref in world frame
        Return :
        - update self.dt_cum, self.yaws, self.dx, self.dy
        '''

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

        return 0

    def computeFootsteps(self, q, v, o_vref):
        ''' 
        Params : 
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here x,y,z,w quaternion
        - v (6x1) : v in world frame (lin + ang)
        - o_vref (6x1) : o_vref in world frame
        '''
        # Update gait and clear list of foosteps
        gait = self.gait.getCurrentGait()

        self.footsteps.clear()
        for i in range(self.N_gait):
            self.footsteps.append(np.zeros((3, 4)))

        # Set current position of feet for feet in stance phase
        for j in range(4):
            if gait[0, j] == 1.:
                self.footsteps[0][:, j] = self.current_fstep[:, j]

        # Integration of the desired velocities (Vx,Vy,Wyaw)
        self.integration_desired_velocity(o_vref, v, gait)

        # Get current and reference velocities in base frame (rotated yaw)
        RPY_tmp = np.zeros(3)
        Rz = pin.rpy.rpyToMatrix(self.RPY_b)
        b_vlin = Rz.transpose().dot(v[:3, 0])   # linear velocity in base frame, array
        b_vref = np.zeros(6)
        b_vref[:3] = Rz.transpose().dot(o_vref[:3, 0])
        b_vref[3:] = Rz.transpose().dot(o_vref[3:, 0])

        # Current position without height
        q_tmp = q[:3, 0].copy()
        q_tmp[2] = 0.

        # Update remaining time flying phase
        self.update_remaining_time()

        # List of selected surface and next position to optimise
        # L = [  [x =1 , y=0  ,  surface_ineq  , next_ft_qp ] , ... ]  --> First variable in gait[1,0]
        # Ax <= b : A = surface_ineq[0] , b = surface_ineq[1]
        L = []

        # Update the footstep matrix depending on the different phases of the gait (swing & stance)
        i = 1
        phase = 0

        while (gait[i, :].any()):
            # Feet that were in stance phase and are still in stance phase do not move
            for j in range(4):
                if gait[i-1, j]*gait[i, j] > 0:
                    self.footsteps[i][:, j] = self.footsteps[i-1][:, j]

            # Feet that were in swing phase and are now in stance phase need to be updated
            moving_foot_index = 0
            for j in range(4):

                if (1 - gait[i-1, j]) * gait[i, j] > 0:
                    # Offset to the future position
                    q_dxdy = np.array([self.dx[i - 1], self.dy[i - 1], 0.0])

                    # Get future desired position of footsteps
                    nextFootstep = self.computeNextFootstep(i, j, b_vlin, b_vref, True)

                    # Get future desired position of footsteps without k_feedback
                    nextFootstep_qp = self.computeNextFootstep(i, j, b_vlin, b_vref, False)

                    # Get desired position of footstep compared to current position
                    RPY_tmp[2] = self.yaws[i-1]
                    Rz_tmp = pin.rpy.rpyToMatrix(RPY_tmp)
                    next_ft = Rz_tmp.dot(nextFootstep) + q_tmp + q_dxdy
                    next_ft_qp = Rz_tmp.dot(nextFootstep_qp) + q_tmp + q_dxdy

                    if j in self.feet and phase == 0:
                        # feet currently in flying phase
                        if self.t0s[j] < 10e-4 and (self.k % self.k_mpc) == 0:
                            # Beginning of flying phase, selection of surface

                            if self.surfacePlanner.mip_success:
                                # Surface from SL1M if converged
                                self.surface_selected[j] = self.surfacePlanner.selected_surfaces[phase][moving_foot_index]
                                pass
                            else:
                                # Surface from X,Y Heuristic
                                self.surface_selected[j] = self.select_surface_fromXY(next_ft, phase, moving_foot_index)

                        L.append([i, j, self.surface_selected[j], next_ft_qp])

                    else:
                        if self.surfacePlanner.mip_success:
                            # Surface from SL1M if converged
                            surface_selected = self.surfacePlanner.selected_surfaces[phase][moving_foot_index]
                            pass
                        else:
                            # Surface from X,Y Heuristic
                            surface_selected = self.select_surface_fromXY(next_ft, phase, moving_foot_index)

                        L.append([i, j, surface_selected, next_ft_qp])

                    moving_foot_index += 1

            if ((1 - gait[i-1, :]) * gait[i, :]).any():
                phase += 1

            i += 1

        t0 = clock()

        # Convert Problem to inequality matrix
        ineqMatrix, ineqVector = self.convert_pb_inequalities(L)

        t1 = clock()
        self.timings[1] = 1000*(t1-t0)

        t0 = clock()
        # Solve QP problem
        res = self.solve_qp(ineqMatrix, ineqVector, o_vref)
        t1 = clock()
        self.timings[2] = 1000*(t1-t0)


        t0 = clock()

        # Store the value for getRefState
        self.o_vref_qp = o_vref.copy()
        self.o_vref_qp[0, 0] = res[0]
        self.o_vref_qp[1, 0] = res[1]

        # Get new reference velocity in base frame to recompute the new footsteps
        b_vref_qp = np.zeros(6)
        b_vref_qp[:3] = Rz.transpose().dot(self.o_vref_qp[:3, 0])
        b_vref_qp[3:] = Rz.transpose().dot(self.o_vref_qp[3:, 0])

        for l in range(len(L)):
            i, j = L[l][0], L[l][1]
            # Offset to the future position
            q_dxdy = np.array([self.dx[i - 1], self.dy[i - 1], 0.0])

            # Get future desired position of footsteps without k_feedback
            nextFootstep = self.computeNextFootstep(i, j, b_vlin, b_vref_qp, True)

            # Get desired position of footstep compared to current position
            RPY_tmp[2] = self.yaws[i-1]
            Rz_tmp = pin.rpy.rpyToMatrix(RPY_tmp)
            next_ft = Rz_tmp.dot(nextFootstep) + q_tmp + q_dxdy

            next_ft += np.array([res[2+3*l], res[2+3*l+1], res[2+3*l+2]])

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

        t1 = clock()
        self.timings[3] = 1000*(t1-t0)

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

    
    @profileWrap.profile
    def computeTargetFootstep(self, k, q, v, b_vref):
        ''' 
        Params : 
        - q (7x1) : [px , py , pz , x , y , z , w]  --> here [x,y,z,w] quaternion
        - v (6x1) : v in world frame (lin + ang)
        - b_vref (6x1) : b_vref in base frame (Yaw rotated)
        '''
        t0 = clock()

        self.k = k

        # pio : (x,y,z,w) , pybullet same
        # c++ : quat_ = {q(6), q(3), q(4), q(5)};  // w, x, y, z ??
        quat = pin.Quaternion(q[3:7])
        self.RPY = pin.rpy.matrixToRpy(quat.toRotationMatrix())   # Array (3,)
        self.RPY_b = self.RPY.copy()

        # print("RPY : ", self.RPY)

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

        t1 = clock()
        self.timings[0] = 1000*(t1 - t0)

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

    def getVref_QP(self):
        return self.o_vref_qp

    def print_profile(self , output_file):
        ''' Print the profile computed with cProfile
        Args : 
        - output_file (str) :  file name
        '''
        profileWrap.print_stats(output_file)
        
        return  0
