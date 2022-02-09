from solo3D.SurfacePlanner import SurfacePlanner

from multiprocessing import Process
from multiprocessing.sharedctypes import Value
import ctypes
import os

import libquadruped_reactive_walking as lqrw

from time import perf_counter as clock
import numpy as np

# TODO : Modify this, should not be defined here
params = lqrw.Params()

N_VERTICES_MAX = 4
N_SURFACE_MAX = 10
N_SURFACE_CONFIG = 3
N_gait = int(params.gait.shape[0])
N_POTENTIAL_SURFACE = 7
N_FEET = 4
N_PHASE = 3


class SurfaceDataCtype(ctypes.Structure):
    ''' ctype data structure for the shared memory between processes, for surfaces
    Ax <= b
    If normal as inequalities : A = [A , n , -n] , b = [b , d + eps, -d-eps]
    Normal as equality : A = [A , n] , b = [b , d]
    A =  inequality matrix, last line equality if normal as equality
    b =  inequality vector,
    vertices =  vertices of the surface  [[x1,y1,z1],
                                                      [x2,y2,z2], ...]

    on = Bool if the surface is used or not
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    nvertices = 4
    nrow = nvertices + 2
    _fields_ = [('A', ctypes.c_double * 3 * nrow), ('b', ctypes.c_double * nrow),
                ('vertices', ctypes.c_double * 3 * nvertices), ('on', ctypes.c_bool)]


class DataOutCtype(ctypes.Structure):
    '''  ctype data structure for the shared memory between processes
    Data Out, list of potential and the selected surfaces given by the MIP
    Potential surfaces are used if the MIP has not converged
    '''
    _fields_ = [('potentialSurfaces', SurfaceDataCtype * N_POTENTIAL_SURFACE * N_FEET),
                ('selectedSurfaces', SurfaceDataCtype * N_FEET), ('all_feet', ctypes.c_double * 12 * N_PHASE),
                ('success', ctypes.c_bool)]


class DataInCtype(ctypes.Structure):
    ''' ctype data structure for the shared memory between processes
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    _fields_ = [('gait', ctypes.c_int64 * 4 * N_gait), ('configs', ctypes.c_double * 7 * N_SURFACE_CONFIG),
                ('bvref', ctypes.c_double * 3), ('contacts', ctypes.c_double * 12), ('iteration', ctypes.c_int64)]


class SurfacePlanner_Wrapper():
    ''' 
    Wrapper for the class SurfacePlanner for the paralellisation
    '''

    def __init__(self, params):

        self.params = params

        # Usefull for 1st iteration of QP
        A = [[-1.0000000, 0.0000000, 0.0000000], [0.0000000, -1.0000000, 0.0000000],
             [0.0000000, 1.0000000, 0.0000000], [1.0000000, 0.0000000, 0.0000000],
             [0.0000000, 0.0000000, 1.0000000], [0.0000000, 0.0000000, -1.0000000]]
        b = [1.3946447, 0.9646447, 0.9646447, 0.5346446, 0.0000, 0.0000]
        vertices = [[-1.3946447276978748, 0.9646446609406726, 0.0], [-1.3946447276978748, -0.9646446609406726, 0.0],
                    [0.5346445941834704, -0.9646446609406726, 0.0], [0.5346445941834704, 0.9646446609406726, 0.0]]
        self.floor_surface = lqrw.Surface(np.array(A), np.array(b), np.array(vertices))

        # Results used by controller
        self.potential_surfaces = lqrw.SurfaceVectorVector()
        self.selected_surfaces = lqrw.SurfaceVector()
        self.all_feet_pos = []
        self.mip_iteration = 0
        self.mip_success = False
        self.first_iteration = True

        # When synchronous, values are stored to be used by controller only at the next flying phase
        self.mip_iteration_syn = 0
        self.mip_success_syn = False
        self.selected_surfaces_syn = lqrw.SurfaceVector()
        self.all_feet_pos_syn = []

        self.multiprocessing = params.enable_multiprocessing_mip
        if self.multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', True)
            self.running = Value('b', True)
            self.dataOut = Value(DataOutCtype)
            self.dataIn = Value(DataInCtype)
        else:
            self.surfacePlanner = SurfacePlanner(params)

    def run(self, configs, gait_in, current_contacts, bvref):
        if self.multiprocessing:
            self.run_asynchronous(configs, gait_in, current_contacts, bvref)
        else:
            self.run_synchronous(configs, gait_in, current_contacts, bvref)

    def run_synchronous(self, configs, gait_in, current_contacts, bvref):
        surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success = self.surfacePlanner.run(
            configs, gait_in, current_contacts, bvref)
        self.mip_iteration_syn += 1
        self.mip_success_syn = success

        # Retrieve potential data, usefull if solver not converged
        self.potential_surfaces = lqrw.SurfaceVectorVector()
        for foot, foot_surfaces in enumerate(surface_inequalities):
            list_surfaces = lqrw.SurfaceVector()
            for i, (S, s) in enumerate(foot_surfaces):
                list_surfaces.append(lqrw.Surface(S, s, surfaces[foot][i].T))
            self.potential_surfaces.append(list_surfaces)

        # Mimic the multiprocessing behaviour, store the resuts and get them with update function
        self.selected_surfaces_syn = lqrw.SurfaceVector()
        if success:
            for foot, foot_surfaces in enumerate(surface_inequalities):
                i = surfaces_indices[foot]
                S, s = foot_surfaces[i]
                self.selected_surfaces_syn.append(lqrw.Surface(S, s, surfaces[foot][i].T))

            self.all_feet_pos_syn = all_feet_pos.copy()

    def run_asynchronous(self, configs, gait_in, current_contacts, bvref):

        # If this is the first iteration, creation of the parallel process
        with self.dataIn.get_lock():
            if self.dataIn.iteration == 0:
                p = Process(target=self.create_MIP_asynchronous,
                            args=(self.newData, self.newResult, self.running, self.dataIn, self.dataOut))
                p.start()
        # Stacking data to send them to the parallel process
        self.compress_dataIn(configs, gait_in, current_contacts, bvref)
        self.newData.value = True

    def create_MIP_asynchronous(self):
        while self.running.value:
            # Checking if new data is available to trigger the asynchronous MPC
            if self.newData.value:
                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                self.newData.value = False

                configs, gait_in, bvref, current_contacts = self.decompress_dataIn()

                with self.dataIn.get_lock():
                    if self.dataIn.iteration == 0:
                        loop_planner = SurfacePlanner(self.params)

                surfaces, surface_inequalities, surfaces_indices, all_feet_pos, success = loop_planner.run(
                    configs, gait_in, current_contacts, bvref)

                with self.dataIn.get_lock():
                    self.dataIn.iteration += 1

                self.compress_dataOut(surfaces, surface_inequalities, surfaces_indices, success)

                # Set shared variable to true to signal that a new result is available
                self.newResult.value = True

    def compress_dataIn(self, configs, gait_in, current_contacts, bvref):

        with self.dataIn.get_lock():

            for i, config in enumerate(configs):
                dataConfig = np.frombuffer(self.dataIn.configs[i])
                dataConfig[:] = config[:]

            gait = np.frombuffer(self.dataIn.gait).reshape((N_gait, 4))
            gait[:, :] = gait_in

            bvref = np.frombuffer(self.dataIn.bvref)
            bvref[:] = bvref[:]

            contact = np.frombuffer(self.dataIn.contacts).reshape((3, 4))
            contact[:, :] = current_contacts[:, :]

    def decompress_dataIn(self):
        with self.dataIn.get_lock():
            configs = [np.frombuffer(config) for config in self.dataIn.configs]
            gait = np.frombuffer(self.dataIn.gait).reshape((N_gait, 4))
            bvref = np.frombuffer(self.dataIn.bvref).reshape((3))
            contacts = np.frombuffer(self.dataIn.contacts).reshape((3, 4))

        return configs, gait, bvref, contacts

    def compress_dataOut(self, surfaces, surface_inequalities, surfaces_indices, success):
        nrow = N_VERTICES_MAX + 2
        with self.dataOut.get_lock():
            # Compress potential surfaces :
            for foot, foot_surfaces in enumerate(surface_inequalities):
                i = 0
                for i, (S, s) in enumerate(foot_surfaces):
                    A = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].A).reshape((nrow, 3))
                    b = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].b)
                    vertices = np.frombuffer(self.dataOut.potentialSurfaces[foot][i].vertices).reshape((nvertices, 3))
                    A[:, :] = S[:, :]
                    b[:] = s[:]
                    vertices[:, :] = surfaces[foot][i].T[:, :]
                    self.dataOut.potentialSurfaces[foot][i].on = True

                for j in range(i + 1, N_POTENTIAL_SURFACE):
                    self.dataOut.potentialSurfaces[foot][j].on = False

            if success:
                self.dataOut.success = True

                # Compress selected surfaces
                for foot, index in enumerate(surfaces_indices):
                    A = np.frombuffer(self.dataOut.selectedSurfaces[foot].A).reshape((nrow, 3))
                    b = np.frombuffer(self.dataOut.selectedSurfaces[foot].b)
                    vertices = np.frombuffer(self.dataOut.selectedSurfaces[foot].vertices).reshape((nvertices, 3))
                    A[:, :] = surface_inequalities[foot][index][0][:, :]
                    b[:] = surface_inequalities[foot][index][1][:]
                    vertices = surfaces[foot][index].T[:, :]
                    self.dataOut.selectedSurfaces[foot].on = True

            else:
                self.dataOut.success = False

    def update_latest_results(self):
        ''' Update latest results : 2 list 
        - potential surfaces : used if MIP has not converged
        - selected_surfaces : surfaces selected for the next phases
        '''
        if self.multiprocessing:
            with self.dataOut.get_lock():
                if self.newResult.value:
                    self.newResult.value = False

                    self.potential_surfaces = lqrw.SurfaceVectorVector()
                    for foot_surfaces in self.dataOut.potentialSurfaces:
                        list_surfaces = lqrw.SurfaceVector()
                        for s in foot_surfaces:
                            if s.on:
                                list_surfaces.append(lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))
                        self.potential_surfaces.append(list_surfaces)

                    self.selected_surfaces = lqrw.SurfaceVector()

                    if self.dataOut.success:
                        self.mip_success = True
                        self.mip_iteration += 1

                        for s in self.dataOut.selectedSurfaces:
                            self.selected_surfaces.append(
                                lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))
                    else:
                        self.mip_success = False
                        self.mip_iteration += 1

                else:
                    # TODO : So far, only the convergence or not of the solver has been taken into account,
                    # What if the solver take too long ?
                    pass
        else:
            # Results have been already updated
            self.mip_success = self.mip_success_syn
            self.mip_iteration = self.mip_iteration_syn

            if self.mip_success:
                self.selected_surfaces = self.selected_surfaces_syn
                self.all_feet_pos = self.all_feet_pos_syn.copy()

    def stop_parallel_loop(self):
        """
        Stop the infinite loop in the parallel process to properly close the simulation
        """
        self.running.value = False
