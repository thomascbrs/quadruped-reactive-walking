from solo3D.SurfacePlanner import SurfacePlanner

from multiprocessing import Process
from multiprocessing.sharedctypes import Value
import ctypes
import time

import libquadruped_reactive_walking as lqrw
import numpy as np

N_VERTICES_MAX = 4
N_ROWS = N_VERTICES_MAX + 2
N_SURFACE_MAX = 10
N_SURFACE_CONFIG = 3
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
    _fields_ = [('A', ctypes.c_double * 3 * N_ROWS), ('b', ctypes.c_double * N_ROWS),
                ('vertices', ctypes.c_double * 3 * N_VERTICES_MAX), ('on', ctypes.c_bool)]


class DataOutCtype(ctypes.Structure):
    '''  ctype data structure for the shared memory between processes
    Data Out, list of potential and the selected surfaces given by the MIP
    Potential surfaces are used if the MIP has not converged
    '''
    _fields_ = [('potentialSurfaces', SurfaceDataCtype * N_POTENTIAL_SURFACE * N_FEET),
                ('selectedSurfaces', SurfaceDataCtype * N_FEET), ('all_feet', ctypes.c_double * 12 * N_PHASE),
                ('success', ctypes.c_bool), ('t_mip', ctypes.c_float)]


class DataInCtype(ctypes.Structure):
    ''' ctype data structure for the shared memory between processes
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    params = lqrw.Params()                  # Object that holds all controller parameters
    n_gait = int(params.gait.shape[0])
    _fields_ = [('gait', ctypes.c_int64 * 4 * n_gait), ('configs', ctypes.c_double * 7 * N_SURFACE_CONFIG),
                ('b_v_ref', ctypes.c_double * 3), ('contacts', ctypes.c_double * 12)]


class SurfacePlanner_Wrapper():
    '''
    Wrapper for the class SurfacePlanner for the paralellisation
    '''

    def __init__(self, params):

        self.params = params
        self.n_gait = int(params.gait.shape[0])

        # Usefull for 1st iteration of QP
        A = [[-1.0000000, 0.0000000, 0.0000000], [0.0000000, -1.0000000, 0.0000000],
             [0.0000000, 1.0000000, 0.0000000], [1.0000000, 0.0000000, 0.0000000],
             [0.0000000, 0.0000000, 1.0000000], [0.0000000, 0.0000000, -1.0000000]]
        b = [1.3946447, 0.9646447, 0.9646447, 0.5346446, 0.0000, 0.0000]
        vertices = [[-1.3946447276978748, 0.9646446609406726, 0.0], [-1.3946447276978748, -0.9646446609406726, 0.0],
                    [0.5346445941834704, -0.9646446609406726, 0.0], [0.5346445941834704, 0.9646446609406726, 0.0]]
        self.floor_surface = lqrw.Surface(np.array(A), np.array(b), np.array(vertices))

        # Results used by controller
        self.mip_success = False
        self.t_mip = 0.
        self.mip_iteration = 0
        self.potential_surfaces = lqrw.SurfaceVectorVector()
        self.selected_surfaces = lqrw.SurfaceVector()
        self.all_feet_pos = []

        # When synchronous, values are stored to be used by controller only at the next flying phase
        self.mip_success_syn = False
        self.mip_iteration_syn = 0
        self.potential_surfaces_syn = lqrw.SurfaceVectorVector()
        self.selected_surfaces_syn = lqrw.SurfaceVector()

        self.multiprocessing = params.enable_multiprocessing_mip
        if self.multiprocessing:
            self.new_data = Value('b', False)
            self.new_result = Value('b', True)
            self.running = Value('b', True)
            self.data_out = Value(DataOutCtype)
            self.data_in = Value(DataInCtype)
            p = Process(target=self.asynchronous_process, args=(self.new_data, self.new_result, self.running, self.data_in, self.data_out))
            p.start()
        else:
            self.planner = SurfacePlanner(params)

        self.initialized = False

    def run(self, configs, gait_in, current_contacts, b_v_ref):
        ''' 
        Either call synchronous or asynchronous planner
        '''
        if self.multiprocessing:
            self.run_asynchronous(configs, gait_in, current_contacts, b_v_ref)
        else:
            self.run_synchronous(configs, gait_in, current_contacts, b_v_ref)

    def run_synchronous(self, configs, gait, current_contacts, b_v_ref):
        ''' 
        Call the planner and store the result in syn variables
        '''
        t_start = time.time()
        vertices, inequalities, indices, self.all_feet_pos, success = self.planner.run(configs, gait, current_contacts, b_v_ref)
        self.mip_iteration_syn += 1
        self.mip_success_syn = success

        self.potential_surfaces = lqrw.SurfaceVectorVector()
        for foot, foot_surfaces in enumerate(inequalities):
            potential_surfaces = lqrw.SurfaceVector()
            for i, (S, s) in enumerate(foot_surfaces):
                potential_surfaces.append(lqrw.Surface(S, s, vertices[foot][i].T))
            self.potential_surfaces.append(potential_surfaces)

        self.selected_surfaces_syn = lqrw.SurfaceVector()
        if success:
            for foot, foot_inequalities in enumerate(inequalities):
                S, s = foot_inequalities[indices[foot]]
                self.selected_surfaces_syn.append(lqrw.Surface(S, s, vertices[foot][indices[foot]].T))
        self.t_mip = time.time() - t_start

    def run_asynchronous(self, configs, gait_in, current_contacts, b_v_ref):
        ''' 
        Compress the data and send them for asynchronous process 
        '''
        with self.data_in.get_lock():
            for i, config in enumerate(configs):
                data_config = np.frombuffer(self.data_in.configs[i])
                data_config[:] = config[:]
            gait = np.frombuffer(self.data_in.gait).reshape((self.n_gait, 4))
            gait[:, :] = gait_in
            b_v_ref = np.frombuffer(self.data_in.b_v_ref)
            b_v_ref[:] = b_v_ref[:]
            contact = np.frombuffer(self.data_in.contacts).reshape((3, 4))
            contact[:, :] = current_contacts[:, :]
        self.new_data.value = True

    def asynchronous_process(self, new_data, new_result, running, data_in, data_out):
        ''' 
        Asynchronous process created during initialization 
        '''
        planner = SurfacePlanner(self.params)
        while running.value:
            if new_data.value:
                new_data.value = False
                t_start = time.time()

                with data_in.get_lock():
                    configs = [np.frombuffer(config) for config in data_in.configs]
                    gait = np.frombuffer(data_in.gait).reshape((self.n_gait, 4))
                    b_v_ref = np.frombuffer(data_in.b_v_ref).reshape((3))
                    contacts = np.frombuffer(data_in.contacts).reshape((3, 4))

                vertices, inequalities, indices, _, success = planner.run(configs, gait, contacts, b_v_ref)
                
                t = time.time() - t_start
                self.compress_result(vertices, inequalities, indices, success, t, data_out)
                new_result.value = True


    def compress_result(self, vertices, inequalities, indices, success, t, data_out):
        ''' 
        Store the planner result in data_out
        '''
        with data_out.get_lock():
            data_out.success = success
            data_out.t_mip = t
            for foot, foot_inequalities in enumerate(inequalities):
                i=0
                for i, (S, s) in enumerate(foot_inequalities):
                    A = np.frombuffer(data_out.potentialSurfaces[foot][i].A).reshape((N_ROWS, 3))
                    A[:, :] = S[:, :]
                    b = np.frombuffer(data_out.potentialSurfaces[foot][i].b)
                    b[:] = s[:]
                    v = np.frombuffer(data_out.potentialSurfaces[foot][i].vertices).reshape((N_VERTICES_MAX, 3))
                    v[:, :] = vertices[foot][i].T[:, :]
                    data_out.potentialSurfaces[foot][i].on = True

                for j in range(i + 1, N_POTENTIAL_SURFACE):
                    data_out.potentialSurfaces[foot][j].on = False

            if success:
                for foot, index in enumerate(indices):
                    A = np.frombuffer(data_out.selectedSurfaces[foot].A).reshape((N_ROWS, 3))
                    A[:, :] = inequalities[foot][index][0][:, :]
                    b = np.frombuffer(data_out.selectedSurfaces[foot].b)
                    b[:] = inequalities[foot][index][1][:]
                    v = np.frombuffer(data_out.selectedSurfaces[foot].vertices).reshape((N_VERTICES_MAX, 3))
                    v = vertices[foot][index].T[:, :]
                    data_out.selectedSurfaces[foot].on = True

    def get_latest_results(self):
        ''' 
        Update latest results : 2 list 
        - potential surfaces : used if MIP has not converged
        - selected_surfaces : surfaces selected for the next phases
        '''
        if self.multiprocessing:
            if self.new_result.value:
                self.new_result.value = False

                with self.data_out.get_lock():
                    self.mip_success = self.data_out.success
                    self.t_mip = self.data_out.t_mip
                    self.mip_iteration += 1

                    self.potential_surfaces = lqrw.SurfaceVectorVector()
                    for foot_surfaces in self.data_out.potentialSurfaces:
                        potential_surfaces = lqrw.SurfaceVector()
                        for s in foot_surfaces:
                            if s.on:
                                potential_surfaces.append(lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))
                        self.potential_surfaces.append(potential_surfaces)

                    self.selected_surfaces = lqrw.SurfaceVector()
                    if self.data_out.success:
                        for s in self.data_out.selectedSurfaces:
                            self.selected_surfaces.append(lqrw.Surface(np.array(s.A), np.array(s.b), np.array(s.vertices)))
            else:
                print("Error: No new MIP result available \n")
                pass
        else:
            self.mip_success = self.mip_success_syn
            self.mip_iteration = self.mip_iteration_syn
            self.potential_surfaces = self.potential_surfaces_syn
            if self.mip_success:
                self.selected_surfaces = self.selected_surfaces_syn
        
        return not self.mip_success

    def stop_parallel_loop(self):
        """
        Stop the infinite loop in the parallel process to properly close the simulation
        """
        self.running.value = False
