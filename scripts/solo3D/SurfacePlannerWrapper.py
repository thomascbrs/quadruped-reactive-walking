from solo3D.SurfacePlanner import SurfacePlanner

from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double
import ctypes

from time import perf_counter as clock
import numpy as np 

# TODO : Modify this, should not be defined here
N_VERTICES_MAX = 4
N_SURFACE_MAX = 10
EQ_AS_INEQ = True
N_SURFACE_CONFIG = 5
N_gait = 100
N_POTENTIAL_SURFACE = 10
N_MOVING_FEET = 2
N_PHASE = 5

class SurfaceDataCtype(Structure ):
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
    if EQ_AS_INEQ :
        nrow = nvertices + 2
    else : 
        nrow = nvertices + 1
    _fields_ = [('A', ctypes.c_double *  3 * nrow ), ('b', ctypes.c_double * nrow ) , ('vertices' , ctypes.c_double *  3 * nvertices) , 
                ('on' , ctypes.c_bool)]

class DataOutCtype(Structure ):
    '''  ctype data structure for the shared memory between processes
    Data Out, list of potential and the selected surfaces given by the MIP
    Potential surfaces are used if the MIP has not converged
    '''
    _fields_ = [('potentialSurfaces',  SurfaceDataCtype * N_POTENTIAL_SURFACE * N_MOVING_FEET * N_PHASE ) ,
                ('selectedSurfaces',   SurfaceDataCtype* 2 * N_PHASE ) , 
                ('all_feet' , ctypes.c_double *12 * N_PHASE )  , 
                ('success' , ctypes.c_bool )]



class DataInCtype(Structure ):
    ''' ctype data structure for the shared memory between processes
    TODO : if more than 4 vertices, add a variable for the number of vertice to reshape the appropriate buffer
    '''
    _fields_ = [('gait',  ctypes.c_int64 * 4* N_gait ) ,
                ('configs',   ctypes.c_double* 7 * N_SURFACE_CONFIG ) , 
                ('o_vref' , ctypes.c_double *6 ) , 
                ('contacts' , ctypes.c_double*12  ) , 
                ('iteration' , ctypes.c_int64) ]


class SurfaceData():
    """
    Ax <= b
    If normal as inequalities : A = [A , n , -n] , b = [b , d + eps, -d-eps]
    Normal as equality : A = [A , n] , b = [b , d] 
    surfaceData.normal_as_inequality = Boolean
    surfaceData.A =  inequality matrix, last line equality if normal as equality
    surfaceData.b =  inequality vector, 
    surfaceData.vertices =  vertices of the surface  [[x1,y1,z1],
                                                      [x2,y2,z2], ...] 
    """

    def __init__(self , A , b , vertices):
        self.normal_as_inequality = True
        self.A = A
        self.b = b 
        self.vertices = vertices 

    def isInside_XY(self,point ) :
        ''' Return a boolean whether point in x,y axes is inside the surface
        Args : 
        - point (array x2), works with array x3
        '''     
        if self.normal_as_inequality :   
            Sx = np.dot(self.A[:-2,:-1], point[:2])
            return np.sum(Sx <= self.b[:-2]) == (self.b.shape[0] - 2)
        else : 
            Sx = np.dot(self.A[:-1,:-1], point[:2])
            return np.sum(Sx <= self.b[:-1]) == (self.b.shape[0] - 1)


    def getHeight(self,point ) :
            ''' For a given X,Y point that belongs to the surface, return the height
            d/c -a/c*x -b/c*y
            Args : 
            - point (array x2), works with arrayx3
            '''
            return abs(self.b[-1] - point[0]*self.A[-1,0]/self.A[-1,2] - point[1]*self.A[-1,1]/self.A[-1,2])


class SurfacePlanner_Wrapper():
    ''' Wrapper for the class SurfacePlanner for the paralellisation
    '''

    def __init__(self, environment_URDF, T_gait , N_gait , n_surface_configs) : 

        self.environment_URDF = environment_URDF
        self.T_gait = T_gait
        
        # TODO : Modify this
        # Usefull for 1st iteration of QP 
        A = [[ 0.00000000e+00 ,-1.00000000e+00 , 0.00000000e+00],
            [ 1.00000000e+00,  0.00000000e+00 , 0.00000000e+00],
            [-1.00000000e+00  ,0.00000000e+00 , 0.00000000e+00],
            [ 0.00000000e+00 , 1.00000000e+00 , 0.00000000e+00],
            [ 0.00000000e+00 ,-2.80426163e-06 , 1.00000000e+00],
            [-0.00000000e+00,  2.80426163e-06 ,-1.00000000e+00]]

        b = [ 1.24222691e+00 , 7.39384262e-01 , 2.03938475e+00 , 1.60222740e+00 ,-2.50476789e-06 , 2.50476789e-06]

        vertices = [[-2.03938475e+00 ,-1.24222691e+00 ,-5.98829714e-06],
                    [ 7.39384262e-01 ,-1.24222691e+00 ,-5.98829714e-06],
                    [ 7.39384262e-01  ,1.60222740e+00 , 1.98829693e-06],
                    [-2.03938475e+00  ,1.60222740e+00 , 1.98829693e-06]]

        self.floor_surface = SurfaceData(np.array(A) , np.array(b) , np.array(vertices) )    

        # Results from MIP
        # Potential and selected surfaces for QP planner
        self.potential_surfaces = [] 
        self.selected_surfaces = []
        self.all_feet_pos = []

        # MIP status
        self.mip_iteration = 0
        self.mip_success = False 
        self.first_iteration = True

        self.multiprocessing = False 
        if self.multiprocessing:  # Setup variables in the shared memory
            self.newData = Value('b', False)
            self.newResult = Value('b', True)
            self.running = Value('b', True)

            # Data Out : 
            self.dataOut = Value(DataOutCtype)
            # Data IN : 
            self.dataIn = Value(DataInCtype)    

        else : 
            self.surfacePlanner =  SurfacePlanner(self.environment_URDF , self.T_gait ) 

    def run(self, configs, gait_in, current_contacts, o_v_ref) :
        
        if self.multiprocessing :
            self.run_asynchronous(configs, gait_in, current_contacts, o_v_ref  )
        else : 
            self.run_synchronous(configs, gait_in, current_contacts, o_v_ref  )

    
    def run_synchronous(self, configs, gait_in, current_contacts, o_v_ref) :

        surfaces_vertices , surfaces_inequality , surfaces_indices  , all_feet_pos , success =  self.surfacePlanner.run(configs, gait_in, current_contacts, o_v_ref)
        self.mip_iteration += 1
        self.mip_success = success

        # Retrieve potential data, usefull if solver not converged
        self.potential_surfaces.clear()
        self.potential_surfaces = [[ [SurfaceData(sf[0] , sf[1] , surfaces_vertices[index_phase][index_foot][index_sf].T )
                                                                        for index_sf , sf in enumerate(surfaces)]     \
                                                                        for index_foot,surfaces in enumerate(phase.S)]  \
                                                                        for index_phase , phase in enumerate(surfaces_inequality) ]  
        if success : 
            self.selected_surfaces = [ [SurfaceData(surfaces_inequality[index_phase].S[index_foot][id_selected][0] , 
                                                     surfaces_inequality[index_phase].S[index_foot][id_selected][1]  , 
                                                     surfaces_vertices[index_phase][index_foot][id_selected].T )
                                                                        for index_foot , id_selected in enumerate(phaseid) ]  \
                                                                        for index_phase , phaseid in enumerate(surfaces_indices) ]  

            self.all_feet_pos = all_feet_pos
        
        return 0


            
    def run_asynchronous(self, configs, gait_in, current_contacts, o_v_ref  ):

        # If this is the first iteration, creation of the parallel process
        with self.dataIn.get_lock():
            if self.dataIn.iteration == 0 :
                p = Process(target=self.create_MIP_asynchronous, args=(
                    self.newData,  self.newResult, self.running , self.dataIn , self.dataOut ))
                p.start()
        # Stacking data to send them to the parallel process
        self.compress_dataIn(  configs, gait_in, current_contacts, o_v_ref  )
        self.newData.value = True

        return 0

    def create_MIP_asynchronous(self , newData , newResult , running ,dataIn , dataOut ) :

        while running.value:
            # Checking if new data is available to trigger the asynchronous MPC
            if newData.value:

                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                newData.value = False

                configs , gait_in , o_v_ref , current_contacts = self.decompress_dataIn(dataIn)
                
                
                with self.dataIn.get_lock():
                    if self.dataIn.iteration == 0 :
                        loop_planner = SurfacePlanner(self.environment_URDF , self.T_gait )

                # Potential surfaces : surfaces_vertices , surfaces_inequality
                # Results MIP : - surface_indices 
                #               - all_feet_pos
                #               - success
                surfaces_vertices , surfaces_inequality , surfaces_indices  , all_feet_pos , success = loop_planner.run(configs, gait_in, current_contacts, o_v_ref)                

                with self.dataIn.get_lock():
                    self.dataIn.iteration += 1
                
                t1 = clock()
                self.compress_dataOut( surfaces_vertices ,surfaces_inequality ,  surfaces_indices  , all_feet_pos , success ) 
                t2 = clock()
                print("TIME COMPRESS DATA [ms] :  " , 1000 *(t2 - t1) )

                # Set shared variable to true to signal that a new result is available
                newResult.value = True

        return 0

    def compress_dataIn(self, configs, gait_in, current_contacts, o_v_ref) :
       
        with self.dataIn.get_lock():
            
            for i,config in enumerate(configs) : 
                dataConfig =  np.frombuffer(self.dataIn.configs[i])
                dataConfig[:] = config[:]

            gait = np.frombuffer(self.dataIn.gait).reshape((N_gait,4))
            gait[:,:] = gait_in

            o_vref =  np.frombuffer(self.dataIn.o_vref)  
            o_vref[:] = o_v_ref[:,0]

            contact = np.frombuffer(self.dataIn.contacts).reshape( (3,4) )
            contact[:,:] = current_contacts[:,:]                   

        return 0

    def decompress_dataIn(self,dataIn) :

        with dataIn.get_lock():
        
            configs = [np.frombuffer(config) for config in dataIn.configs]  

            gait =  np.frombuffer(self.dataIn.gait).reshape((N_gait,4))    

            o_v_ref =  np.frombuffer(self.dataIn.o_vref).reshape((6,1))

            contacts = np.frombuffer(self.dataIn.contacts).reshape( (3,4) )
         
        return configs , gait , o_v_ref , contacts
    
    def compress_dataOut(self, surfaces_vertices , surfaces_inequality , surfaces_indices  , all_feet_pos , success ) :
        
        # Modify this
        nvertices = 4
        if EQ_AS_INEQ :
            nrow = nvertices + 2
        else : 
            nrow = nvertices + 1

        with self.dataOut.get_lock():

            # Compress potential surfaces :
            for index_phase , phase in enumerate(surfaces_inequality) :
                for index_foot , surfaces in enumerate(phase.S) :
                    for index_sf , sf in enumerate(surfaces) :
                        A = np.frombuffer(self.dataOut.potentialSurfaces[index_phase][index_foot][index_sf].A ).reshape( (nrow , 3) )
                        b = np.frombuffer(self.dataOut.potentialSurfaces[index_phase][index_foot][index_sf].b )
                        vertices = np.frombuffer(self.dataOut.potentialSurfaces[index_phase][index_foot][index_sf].vertices ).reshape( (nvertices , 3) )
                        A[:,:] = sf[0][:,:]
                        b[:] = sf[1][:]
                        vertices[:,:] = surfaces_vertices[index_phase][index_foot][index_sf].T[:,:]
                        self.dataOut.potentialSurfaces[index_phase][index_foot][index_sf].on = True
                    
                    for i in range(index_sf + 1 , N_POTENTIAL_SURFACE ):
                        self.dataOut.potentialSurfaces[index_phase][index_foot][i].on = False

            if success : 

                self.dataOut.success = True

                # Compress selected surfaces
                for index_phase , phaseid in enumerate(surfaces_indices) :

                    for index_foot , id_selected in enumerate(phaseid) :
                        A = np.frombuffer(self.dataOut.selectedSurfaces[index_phase][index_foot].A ).reshape( (nrow , 3) )
                        b = np.frombuffer(self.dataOut.selectedSurfaces[index_phase][index_foot].b )
                        vertices = np.frombuffer(self.dataOut.selectedSurfaces[index_phase][index_foot].vertices ).reshape( (nvertices , 3) )
                        A[:,:] = surfaces_inequality[index_phase].S[index_foot][id_selected][0][:,:]
                        b[:] = surfaces_inequality[index_phase].S[index_foot][id_selected][1][:]
                        vertices = surfaces_vertices[index_phase][index_foot][id_selected].T[:,:]
                        self.dataOut.selectedSurfaces[index_phase][index_foot].on = True

                # Compress feet : TODO
                # all_feet_pos : list of optimized position such as : [[Foot 1 next_pos, None , Foot1 next_pos] , [Foot 2 next_pos, None , Foot2 next_pos] ]
                # for index_phase , phase in enumerate(all_feet_pos) :
                #     arr_feet = np.frombuffer( self.dataOut.all_feet[index_phase] ).reshape((3,4))
            
            else : 

                self.dataOut.success = False

        return 0


    def update_latest_results(self) :
        ''' Update latest results : 2 list 
        - potential surfaces : used if MIP has not converged
        - selected_surfaces : surfaces selected for the next phases
        ''' 
        if self.multiprocessing :      

            with self.dataOut.get_lock():

                if self.newResult.value :
                    self.newResult.value = False

                    self.potential_surfaces.clear()
                    self.selected_surfaces.clear()

                    self.potential_surfaces = [[ [SurfaceData(np.array(sf.A) , np.array(sf.b) , np.array(sf.vertices) )
                                                                                    for index_sf , sf in enumerate(surfaces) if sf.on == True]     \
                                                                                    for index_foot,surfaces in enumerate(phase)]  \
                                                                                    for index_phase , phase in enumerate(self.dataOut.potentialSurfaces) ]  

                    if self.dataOut.success : 
                        self.mip_success = True
                        self.mip_iteration += 1 

                        self.selected_surfaces = [ [SurfaceData(np.array(surface.A) , np.array(surface.b) , np.array(surface.vertices) )
                                                                                        for index_foot,surface in enumerate(phase)]  \
                                                                                    for index_phase , phase in enumerate(self.dataOut.selectedSurfaces) ]                        

                        # self.all_pos_feet  =     TODO
                    else :
                        self.mip_success = False
                        self.mip_iteration += 1

                else : 
                    # TODO : So far, only the convergence or not of the solver has been taken into account,
                    # What if the solver take too long ? 
                    pass  
        else : 
            # Results have been already updated 
            pass

        return  0

    def stop_parallel_loop(self):
        """Stop the infinite loop in the parallel process to properly close the simulation
        """

        self.running.value = False

        return 0




    