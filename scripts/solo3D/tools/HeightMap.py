import numpy as np
from solo3D.tools.Surface import Surface
import pickle
import yaml

from time import perf_counter as clock

class HeightMap():

    def __init__(self , object_stair , margin = 0.01) :
        ''' Load the heighMap and compute the distance margin for edges
        Args :
        - object_stair (int) : Stair object number
        - margin (float) : margin distance to avoid edges of surfaces

        Paremeters :
        self.heigtMap : 3D ,matrix heightMap[idx,idy] --> [height , Surface associated]
                        if no surface height = np.nan , -99 
        self.Surfaces : list of surfaces
        '''

        self.object_stair = object_stair
        self.path = "solo3D/objects/object_" + str(object_stair) + "/heightmap/"
        self.margin = margin

        self.heightMap , self.Surfaces , self.bounds = self.load_data()

        # Update inner surface 
        for surface in self.Surfaces :
            surface.margin = margin
            t1 = clock()
            surface.compute_inner_inequalities()
            surface.compute_inner_vertices()    
            t2 = clock()
            print("TIME COMPUTE INNER : " , 1000*(t2 - t1))
        # Get bounds for x and y
        self.Nx = int(self.bounds.get('Nx'))
        self.Ny = int(self.bounds.get('Ny'))

        self.x_bounds = [self.bounds.get('X_bound_lower'), self.bounds.get('X_bound_upper')]  
        self.y_bounds = [self.bounds.get('Y_bound_lower'), self.bounds.get('Y_bound_upper')] 
        self.x = np.linspace(self.x_bounds[0],self.x_bounds[1],self.Nx)
        self.y = np.linspace(self.y_bounds[0],self.y_bounds[1],self.Ny)


    def load_data(self):
        try :
            name = self.path + "heightMap.dat"
            with open(name , "rb") as f :
                hm = pickle.load(f)

            name = self.path + "surfaces.dat"            
            with open(name , "rb") as g :
                Sf = pickle.load(g)
            
            name = "solo3D/objects/object_" + str(self.object_stair) + '/heightMap_bounds.yaml'
            with open(name , 'r') as file:
                bounds = yaml.load(file, Loader=yaml.FullLoader)
            
        except :
            hm = []
            Sf = []
            bounds = []
            print("Error : heighmap not loaded")

        return hm , Sf , bounds

    def find_nearest(self, xt , yt):
        ''' Find the nearest index for x,y in the list that discretize the heighmap
        TODO Find a better way to do that

        Returns : (bool) , (int) , (int)   = isInside , idx , idy
        '''
        if xt >= self.x_bounds[0] and xt <= self.x_bounds[1] and yt >= self.y_bounds[0] and yt <= self.y_bounds[1] :
            idx = (np.abs(self.x - xt)).argmin()
            idy = (np.abs(self.y - yt)).argmin()  

            return True , idx, idy
        
        else : 
            return False , -99 , -99
    
    def getHeight(self, xt , yt) :
        ''' Return the height of the couple (x,y) and the associated surface object
        nan 
        TODO find a better way for that

        Returns : (float) , (Surface object)   = height , Surface object or None if not
        '''

        isInside , idx , idy = self.find_nearest(xt,yt)

        if isInside : 

            height , id_surface = self.heightMap[idx,idy]

            if not(np.isnan(id_surface) ) :
                return height , int(id_surface)
            else :
                return 0. , None
        
        else :
            return 0. , None 

