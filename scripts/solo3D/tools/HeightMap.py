import numpy as np
from solo3D.tools.Surface import Surface
import pickle

class HeightMap():

    def __init__(self , path = "solo3D/heightmap/" , margin = 0.01) :
        ''' Load the heighMap and compute the distance margin for edges
        Args :
        - path (str) : path where the files are located
        - margin (float) : margin distance to avoid edges of surfaces

        Paremeters :
        self.heigtMap : 3D ,matrix heightMap[idx,idy] --> [height , Surface associated]
                        if no surface height = np.nan , -99 
        self.Surfaces : list of surfaces
        '''

        self.path = path
        self.margin = margin
        self.heightMap , self.Surfaces = self.load_data()

        for surface in self.Surfaces :
            surface.margin = margin
            surface.compute_inner_inequalities()
            surface.compute_inner_vertices()    

        #TODO improve with heighmap
        # Find a way to automatically generate bounds for x and y
        self.Nx = self.heightMap.shape[0]
        self.Ny = self.heightMap.shape[1]
        self.x_bounds = [-2. , 4.]
        self.y_bounds = [-2. , 1.]
        self.x = np.linspace(-2.0,4.0,self.Nx)
        self.y = np.linspace(-2.0,1.0,self.Ny)



    def load_data(self):
        try :
            name = self.path + "heightMap.dat"
            with open(name , "rb") as f :
                hm = pickle.load(f)

            name = self.path + "surfaces.dat"            
            with open(name , "rb") as g :
                Sf = pickle.load(g)
            
        except :
            hm = []
            Sf = []
            print("Error : heighmap not loaded")

        return hm , Sf 

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
                return height , self.Surfaces[int(id_surface)]
            else :
                return 0. , None
        
        else :
            return 0. , None 
