import numpy as np
import trimesh
from collision_tool import *

simple_stairs = "simple_stairs.stl"

class simple_object :

    def __init__(self , vertices = None , faces = None ) :
        self.vertices = vertices
        self.faces = faces 


# Load object from .stl using trimesh library
# simple stairs --> modified stairs of bauzil room 
mesh = trimesh.load_mesh(simple_stairs)
meshsplit = mesh.split()

# Number of point x, y 
Nx = 100
Ny = 100
x = np.linspace(-2.0,2.0,Nx)
y = np.linspace(-1.0,2.0,Ny)
Z = np.zeros((Nx,Ny))

# Simple segment modified at each loop with new x,y from grid
obj2 = simple_object()

# For each loop, list of intersection point between segment and objects
res = []

# x,y grid
xv , yv = np.meshgrid(x,y,sparse=False , indexing = 'ij')

print("Creating height map...")

for i in range(Nx) :
    
    if i%10 == 0 :
        print(100*i/Nx , " %")
    
    for j in  range(Ny) :         
        
        # Create segment [x,y,-1] and [x,y,10.] 
        
        p1 = np.array([xv[i,j],yv[i,j],-1.])
        p2 =  p1 + np.array([0.,0.,10.])
        
        obj2.vertices = np.array([p1 , p2 ])
        obj2.faces = [[0,1,0]]
        
        res.clear()
        
        # Check for each solid if it intersect the segment with hppcl
        for obj in meshsplit :
            
            o1 = fclObj_trimesh(obj)
            o2 = fclObj_trimesh(obj2) 

            t1 = hppfcl.Transform3f()
            t2 = hppfcl.Transform3f()

            gg = gjk(o1,o2,t1,t2)
        
            if gg.distance < 0 :                                 
                # if intersection, check which face intersect the segment 
                # and get the position of the intersection point
                
                for idx, face in enumerate(obj.faces) :
                    
                    # Avoid check faces with normal vector in X,Y plan
                    # but wrong idx for some faces 
                    #if rec.face_normals[idx][2] != 0 :                    
                    
                    q1 = np.array(obj.vertices[face[0]])
                    q2 = np.array(obj.vertices[face[1]])
                    q3 = np.array(obj.vertices[face[2]])

                    if intersect_line_triangle(p1 , p2 , q1 , q2 , q3 ) :
                        res.append(get_point_intersect_line_triangle(p1 , p2 , q1 ,q2 , q3))
                        
                
        if len(res) !=0 : 
            Z[i,j] = np.max(np.array(res)[:,2])
        else :
            Z[i,j] = 0.
        
print("Height map created") 

# Save the map in .npy
np.save("heightMap_z.npy" , Z)

