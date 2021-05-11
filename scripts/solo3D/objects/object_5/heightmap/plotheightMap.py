import os 
import time 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import hppfcl
import time
from scipy.spatial import ConvexHull

from solo3D.tools.Surface import Surface
from solo3D.tools.collision_tool import simple_object , fclObj_trimesh , gjk , intersect_line_triangle , get_point_intersect_line_triangle
import pickle

from scipy.spatial import ConvexHull
import trimesh



t_init = time.clock()

Nx = 400
Ny = 400
x = np.linspace(-0.4,2.5,Nx)
y = np.linspace(-2.5,1.,Ny)

# To Edit : 
object_ = "object_5"
path_object = "solo3D/objects/" + object_ + "/"
path_mesh = "solo3D/objects/" + object_ + "/meshes/"
path_saving = "solo3D/objects/" + object_ + "/heightmap/"
simple_stairs = "stair_rotation_5cm.stl"   # Full .stl file

print("\n Object selected : " , simple_stairs)
print("Folder for the selected object : " , object_)
print("X bounds : [ " , x[0] , " ; " , x[-1] , " ]" ,  "          Number of points : " , Nx  )
print("Y bounds : [ " , y[0] , " ; " , y[-1] , " ]" ,  "          Number of points : " , Ny , "\n"  )


# x,y grid
xv , yv = np.meshgrid(x,y,sparse=False , indexing = 'ij')

# Load from the saved files to check 
#Plot the 2 surfaces 
ax = a3.Axes3D(plt.figure())
with open(path_saving + "heightMap.dat" , "rb") as f :
    hm = pickle.load(f)
with open(path_saving + "surfaces.dat" , "rb") as g :
    Sf = pickle.load(g)

next_ft = np.array([0.67, -0.1345 ,0.     ])

for surface in Sf :
            surface.margin = 0.05
            surface.compute_inner_inequalities()
            surface.compute_inner_vertices()    

id_surface = None
for id_sf,sf in enumerate(Sf) :
    if sf.isInsideIneq(next_ft) :
        id_surface = id_sf

print("id_surface : " , id_surface)

for surface in Sf :
    surface.margin = 0.05
    surface.compute_inner_inequalities()
    surface.compute_inner_vertices()
    pc1 = a3.art3d.Poly3DCollection([surface.vertices],  facecolor="grey", 
                                   edgecolor="k", alpha=0.2)

    pc2 = a3.art3d.Poly3DCollection([surface.vertices_inner],  facecolor="r", 
                                   edgecolor="k", alpha=1.)


    ax.add_collection3d(pc2)
    ax.add_collection3d(pc1)

plt.plot(next_ft[0] , next_ft[1] , next_ft[2]  , "o" )

ax.set_zlim([0,2])
ax.set_xlim([-2,2])
ax.set_ylim([-3,5])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.dist=10
ax.azim=125
ax.elev=20

fig = plt.figure()
ax = fig.gca(projection = "3d")
ax.plot_surface(xv[:,:],yv[:,:],hm[:,:,0])


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim([0,2])
ax.dist=7
ax.azim=116
ax.elev=18
plt.show()

plt.show()


