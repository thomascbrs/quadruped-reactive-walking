import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import hppfcl
from time import perf_counter as clock

from solo3D.tools.Surface import Surface
from solo3D.tools.collision_tool import simple_object, fclObj_trimesh, gjk, intersect_line_triangle, get_point_intersect_line_triangle
import pickle

from scipy.spatial import ConvexHull
import trimesh
import yaml

t_init = clock()

Nx = 500
Ny = 300
x = np.linspace(-2., 4., Nx)
y = np.linspace(-2., 1., Ny)

# To Edit :
object_ = "object_1"
path_object = "solo3D/objects/" + object_ + "/"
path_mesh = "solo3D/objects/" + object_ + "/meshes/"
path_saving = "solo3D/objects/" + object_ + "/heightmap/"
simple_stairs = "simple_stairs.stl"   # Full .stl file

print("\n Object selected : ", simple_stairs)
print("Folder for the selected object : ", object_)
print("X bounds : [ ", x[0], " ; ", x[-1], " ]",  "          Number of points : ", Nx)
print("Y bounds : [ ", y[0], " ; ", y[-1], " ]",  "          Number of points : ", Ny, "\n")


# Creation of the meshes folder & saving
if not os.path.exists(path_mesh):
    os.makedirs(path_mesh)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# separate file into multiples .stl for pybullet
mesh = trimesh.load_mesh(path_object + simple_stairs)

# Apply translation
translation = np.array([0.5, -1.2, 0.])
mesh.apply_translation(translation)

print("Transformation, translation : ", translation)

meshsplit = mesh.split()
for id, elt in enumerate(meshsplit):
    txt = path_mesh + "object_" + str(id) + '.stl'
    mesh = elt.copy()
    mesh.export(txt)

# From the separate .stl --> same result with elt in meshsplit
List_object = []
for elt in os.listdir(path_mesh):
    name_object = path_mesh + elt
    obj = trimesh.load_mesh(name_object)
    List_object.append(obj)

# height map + associated surface
Z = np.zeros((Nx, Ny, 2))
Surfaces = []
marge = 0.1
epsilon = 10e-3   # used to know wheter the point lies in the surface

# Simple segment modified at each loop with new x,y from grid
obj2 = simple_object()

# For each x,y , the maximum height found that intersect an object
z_max = 0.

# x,y grid
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

print("Creating height map...")

for i in range(Nx):

    if i % 10 == 0:
        print(100*i/Nx, " %")

    for j in range(Ny):
        # Create segment [x,y,-1] and [x,y,10.]
        p1 = np.array([xv[i, j], yv[i, j], -1.])
        p2 = p1 + np.array([0., 0., 10.])

        obj2.vertices = np.array([p1, p2])
        obj2.faces = [[0, 1, 0]]

        # res contains the intersected point
        # Multiple surfaces can be intersected, the higher is kept
        L_vertices = []

        z_max = 0.
        id_surface = None
        id_obj_int = 0.
        id_face_int = 0.

        # Check for each solid if it intersect the segment with hppcl
        for id_obj, obj in enumerate(List_object):

            o1 = fclObj_trimesh(obj)
            o2 = fclObj_trimesh(obj2)

            t1 = hppfcl.Transform3f()
            t2 = hppfcl.Transform3f()

            gg = gjk(o1, o2, t1, t2)

            if gg.distance < 0:
                # if intersection, check which face intersect the segment
                # and get the position of the intersection point
                for id_face, face in enumerate(obj.faces):
                    # Avoid check faces with normal vector in X,Y plan
                    # but wrong idx for some faces
                    # if rec.face_normals[idx][2] != 0 :
                    q1 = np.array(obj.vertices[face[0]])
                    q2 = np.array(obj.vertices[face[1]])
                    q3 = np.array(obj.vertices[face[2]])

                    if intersect_line_triangle(p1, p2, q1, q2, q3):
                        inter_pt = get_point_intersect_line_triangle(p1, p2, q1, q2, q3)
                        if inter_pt[2] > z_max:
                            # found a new surface higher than the previous one for index i,j
                            z_max = inter_pt[2]
                            id_obj_int = id_obj
                            id_face_int = id_face

        if z_max != 0:
            Z[i, j, 0] = z_max

            object_int = List_object[id_obj_int]
            q1 = np.array(object_int.vertices[object_int.faces[id_face_int][0]])
            q2 = np.array(object_int.vertices[object_int.faces[id_face_int][1]])
            q3 = np.array(object_int.vertices[object_int.faces[id_face_int][2]])

            if intersect_line_triangle(p1, p2, q1, q2, q3):
                inter_pt = get_point_intersect_line_triangle(p1, p2, q1, q2, q3)

            id_surface = None
            # Is the surface intersected exist ?
            # yes --> take the number of the surface
            # no --> look for all point of the object defining the surface and create one, new index

            for id_sf, surface in enumerate(Surfaces):
                if surface.isPointInside(inter_pt, epsilon=10e-3):
                    id_surface = id_sf

            if id_surface is None: # create new surface
                # List of vertices in the plan
                L_vertices = []

                # index of the vertices in the face intersected
                id_1 = object_int.faces[id_face_int][0]
                id_2 = object_int.faces[id_face_int][1]
                id_3 = object_int.faces[id_face_int][2]

                L_vertices.append(q1)
                L_vertices.append(q2)
                L_vertices.append(q3)

                for id_vert, vert_3d in enumerate(object_int.vertices):
                    if id_vert != id_1 and id_vert != id_2 and id_vert != id_3:
                        # Mixed product = 0 --> M in surface
                        M1 = object_int.vertices[id_vert]
                        if abs(np.dot(q1-M1, np.cross(q1-q2, q1-q3))) < 0.001:
                            L_vertices.append(np.array(M1))

                id_surface = len(Surfaces)
                Surfaces.append(Surface(vertices=np.array(L_vertices), margin=marge))
                Surfaces[-1].compute_all_inequalities()

            Z[i, j, 1] = id_surface  # surface accesible at Surfaces[id_surface]
        else:
            Z[i, j, 0] = z_max
            Z[i, j, 1] = None


print("Height map created")
t_end = clock()
print("Time [s] : ", t_end - t_init)

# Saving
print("Saving ...")
with open(path_saving + "heightMap.dat", "wb") as g:
    pickle.dump(Z, g)

with open(path_saving + "surfaces.dat", "wb") as f:
    pickle.dump(Surfaces, f)

# Load from the saved files to check

with open(path_saving + "heightMap.dat", "rb") as f:
    hm = pickle.load(f)
with open(path_saving + "surfaces.dat", "rb") as g:
    Sf = pickle.load(g)


dict = {'X_bound_lower': float(x[0]), 'X_bound_upper': float(x[-1]), 'Y_bound_lower': float(y[0]), 'Y_bound_upper': float(y[-1]), 'Nx': Nx, 'Ny': Ny}
with open(path_object + 'heightMap_bounds.yaml', 'w') as file:
    doc = yaml.dump(dict, file)

# Plot the 2 surfaces
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


for surface in Sf:
    surface.margin = 0.05
    surface.compute_inner_inequalities()
    surface.compute_inner_vertices()
    pc1 = a3.art3d.Poly3DCollection([surface.vertices],  facecolor="grey",
                                    edgecolor="k", alpha=0.2)

    pc2 = a3.art3d.Poly3DCollection([surface.vertices_inner],  facecolor="r",
                                    edgecolor="k", alpha=1.)

    ax.add_collection3d(pc2)
    ax.add_collection3d(pc1)

ax.set_zlim([0, 2])
ax.set_xlim([-2, 2])
ax.set_ylim([-3, 5])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.dist = 10
ax.azim = 125
ax.elev = 20

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xv[:, :], yv[:, :], hm[:, :, 0])


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim([0, 2])
ax.dist = 7
ax.azim = 116
ax.elev = 18
plt.show()

plt.show()
