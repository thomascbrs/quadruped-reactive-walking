import numpy as np
import time
import hppfcl

# method to get collision with hppfcl

def fclObj_trimesh(obj): 
    # used to get fcl object from the .stl and trimesh parser
    verts = hppfcl.StdVec_Vec3f ()
    faces = hppfcl.StdVec_Triangle ()
    verts.extend(obj.vertices)
    [faces.append(hppfcl.Triangle(int(el[0]),int(el[1]),int(el[2]) ))  for el in obj.faces ]
    return hppfcl.Convex(verts, faces)


def gjk(s1,s2, tf1, tf2):
    guess = np.array([1., 0., 0.])
    support_hint = np.array([0, 0], dtype=np.int32)

    shape = hppfcl.MinkowskiDiff()
    t1  = time.clock()
    shape.set(s1, s2, tf1, tf2)
    gjk = hppfcl.GJK(150, 1e-8)
    status = gjk.evaluate(shape, guess, support_hint)
    t2  = time.clock()
    return  gjk 

# Methods to intersect triangle and segment 
def signed_tetra_volume(a,b,c,d):
    return np.sign(np.dot(np.cross(b-a,c-a) , d-a)/6.0)


def intersect_line_triangle(q1,q2,p1,p2,p3) :
    s1 = signed_tetra_volume(q1,p1,p2,p3)
    s2 = signed_tetra_volume(q2,p1,p2,p3)
    
    if s1 != s2 : 
        s3 = signed_tetra_volume(q1,q2,p1,p2)
        s4 = signed_tetra_volume(q1,q2,p2,p3)
        s5 = signed_tetra_volume(q1,q2,p3,p1)
        
        if s3 == s4 and s4 == s5 :
            return True
        else : 
            return False
    else :
        return False

def get_point_intersect_line_triangle(q1,q2,p1,p2,p3) :
    s1 = signed_tetra_volume(q1,p1,p2,p3)
    s2 = signed_tetra_volume(q2,p1,p2,p3)
    
    if s1 != s2 : 
        s3 = signed_tetra_volume(q1,q2,p1,p2)
        s4 = signed_tetra_volume(q1,q2,p2,p3)
        s5 = signed_tetra_volume(q1,q2,p3,p1)
        
        if s3 == s4 and s4 == s5 :
            n = np.cross(p2-p1,p3-p1)
            t = np.dot(p1 - q1, n) / np.dot(q2 - q1 , n)            
            return q1 + t * (q2 - q1)
        else : 
            return np.zeros(3)
    else :
        return np.zeros(3)