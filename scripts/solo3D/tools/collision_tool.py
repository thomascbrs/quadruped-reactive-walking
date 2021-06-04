import numpy as np
import time
import hppfcl

# method to get collision with hppfcl

class simple_object :
    # Dummy trimesh object 

    def __init__(self , vertices = None , faces = None ) :
        self.vertices = vertices
        self.faces = faces 

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
    t1  = time.time()
    shape.set(s1, s2, tf1, tf2)
    gjk = hppfcl.GJK(150, 1e-8)
    status = gjk.evaluate(shape, guess, support_hint)
    t2  = time.time()
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
        

# Method to intersect 2 segment --> useful to retrieve which inequality is crossed in a surface (2D)
def get_intersect_segment(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

  
# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
    # Modified, remove class Point, directly handle p = [px,py]
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Colinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect_segment(p1,q1,p2,q2):
    '''
    Args : p (x2 array) : px , py
    '''
    
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False