# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:06:53 2021

@author: LinhTruongHong
"""
import numpy as np

# Compare/searching
def inNd(a, b, assume_unique=False):

    # Check if a or b is not 1 dimension
    if a.ndim == 1:
        a = a.reshape(-1, len(a))

    if b.ndim == 1:
        b = b.reshape(-1, len(b))

    a = np.asarray(a, order='C')
    b = np.asarray(b, order='C')
    a = a.ravel().view((np.void, a.dtype.itemsize*a.shape[1]))
    b = b.ravel().view((np.void, b.dtype.itemsize*b.shape[1]))

    return np.in1d(a, b, assume_unique)
# In[Coordinate system]
def cart2pol(x, y):
    """
    This function is to translate 2D points in Cartersian coordinate system to polar coordinate system
    Parameters:
    ------------------------
        x                               : x-coordinate
        y                               : y-coordinate

    Return:
    ------------------------
        rho                             : radius
        theta                           : angle
    Demo:
    ------------------------
    """
    # Cal rho and theta
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Convert [-pi pi] to [0 - 2pi]
    if np.isscalar(theta):
        if theta < 0:
            theta = theta + 2.0*np.pi
    else:
        mask = np.where((theta < 0) == True)[0]
        theta[mask] = theta[mask] + 2.0*np.pi

    return rho, theta

# In[]
def pol2cart(rho, theta):
    """
    This function is to translate 2D points in polar coordinate system to Cartersian coordinate system
    Parameters:
    ------------------------
        rho                             : radius
        theta                           : angle
    Return:
    ------------------------
        x                               : x-coordinate
        y                               : y-coordinate

    Demo:
    ------------------------
    """
    # Copy data
    rho = np.asarray(rho)
    theta = np.asarray(theta)

    # Call x, and t
    if (theta.size > 1) and (rho.size == 1):
        rho = np.full(theta.size[0], rho)
    elif (rho.size > 1) and (theta.size == 1):
        theta = np.full(rho.size[0], theta)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y
def vector_length(pt0):
    """This function is to calculate the length of a vector

    Parameters:
    ------------------------
        pt0                 : vector 0

    Return:
    ------------------------
        vector length       : new objects

    Demo:
        pt0 = np.array([[1,2,4],[1,3,4]])
    ------------------------
    """
    # Check input data
    pt0 = np.asarray(pt0, dtype = float)
    if pt0.ndim == 1:
        return np.sqrt(np.sum(pt0*pt0))
    else:
        return np.sqrt(np.sum(pt0*pt0, axis = 1))

def vector_unit(pt0):
    """This function is to create unit vector
    Parameters:
    ------------------------
        pt0                 : vector

    Return:
    ------------------------
        vector length       : new objects

    Demo:
        pt0 = np.array([1,2,4], dtype = float)
        pt0 = np.array([[1,2,4],[1,3,4]], dtype = float)
    ------------------------
    """
    # Cal vector lengths
    vect_len = vector_length(pt0)
    if pt0.ndim == 1:
        return pt0/vect_len
    else:
        return np.divide(pt0, np.tile(vect_len, (pt0.shape[1], 1)).T)

def orthogonal_2Dvector(pt0):
    """This function is to create orthogonal vector of a 2D vector
    Parameters:
    ------------------------
        pt0                 : vector

    Return:
    ------------------------
        pt0                 : a orthogonal vector

    Demo:
        pt0 = np.array([1,2,4], dtype = float)
        pt0 = np.array([[1,2,4],[1,3,4]], dtype = float)
    ------------------------
    """
    # Get othorginal vector
    if pt0.ndim == 1:
        return np.array([-pt0[1], pt0[0]])
    elif pt0.ndim == 2:
        if pt0.shape[1] > 2:
            print('The method only support 2D vectors and first two columns are used')
            pt0 = pt0[:,:2]
            pt0 = np.fliplr(pt0)
            pt0[:, 0] = -pt0[:,0]
            return pt0
    else:
        raise ValueError("This function only support 2D vector")
        return
# In[Calculate cosin between two vectors]
def cosin_vectors(vect1, vect2):
    """
    This function is to compute a cosin between 2 vectors
    Parameters:
    ------------------------
        vect1                               : [Nx3 or Nx2], x-, y-, z-
        vect2                               : [Nx3 or Nx2], x-, y-, z-

    Return:
    ------------------------

    Demo:
    vect1 = np.array([[1.,2., 3.], [2.,3., 4.]])
    vect2 = np.array([[2.,2., 3.], [1.,3., 4.]])

    vect1 = np.array([[1.,2.], [2.,3.]])
    vect2 = np.array([[2.,2.], [1.,3.]])

    """

    # Check input variables
    vect1 = np.array(vect1, dtype = np.float32, copy=True)
    if vect1.ndim == 1:
        vect1 = vect1.reshape(-1, len(vect1))

    vect2 = np.array(vect2, dtype=np.float32, copy=True)
    if vect2.ndim == 1:
        vect2 = vect2.reshape(-1, len(vect2))
    # Check size of vectors
    if (vect1.shape[1] != vect2.shape[1]):
        raise ValueError("Vectors are incompatile. They must the same size")
        return

    cosin2Vect = np.sum(vect1*vect2, axis = 1)

    return cosin2Vect.flatten()

# In[Rotation between two vectors]
def rot_vectors(vect_1, vect_2):
    ''' This function is to identify the rotation matrix between two vectors in 3D space
    Ref: see vrrtvec.m and vrrotvec2mat.m

    Parameters:
    ------------------------
        vect1                               : [Nx3 or Nx2], x-, y-, z-
        vect2                               : [Nx3 or Nx2], x-, y-, z-

    Return:
    ------------------------

    Demo:
    vect_1 = copy.deepcopy(vect_1)
    vect_2 = copy.deepcopy(vect_2)

    vect_1 = np.array([0.,0.,1.])
    vect_2 = np.array([-1,0.,0.])
    vect_1 = rect_long_dir
    vect_2 = np.array([1.0, 0.0, 0.0], dtype = np.float64)


    '''
    # Check input vectors
    vect_1 = np.asarray(vect_1, dtype = float)
    if vect_1.shape[0] != 3:
        raise ValueError('Vector vect1 {} must be [1x3]'.format(vect_1))

    vect_2 = np.asarray(vect_2, dtype = float)
    if vect_2.shape[0] != 3:
        raise ValueError('Vector vect1 {} must be [1x3]'.format(vect_2))

    # Normalization
    vect_1 = vector_unit(vect_1)
    vect_2 = vector_unit(vect_2)

    # Estimate a rotation vector
    vect_3 = np.cross(vect_1, vect_2)
    if vector_length(vect_3) <= 1e-3:
        ax = np.array([0.0,0.0,0.0])
    else:
        ax = vector_unit(np.cross(vect_1, vect_2))
    angle = np.arccos(np.dot(vect_1, vect_2))

    if not np.any(ax):
        abs_vect_1 = np.abs(vect_1)
        mask = np.argmin(abs_vect_1)
        c = np.zeros((1,3))
        c[0,mask] = 1
        ax = vector_unit(np.cross(vect_1, c)).flatten()

    # Rotation maxtrix
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    x = ax[0]
    y = ax[1]
    z = ax[2]
    rot_matrix = np.full((3,3), np.inf)
    rot_matrix[0] = [t*x*x + c, t*x*y - s*z, t*x*z + s*y]
    rot_matrix[1] = [t*x*y + s*z, t*y*y + c, t*y*z - s*x]
    rot_matrix[2] = [t*x*z - s*y, t*y*z + s*x, t*z*z + c]

    return rot_matrix
# In[]
def proj_3Dpoints_3Dline(ptc, line):
    """This function is to project 3D points onto 3D line
    Parameters:
    ------------------------
        ptc           : [Nx3] - x, y, z coordinates
        line          : [1x6] - P0 (x, y, z), t (tx, ty, tz)
    Return:
    ------------------------
        proj_ptc      : [Nx3]
    Demo:
        ptc = web_end_ptc.copy()
        line = np.hstack((web_ptc_center, web_largest_tangent_vector))

        ptc = np.array([0, 3, 0])
        line = np.array([0,0,0, 1,1,0])


    """
    p0 = line[:3]
    vl = line[3:]
    w = ptc - p0
    b = np.dot(w, vl)/np.dot(vl, vl)
    proj_ptc = p0 + b.reshape(-1, 1)*vl

    return proj_ptc
#%%
def proj_3Dpoints_plane(ptc, surface):
    """This function is to porject points onto a surface
    Parameters:
    ------------------------
        ptc                 : [Nx3
        surface             : [1x6]

    Return:
    ------------------------
        ptc0                : a orthogonal vector

    Demo:

    """
    # Get surface
    surface = np.asarray(surface).flatten()
    surface_cent = surface[:3]
    surface_normal = surface[3:]

    # Project points onto the surface
    c = np.dot(surface_normal, surface_cent)
    a = np.sum(ptc*surface_normal, axis = 1) - c
    ptc0 = ptc - a.reshape(-1,1)*surface_normal.reshape(1,3)

    return ptc0

# In[Dist point to plane]
def dist_3Dpoints_plane(point3D, surface_center, surface_normal):

	# Compute a distance
	dist = point3D.dot(surface_normal) - surface_normal.dot(surface_center)
	return dist
# In[Residual]
def residual(dist_ptc):
    return np.sqrt(np.sum(dist_ptc**2)/len(dist_ptc))

# In[Determine the intersection between the 3D line and 3D plane]
def line_plane_intersection(line, plane):
    """This function is to determine an intersection points between the 3D line and 3D plane
    Input
    Output
    Demo
    """
    # Retrieve line parameters
    line = line.flatten()
    line_pt = line[:3]
    line_dir = line[3:]
    
    # Retrieve plane parameters
    plane = plane.flatten()
    plane_pt = plane[:3]
    plane_norm = plane[3:]
    
    if np.abs(line_dir.dot(plane_norm)) <= np.cos(np.deg2rad(5.0)):
        # A line parallel to the plane
        pt = np.full(3, np.inf)
    else:
        # A line intersects to the plane
        s = np.dot(plane_norm, (plane_pt - line_pt))/np.dot(plane_norm, line_dir);

        pt = line_pt + s*line_dir;
    # Return results
    return pt
# In[intersection between two plane]
def plane_plane_intersection(src_plane, trg_plane, angle_threshold=np.deg2rad(5.0)):
    """
    The function to determine an intersection between two planes
    % Inputs: 
    %       src_plane: [x, y, z, nx, ny, nz] center and normal of Plane 1
    %       trg_plane: [x, y, z, nx, ny, nz] center and normal of Plane 2
    %
    % Outputs:
    %   P    is a point that lies on the interection straight line.
    %   N    is the direction vector of the straight line
    % check is an integer (0:Plane 1 and Plane 2 are parallel' 
    %                              1:Plane 1 and Plane 2 coincide
    %                              2:Plane 1 and Plane 2 intersect)
    %
    % Example:
    % Determine the intersection of these two planes:
    % 2x - 5y + 3z = 12 and 3x + 4y - 3z = 6
    % The first plane is represented by the normal vector N1=[2 -5 3]
    % and any arbitrary point that lies on the plane, ex: A1=[0 0 4]
    % The second plane is represented by the normal vector N2=[3 4 -3]
    % and any arbitrary point that lies on the plane, ex: A2=[0 0 -2]
    %[P,N,check]=plane_intersect([2 -5 3],[0 0 4],[3 4 -3],[0 0 -2]);
    %This function is written by :
    %                             Nassim Khaled
    %                             Wayne State University
    %                             Research Assistant and Phd candidate
    # Demo:
    src_plane = np.append(src_surf_p, src_surf_norm)
    trg_plane = np.append(trg_surf_p, trg_surf_norm)
    """

    #Preallocation a point on an intersection line
    P = np.array([0., 0., 0.,], dtype=np.float32)
    
    # Retrieve para.s of the plane
    src_surf_p, src_surf_norm = src_plane[:3], src_plane[3:]
    trg_surf_p, trg_surf_norm = trg_plane[:3], trg_plane[3:]
    
    #  Test if the two planes are parallel
    if np.abs(cosin_vectors(src_surf_norm, trg_surf_norm)) >= np.cos(angle_threshold): # Plane 1 and Plane 2 are near parallel norm(cross) >= xxx
        v = src_surf_p - trg_surf_p
        if np.abs(cosin_vectors(src_surf_norm, v)) <= np.cos(np.pi/2. - angle_threshold):         
            check = -1;                    # src_plane and trg_plane coincide
            intersection_line = [];
            return check, intersection_line
        else:
            check = 0;                     # src_plane and trg_plane paralell
            intersection_line = [];
            return check, intersection_line
    # The planes are intersected
    check = 1
     
    # src_plane and trg_plane intersect in a line first determine max abs coordinate of cross product
    # Compute angle between two normals
    orthogonal_vect = np.cross(src_surf_norm, trg_surf_norm)
    axis = np.argmax(np.abs(orthogonal_vect)) 
    
    # get a point on the intersection line and zero the max coord, and solve for the other two     
    d1 = -np.dot(src_surf_norm, src_surf_p)    # intercept of the src_plane equations
    d2 = -np.dot(trg_surf_norm, trg_surf_p)    # intercept of the trg_plane equations
    
    if axis == 0: 
        # intersect with x=0
        P[0] = 0.0
        P[1] = (d2*src_surf_norm[2] - d1*trg_surf_norm[2])/orthogonal_vect[0]
        P[2] = (d1*trg_surf_norm[1] - d2*src_surf_norm[1])/orthogonal_vect[0]
    elif axis == 1:
        # intersect with y=0
        P[0] = (d1*trg_surf_norm[2] - d2*src_surf_norm[2])/orthogonal_vect[1]
        P[1] = 0.0
        P[2] = (d2*src_surf_norm[0] - d1*trg_surf_norm[0])/orthogonal_vect[1]
    else:
        # intersect with z=0
        P[0] = (d2*src_surf_norm[1] - d1*trg_surf_norm[1])/orthogonal_vect[2]
        P[1] = (d1*trg_surf_norm[0] - d2*src_surf_norm[0])/orthogonal_vect[2]
        P[2] = 0.0
    # An intersection line
    intersection_line = np.append(P, orthogonal_vect/vector_length(orthogonal_vect))
    # return 
    return check, intersection_line
# In[Find enpoint of the plane on the line]
def find_end_points(ptc_xyz, plane, line):
    """This function is to fine the end points of the plane projected onto the line"
    Parameters:
    ------------------------
        ptc_xyz                 : [Nx3
        plane                   : [1x6]
        line                    : [1x6]

    Return:
    ------------------------
        end_points              : [1x6] - [P1, P2]
        line_segment            : scalar

    
    Demo:
        ptc_xyz, plane, line = src_ptc_xyz, src_plane, intersection_line
    """
    # get line
    line_p, line_t = line[:3], line[3:]
    # project the points to the plane
    proj_plane_ptc_xyz = proj_3Dpoints_plane(ptc_xyz, plane)
    
    # project points to line
    proj_line_ptc_xyz = proj_3Dpoints_3Dline(proj_plane_ptc_xyz, line)
     
    # Determine the end points
    s = np.sum((proj_line_ptc_xyz - line_p)*line_t, axis=1)
    max_id = np.argmax(s)
    min_id = np.argmin(s)
    end_points = np.append(proj_line_ptc_xyz[min_id], proj_line_ptc_xyz[max_id])
    line_segment = vector_length(end_points[:3] - end_points[3:])
      
    # return
    return end_points, line_segment
# In[Call overlap between 2 line segment]
def cal_line_segments_overlap(line_1, line_2):
    """ This function is to compute an overlap length between two line section
    Ref: http://stackoverflow.com/questions/22456517/algorithm-for-finding-the-segment-overlapping-two-collinear-segments
    
    Parameters:
    ------------------------
        line_1                             : [P1x, P1y, P1z, P2x, P2y, P2z] - source line
        line_2                             : [P1x, P1y, P1z, P2x, P2y,P2z] : target line
    Return:
    ------------------------
        overlap_length                      : scalar
    Demo:
        line_1, line_2 = src_end_points, trg_end_points
        
    """
    # Project end points of LP2 onto LP1
    line_1_p = line_1[:3]
    line_1_t = line_1[3:] - line_1[:3]
    # Points of line_2
    line_2_p = line_2.reshape(-1, 3)

    # Project end points of line_2 onto line_1
    proj_line_line_2_p = proj_3Dpoints_3Dline(line_2_p, np.append(line_1_p, line_1_t))
    s = np.sum((proj_line_line_2_p - line_1_p)*line_1_t, axis=1)
    line_2_new_p = [line_1_p + line_1_t*s[ind] for ind in range(s.shape[0])]
    line_2_new_p = np.asarray(line_2_new_p).reshape(-1,6).flatten()
    
    # Compute min and max each line segment
    # Line 1
    min_line_1 = np.min(line_1.reshape(-1, 3), axis=0)
    max_line_1 = np.max(line_1.reshape(-1, 3), axis=0)
    # Line 2
    min_line_2 = np.min(line_2_new_p.reshape(-1, 3), axis=0)
    max_line_2 = np.max(line_2_new_p.reshape(-1, 3), axis=0)
    # Determine intersection points
    min_intersection = np.max(np.vstack((min_line_1, min_line_2)), axis=0)
    max_intersection = np.min(np.vstack((max_line_1, max_line_2)), axis=0)
    
    # Check intersection
    check_intersection = ((min_intersection[0] <= max_intersection[0]) &
                          (min_intersection[1] <= max_intersection[1]) &
                          (min_intersection[2] <= max_intersection[2]))
    # Compute intersection line
    if check_intersection:
        overlap_length = vector_length(max_intersection - min_intersection)
    else:
        overlap_length = 0
    # return 
    return overlap_length
    

def min_2Dbounding_box(ptc, option = 'area'):
    """
    This function is to estimate the 2D minimum bounding box of the data
    Parameters:
    ------------------------
        ptc                                 : [Nx3], x-, y-, z-

    Return:
    ------------------------
        min_rectangle.area                  : Area
        min_rectangle.parallel_length       : Length of edge 1 - scalar
        min_rectangle.orthogonal_length     : Length of edge 2 - scalar
        min_rectangle.rectangle_center      : Center - [1x2]
        min_rectangle.rectangle_corners     : Corners [4x2] [xi,yi]
        min_rectangle.parallel_edge_vector  : a unit vector of edge 1

    Demo:
    ptc = ptc_xyz[:,:2]
    """
    # Call libs
    from scipy.spatial import ConvexHull
    # Check input variable
    ptc = np.asarray(ptc, dtype = float)
    if ptc.shape[0] < 4:
        raise ValueError('Points {:d} are less than required points {:d}'.format(ptc.shape[0], 4))
    ptc = ptc[:,:2]

    # Generate convexhull
    hull_ptc_ids = ConvexHull(ptc).vertices
    hull_vert_xy = ptc[hull_ptc_ids]
    hull_vert_xy = np.insert(hull_vert_xy, hull_vert_xy.shape[0], hull_vert_xy[0], axis = 0)

    # Find minimum rectangle
    min_rectangle = bounding_rectangle(0, hull_vert_xy)
    for i in range(1, len(hull_vert_xy)-1):
        rectangle = bounding_rectangle(i, hull_vert_xy)
        if option.casefold() == 'area'.casefold():
            if rectangle['area'] < min_rectangle['area']:
                min_rectangle = rectangle
        else:
            if (rectangle['parallel_length'] + rectangle['orthogonal_length']) < (min_rectangle['parallel_length'] + min_rectangle['orthogonal_length']):
                min_rectangle = rectangle
    # Get the minimum rectangle
    parallel_edge_angle = np.arctan2(min_rectangle['parallel_edge_vector'][1], min_rectangle['parallel_edge_vector'][0])
    orthogonal_edge_angle = parallel_edge_angle + np.pi/2.
    map_array = np.array([[np.cos(parallel_edge_angle), np.cos(orthogonal_edge_angle)],
                          [np.sin(parallel_edge_angle), np.sin(orthogonal_edge_angle)]])
    min_rectangle['rectangle_center'] = np.sum(min_rectangle['rectangle_center']*map_array, axis = 1)

    # Get the corners
    rect_corner_pts = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1.*i1):
            corner_pt = np.array([min_rectangle['rectangle_center'][0] + i1*min_rectangle['parallel_length'],
                                  min_rectangle['rectangle_center'][1] + i2*min_rectangle['orthogonal_length']])
            rect_corner_pts = np.append(rect_corner_pts, corner_pt)
    rect_corner_pts = rect_corner_pts.reshape(-1,2)

    # Rotate the rectangle corners to a global coordinate system
    diff_rect_corner_pts = rect_corner_pts - min_rectangle['rectangle_center']
    diff_angle = np.arctan2(diff_rect_corner_pts[:,1], diff_rect_corner_pts[:,0]) + parallel_edge_angle
    diff_rect_corner_length = vector_length(diff_rect_corner_pts)
    rect_corner_pts[:,0] = min_rectangle['rectangle_center'][0] + diff_rect_corner_length*np.cos(diff_angle)
    rect_corner_pts[:,1] = min_rectangle['rectangle_center'][1] + diff_rect_corner_length*np.sin(diff_angle)
    min_rectangle['rectangle_corners'] = rect_corner_pts

    return min_rectangle

def bounding_rectangle(vert_id, hull_verts):
    """
    This function is to get a rectangle where an edge is one edge of 2D polygon
    Parameters:
    ------------------------
        vert_id                             : index of vertices of a 2D polygon
        hull_verts                          : vertices of the 2D polygon - [Nx2] - [xi, yi]

    Return:
    ------------------------
        min_rectangle.area                  : Area
        min_rectangle.parallel_length       : Length of edge 1 - scalar
        min_rectangle.orthogonal_length     : Length of edge 2 - scalar
        min_rectangle.rectangle_center      : Center - [1x2]
        min_rectangle.parallel_edge_vector  : a unit vector of edge 1

    Demo:

    vert_id = 0
    hull_verts = hull_vert_xy
    """

    # Get the first edge through vert_id
    edge_vect = hull_verts[vert_id + 1] - hull_verts[vert_id]
    edge_unit_vect = vector_unit(edge_vect)
    edge_ortho_vect = orthogonal_2Dvector(edge_unit_vect)

    # Call distance from the hull vertices to the vector
    dist_verts_unit_vect = [np.dot(edge_unit_vect, pt) for pt in hull_verts]
    dist_verts_ortho_vect = [np.dot(edge_ortho_vect, pt) for pt in hull_verts]

    # Determine the vertices on the edges of the min rectangle
    vert_unit_vect = np.min(dist_verts_unit_vect)
    vert_ortho_vect = np.min(dist_verts_ortho_vect)
    len_edge_unit_vect = np.max(dist_verts_unit_vect) - vert_unit_vect
    len_edge_ortho_vect = np.max(dist_verts_ortho_vect) - vert_ortho_vect

    rect = {"area": len_edge_unit_vect*len_edge_ortho_vect,
            "parallel_length": len_edge_unit_vect,
            "orthogonal_length": len_edge_ortho_vect,
            "rectangle_center": np.array([vert_unit_vect + len_edge_unit_vect/2.0, vert_ortho_vect + len_edge_ortho_vect/2.0]),
            "parallel_edge_vector": edge_unit_vect}

    return rect


#%%
def check_pair_vectors(pt0, pt1):
    """This function is check if two vectors are compatible or not
    Parameters:
    ------------------------
        pt0                 : vector 0
        pt1                 : vector 1

    Return:
    ------------------------
        pt0                 : vector 0
        pt1                 : vector 1
        check               : True or False

    Demo:
        pt0 = np.array([[1,2,4],[1,3,4]])
        pt1 = np.array([[1,3,5],[4,5,6]])
    ------------------------
    """
    # Call libs
    # Check input data
    pt0 = np.asarray(pt0, dtype = float)
    pt1 = np.asarray(pt1, dtype = float)

    # Check dimension
    check = False
    if pt0.ndim != pt1.ndim:
        print("Imcompatible dimensions between pt0 = {:d} and pt1 = {:d}".format(pt0.ndim, pt1.ndim))
    else:
        if (pt0.ndim == 1) & (pt0.shape[0] != pt1.shape[0]):
            print("Imcompatible length between pt0 = {:d} and pt1 = {:d}".format(pt0.shape[0], pt1.shape[0]))
        elif (pt0.ndim == 2) & np.any(pt0.shape != pt1.shape):
            print("Imcompatible size between pt0 = {} and pt1 = {}".format(pt0.shape, pt1.shape))
        elif pt0.ndim >= 3:
            print("Size vectors pt0 = {:d} does not support".format(pt0.ndim, pt1.ndim))
        else:
            check = True
    return pt0, pt1, check

# In[Calculate covariance matrix]
def covariance_matrix(ptc, centroid = 'centroid'):

	ptc = np.array(ptc, dtype = np.float64, copy = True)
	ptc = np.asmatrix(ptc)

	if centroid.casefold() == 'centroid'.casefold():

		centroid = np.mean(ptc, axis=0)
		ptc = np.asmatrix(ptc - centroid)
		covar_matrix = ptc.T*ptc/len(ptc)
	else:
		covar_matrix = ptc.T*ptc

	return np.asarray(covar_matrix)

# In[Calculate a eigen vector by PCA]
def PCA_eigenspace(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    # Call libs
    from numpy import linalg as LA

    # Calculate eigenspace
    if ptc.shape[0] >= min_ptc:
        cov = covariance_matrix(ptc, centroid = 'centroid')
        w, v = LA.eig(cov)
        ids = np.argsort(w)
    	# Eigenvalues [e1 e2, e3] e1<e2<e3
        w = w[ids]
        v = v[:,ids]
    else:
        w = np.full((3,3), np.inf)
        v = np.full((3,3), np.inf)

    # Return results
    return v, w

def cov_PCA(covariance_matrix):
    """This function is to estimate a normal vector based on PCA from a covarance matrix"""
    
    # Call libs
    from numpy import linalg as LA
    
    w, v = LA.eig(covariance_matrix)
    ids = np.argsort(w)
    v = v[:,ids]
    normal_vector = v[:,0]

    return normal_vector
# In[Calculate a normal vector by PCA]
def PCA_NormalVector(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    eigen_vector,__ = PCA_eigenspace(ptc, min_ptc = min_ptc)
    normal_vector = eigen_vector[:, 0]
    # Return results
    return normal_vector
# In[Targent vector] ]
def PCA_Smallest_TangentVector(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    eigen_vector, __ = PCA_eigenspace(ptc, min_ptc = min_ptc)
    smallest_tangent_vector = eigen_vector[:, 1]

    # Return results
    return smallest_tangent_vector
# In[Targent vector] ]
def PCA_Targest_TangentVector(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    eigen_vector,__ = PCA_eigenspace(ptc, min_ptc = min_ptc)
    largest_tangent_vector = eigen_vector[:, -1]
    # Return results
    return largest_tangent_vector

# In[Robust PCA]
def wPCA(ptc, tol = 0.5*np.pi/180, max_iters = 100, min_ptc = 5, initial_normal = np.array([0.0,0.0,1.0])):
    """This function is to compute a normal vector based weight of points
    Input
    Output
    Demo

    ptc = surface_pts_xyz
    tol = 0.5*np.pi/180
    max_iters = 100
    initial_normal = None
    """

    # call libs
    import numpy.matlib as npm
    import pandas as pd

    # Check input data
    if isinstance(ptc, pd.DataFrame):
        ptc = ptc[["x", "y", "z"]].values
    elif isinstance(ptc, np.ndarray):
        if (ptc.ndim == 0) or (ptc.ndim > 2):
            raise ValueError("Expected 1D or 2D array, got %dD array instead" % ptc.ndim)
        elif ptc.ndim == 1:
            ptc = ptc.reshape(-1, len(ptc))
        else:
            ptc = ptc[:,:3]
    else:
        raise ValueError("Expected 2D array or DataFrame")

    # Convert the threshold
    tol = np.cos(tol)

    # Compute a normal vector with weight
    if ptc.shape[0] > min_ptc:

        # Initial center and normal
        ptc_center = np.mean(ptc, axis = 0)
        if (initial_normal is None):
            ptc_normal = PCA_NormalVector(ptc, min_ptc = min_ptc)
        elif len(initial_normal) < 3:
            ptc_normal = PCA_NormalVector(ptc, min_ptc = min_ptc)
        else:
            ptc_normal = initial_normal

        # Iterations
        flag = True
        iters = 0
        while flag & (iters <= max_iters):

            # Establish weight for each point based on a distance
            dist_ptc_surface = np.abs(dist_3Dpoints_plane(ptc, ptc_center, ptc_normal))

#            if any(dist_ptc_surface == 0.):
#                mask = dist_ptc_surface == 0.0
#                dist_ptc_surface[mask] = 0.001
            weight = np.exp(-(dist_ptc_surface**2)/10.)
            weight = npm.repmat(weight, 3, 1).T

            # Computer new center and normal
            w_ptc = (ptc*weight)#/np.sum(weight, axis = 1)
            ptc_new_cent = np.mean(w_ptc, axis = 0)
            ptc_new_normal = PCA_NormalVector(w_ptc, min_ptc = min_ptc)

            # Check deviation of the normal
            if np.sum(ptc_normal*ptc_new_normal) <= tol: # Angle deviation is still large
                ptc_center = ptc_new_cent
                ptc_normal = ptc_new_normal
                iters += 1
            else:
                flag = False

        # return the result
        return ptc_new_cent, ptc_new_normal
    else:

        return np.mean(ptc, axis = 0), np.full(3, np.nan)
#%%
def rPCA(ptc, tol):
    """
    This is a robust PCA using both normal and surface distance as criteria
    Parameters
    ----------
    ptc : TYPE
        DESCRIPTION.
    tol : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    ptc = surface_pts_xyz
    tol = fit_plane_tol
    """

    # Constant
    # Check input variable
    if ptc.shape[0] < tol["min_ptc"]:

        return None, None
    # Initial center and normal
    ptc_center_0 = np.mean(ptc, axis = 0)
    ptc_normal_0 = PCA_NormalVector(ptc, min_ptc = tol["min_ptc"])

    # Compute the weight and distance
    dist_p_plane = dist_3Dpoints_plane(ptc, ptc_center_0, ptc_normal_0)
    weight = cal_weight(np.abs(dist_p_plane))

    #% Interaction
    no_iter = 1
    flag = True
    while flag&(no_iter <= tol["max_itr"]):
        # Compute the new center and normal vector
        ptc_w = ptc*weight.reshape(-1,1)
        ptc_center_i = np.sum(ptc_w, axis = 0)/np.sum(weight)
        delta_cent = ptc_w - ptc_center_i
        itr_cov = delta_cent.T.dot(delta_cent)/np.sum(weight)
        ptc_normal_i = cov_PCA(itr_cov)

        # Compute the error
        itr_err_normal = np.abs(cosin_vectors(ptc_normal_0, ptc_normal_i))
        itr_err_dist = np.abs(dist_3Dpoints_plane(ptc_center_0, ptc_center_i, ptc_normal_i))

        # Check
        if ((itr_err_normal <= tol["cosin_normal"])&(itr_err_dist <= tol["distance"])):
            flag = False
        else:
            # Update the new weight
            dist_p_plane = dist_3Dpoints_plane(ptc, ptc_center_i, ptc_normal_i)
            weight = cal_weight(np.abs(dist_p_plane))
            no_iter = no_iter + 1

    # Return
    return ptc_center_i, ptc_normal_i
# In[Compute the weight
def cal_weight(data):
    """
    """

    # Compute weight
    data_mean = np.mean(data)
    w_ptc = np.exp(-data**10/(data_mean))

    return w_ptc
#%%
def find_rotation_matrix(mbb, principal_direction, constant):
    """This function is to find the rotation maxtrix to align axes of the box to coordinate axis, where the longest edge parallel to oz
    Parameters:
    ------------------------
        mbb                     : 3D mbb
        principal_direction     : estimated principal direction of a long edge. No need accuracy
        constant                : Constant
    Return:
    ------------------------
        rot                 : Rotation Matrix

    Demo:
        mbb = inlier_cluster_mbb
        constant = constants
    """
    # Find the vector of the longest edge
    if principal_direction is None:
        sort_ids = np.argsort(mbb["edges_length"])
        edges_vect = mbb["edges_vector"][sort_ids]
    else:
        cos_edges_dir = np.abs(cosin_vectors(mbb["edges_vector"], principal_direction))
        # Get the cross-section dimensions
        sort_ids = np.argsort(cos_edges_dir)
        edges_vect = mbb["edges_vector"][sort_ids]

    # Get the rotation maxtrix
    rot_1 = rot_vectors(constant['oz'], edges_vect[-1].flatten())
    edges_vect = np.matrix(edges_vect)*np.matrix(rot_1)
    edges_vect = np.asarray(edges_vect)
    rot_2 = rot_vectors(constant['ox'], edges_vect[1].flatten())

    # Return
    rot = np.matrix(rot_1)*np.matrix(rot_2)
    return np.asarray(rot)
#%% Filtering based distance
def statiscal_outlier_removal(ptc, num_kNN, std_mul):
    """This function is to create orthogonal vector of a 2D vector
    Parameters:
    ------------------------
        ptc                 :
        num_kNN             :
        std_mul             : scalar

    Return:
    ------------------------
        pt0                 : a orthogonal vector

    Demo:
        pt0 = np.array([1,2,4], dtype = float)
        pt0 = np.array([[1,2,4],[1,3,4]], dtype = float)
    ------------------------
    """
    from sklearn.neighbors import KDTree
    ptc = np.array(ptc, dtype = np.float64, copy = True)
	# Create kd-tree
    if ptc.shape[0] < num_kNN + 1: num_kNN = ptc.shape[0] - 1
    node_leaf_no_pts = num_kNN + 1
    kdt = KDTree(ptc, leaf_size = node_leaf_no_pts, metric='euclidean')
    dist, __ = kdt.query(ptc, k = num_kNN+1, return_distance= True, sort_results = True)
    dist_p = np.sum(dist[:,1:], axis = 1)/num_kNN
    dist_mean = dist_p.mean()
    dist_stddev = dist_p.std()
    dist_threshold = dist_mean + std_mul*dist_stddev
	#% Outlier removal
    mask = dist_p <= dist_threshold
    inlier_ptc_ids = np.where(mask == True)[0]
    inlier_ptc = ptc[inlier_ptc_ids]
	# Return
    return inlier_ptc_ids, inlier_ptc

# In[ Ransac plane]
def ransac_plane(model_ptc, model_tol = 0.015, model_min_ptc = 3, max_iters = 500, inlier_ratio = 0.9, debug = False):
    ''' This function is to fit the plane through the point cloud 

    Demo:
        model_ptc = ptc_normal.copy()
        model_tol = 0.01
        model_min_ptc = 3 # Minimum points for fitting plane
        max_iters = 500
        inlier_ratio = 0.9
        
#        outlier_method = {'band_width': 0.1}
#        outlier_method = {'probability': 0.95}
#        outlier_method = {'clustering': 0.1}
        outlier_method = {'residual_threshold': 0.1}
        
        outlier_option = 'hard'
        dist_3Dpoints_plane(point3D, surface_center, surface_normal)
    '''
#    
    # from utils_geometry import PCA_NormalVector, dist_3Dpoints_plane

    # Initial parameters
    best_model = None
    best_error = np.inf
    
    # Preallocation
    model_inlier_ptc_ids = np.arange(len(model_ptc))
    fit_model_info = np.array([])
    
    iteraction = 0
    while iteraction < max_iters:
        
        # Random generate points for fitting and testing
        np.random.shuffle(model_inlier_ptc_ids)
        fit_ptc_ids = model_inlier_ptc_ids[:model_min_ptc]
        test_ptc_ids = model_inlier_ptc_ids[model_min_ptc:]

        # Fit model 

        fit_ptc = model_ptc[fit_ptc_ids]
        test_ptc = model_ptc[test_ptc_ids]
        plane_normal = PCA_NormalVector(fit_ptc, min_ptc = model_min_ptc)
        
        # Compute distances from the points to the fitting plane
        plane_center = np.mean(test_ptc, axis = 0)
        test_error = dist_3Dpoints_plane(test_ptc, plane_center, plane_normal)
        this_error = np.mean(np.abs(test_error))
        
#        # Assign fitting information
        fit_model_info = np.append(fit_model_info, np.concatenate((plane_center, plane_normal)))
        fit_model_info = np.append(fit_model_info, this_error)
        
        # Extract inlier ptc
        inlier_ptc_ids = test_ptc_ids[np.abs(test_error) <= model_tol]

        # Update the best model
        if (len(inlier_ptc_ids) >= inlier_ratio*len(model_ptc))&(this_error < best_error):
            
            best_model = np.concatenate((plane_center, plane_normal)) 
            inlier_ptc_ids = np.concatenate((inlier_ptc_ids, fit_ptc_ids))

        else:
            
            best_error = this_error
         
        iteraction += 1 

    # Select the best model
    if best_model is None:
         # The best model has the smallest root mean square error
        fit_model_info = fit_model_info.reshape(-1,7)
        mask = np.argmin(fit_model_info[:,6])
        
        best_model = (fit_model_info[mask,0:6])
        
        # Calculate errors and the bound error
        model_error = dist_3Dpoints_plane(model_ptc, best_model[:3], best_model[3:])
        model_inlier_ptc_ids = np.sort(model_inlier_ptc_ids)
        inlier_ptc_ids = model_inlier_ptc_ids[np.abs(model_error) <= model_tol]
             
    # Return results
    return best_model, inlier_ptc_ids    

def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def calc_fit_circle_error(x, y, xc, yc, rc):
    '''This function is to calculate errors from the fitting model and data'''
    
    ri = calc_R(x, y, xc, yc)
    ri_errors = ri - rc
    rms = np.sqrt(np.sum(ri_errors**2)/len(ri))
    
    return ri_errors, rms

def ransac_circle(model_ptc, model_tol = 0.005, outlier_tolerance = 0.1, max_iters = 250):
     
    ''' This function is to fit the circle by removing outlier points.
        If the circle radius of the consecutive iteration differs less than the model_tol
        the searching is terminate

    Demo:
        model_ptc = ptc.loc[grid_cell_ptc_ids,['x', 'y']]
        model_tol = 0.01
#        outlier_method = {'band_width': 0.1}
#        outlier_method = {'probability': 0.95}
#        outlier_method = {'clustering': 0.1}
        outlier_method = {'residual_threshold': 0.1}
        
        outlier_option = 'hard'
    '''
    # Call libs
    import pandas as pd
    from skimage.measure import ransac, CircleModel

    if isinstance(model_ptc, pd.DataFrame):
        if any(item in ['x','y','z'] for item in model_ptc.columns):
            model_ptc = model_ptc[['x','y','z']].values    
        else:
            raise ValueError('The input data does not contain x, y, z coordinates')
    
    # Criteria
    model_robust_circle, __ = ransac(model_ptc, CircleModel, min_samples = 3,
                               residual_threshold = model_tol, max_trials = max_iters)
    xc, yc, rc = model_robust_circle.params
    
    # Re-calculate errors to identify inlier points
    model_ptc_x = model_ptc[:,0]
    model_ptc_y = model_ptc[:,1]
    test_error, test_rms = calc_fit_circle_error(model_ptc_x, model_ptc_y, xc, yc, rc)
    
    # Determinate inlier points
    model_ptc_ids = np.arange(model_ptc.shape[0]) 
    upper_bound_error, lower_bound_error = outlier_tolerance, - outlier_tolerance   
    fit_inlier_ptc_ids = model_ptc_ids[(lower_bound_error <= test_error)&(test_error <= upper_bound_error)]
            
    # Return results
    return xc, yc, rc, fit_inlier_ptc_ids 

def leastsq_fit_circle(x,y):
    ''' This function is to fit the circle by using a least square method
    Demo:
        cell_ptc_ids = large_segment_cell_ptc_xyz.index
        cell_ptc_xyz = large_segment_cell_ptc_xyz[['x','y']]
    ''' 
    from scipy import optimize
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    ri = calc_R(x, y, *center)
    rc = ri.mean()
    rms = np.sqrt(np.sum((ri - rc)**2)/len(ri))
    residual = ri - rc
    return xc, yc, rc, ri, rms, residual

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()