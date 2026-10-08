# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:06:53 2021

@author: LinhTruongHong
"""
import numpy as np
import pandas as pd
from numpy import linalg as LA
import numpy.matlib as npm
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

# In[]
def rot_direction(angle, direction):
    # Demo
    # angle, direction = angle, OZ_AXIS
    
    sina = np.sin(angle)
    cosa = np.cos(angle)

    direction = vector_unit(direction[:3])
    R = np.diag(np.array([cosa, cosa, cosa]))
    R += np.outer(direction, direction) *(1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])

    return R
# In[Rotation between two vectors]
def rot_vectors(vect_1, vect_2):
    ''' This function is to identify the rotation matrix betweento map vect_2 to vector_1 in 3D space
    
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
# In[]
def dist_point_points(pts_1, pts_2):
    """This function is to calculate distances between a points and points
    Parameters:
    ------------------------
        pts_1           : [1 x 3] - x, y, z coordinates
        pts_2           : [N x 3] - x, y, z
    Return:
    ------------------------
        dist_p2p        : [N x 1]
    Demo:
    pts_1 = np.array([0, 3, 0])
    pts_2 = np.array([[0, 3, 0], [2, 3, 3]])
    pts_1 = check_pts_lz[init_seeding_id]
    pts_2 = check_pts_lz[seeding_neighbour_ids]
    
    """
    delta_p2p = pts_1 - pts_2
    dist_p2p = vector_length(delta_p2p)
    return dist_p2p
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
    line_p, line_t = line[:3], line[3:]
    w = ptc - line_p
    b = np.dot(w, line_t)/np.dot(line_t, line_t)
    proj_ptc = line_p + b.reshape(-1, 1)*line_t

    return proj_ptc
#%%
def dist_points_line(p, line_p, line_vect):
    """ Calculate distances from points to a line in a parametric  form

    Demo:
        p = segment_features[neighbour_segment_ids, :3]
        line = cur_seeding_feature[:6]
    """
    w = line_p - p
    return vector_length(w - np.reshape(np.dot(w, line_vect), (-1,1))*line_vect)

def dist_3Dpoints_3Dline(p, line):
    """ Calculate distances from points to a line in a parametric  form

    Demo:
        p = segment_features[neighbour_segment_ids, :3]
        line = cur_seeding_feature[:6]
    """
    # Check line
    line = line.flatten()
    if len(line) == 4: #2D line
        line_p, line_t = line[:2], line[2:]
        p = p[:, :2]
    elif len(line) == 6: #3D line
        line_p, line_t = line[:3], line[3:]
        p = p[:, :3]
    else:
        raise ValueError("Only support 2D or 2D lines. Current line is Invalid")
    w = line_p - p
    return vector_length(w - np.reshape(np.dot(w, line_t), (-1,1))*line_t)

def proj_3Dpoints_3Dplane(ptc, plane):
    """This function is to porject points onto a plane
    Parameters:
    ------------------------
        ptc                 : [Nx3
        plane             : [1x6]

    Return:
    ------------------------
        ptc0                : a orthogonal vector

    Demo:

    """
    # Get plane
    plane = np.asarray(plane).flatten()
    plane_cent, plane_normal = plane[:3], plane[3:]
    # Project points onto the plane
    c = np.dot(plane_normal, plane_cent)
    a = np.sum(ptc*plane_normal, axis = 1) - c
    ptc0 = ptc - a.reshape(-1,1)*plane_normal.reshape(1,3)

    return ptc0

# In[Dist point to plane]
def dist_3Dpoints_3Dplane(point3D, plane):
    # Get plane
    plane = np.asarray(plane).flatten()
    plane_cent, plane_normal = plane[:3], plane[3:]
    # Compute a distance
    dist = point3D.dot(plane_normal) - plane_normal.dot(plane_cent)
    return dist
# In[Residual]
def residual(dist_ptc):
    return np.sqrt(np.sum(dist_ptc**2)/len(dist_ptc))

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
    proj_plane_ptc_xyz = proj_3Dpoints_3Dplane(ptc_xyz, plane)
    
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
            print("Size vectors pt0 = {:d} x {:d} does not support".format(pt0.ndim, pt1.ndim))
        else:
            check = True
    return pt0, pt1, check

# In[Calculate covariance matrix]
def covariance_matrix(ptc, centroid = 'centroid'):
    """
    The function estimate a the covariance matrix
    Parameters:
    ------------------------
        ptc                 : [Nx3] - a point cloud
        centroid            : centtroid
    
    Return:
    ------------------------
        covar_matrix       : a covariance matrix
    
    Demo:

    """
    ptc = np.array(ptc, dtype = np.float64, copy = True)
    ptc = np.asmatrix(ptc)
    if centroid.casefold() == 'centroid'.casefold():
        centroid = np.mean(ptc, axis=0)
        ptc = np.asmatrix(ptc - centroid)
        covar_matrix = ptc.T*ptc/len(ptc)
    else:
        covar_matrix = ptc.T*ptc

    return np.asarray(covar_matrix)

def covariance_normal_vector(cov):
    """
    The function estimate a normal vector from the covariance matrix
    Parameters:
    ------------------------
        cov                 : [3x3] - a covariance matrix
    
    Return:
    ------------------------
        normal_vector       : a normal vector
    
    Demo:

    """
    # Estimate a normal vector
    eigen_vals, eigen_vects = LA.eig(cov)
    ids = np.argsort(eigen_vals)
    eigen_vects = eigen_vects[:, ids]
    # Eigenvalues [e1 e2, e3] e1<e2<e3
    return eigen_vects[:, 0]
# In[Calculate a eigen vector by PCA]
def PCA_eigenspace(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
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
# In[Calculate a normal vector by PCA]
def PCA_normal_vector(ptc, min_ptc = 5):

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
def PCA_smallest_tangent_vector(ptc, min_ptc = 5):

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
def PCA_largest_tangent_vector(ptc, min_ptc = 5):

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
def wPCA(pts_xyz, tol = 0.5*np.pi/180, max_iters = 100, min_ptc = 5, initial_normal = np.array([0.0,0.0,1.0])):
    """This function is to compute a normal vector based weight of points
    Parameters:
    ------------------------
        pts_xyz             : [Nx3] - Indices
        tol                 : scalar - A tolerance for a different angle of normals from two consecutive iterations
        max_iters           : max number of iteration
        min_ptc             : min points defined the plane
        initial_normal      : An initial normal
        
    Return:
    ------------------------
        normal_vector       : a largest tangent vector
    
    Demo:

    pts_xyz = region_pts_xyz
    tol = 0.5*np.pi/180
    max_iters = 100
    initial_normal = None
    """
    # call libs
    

    # Check input data
    if isinstance(pts_xyz, pd.DataFrame):
        pts_xyz = pts_xyz[["x", "y", "z"]].values
    elif isinstance(pts_xyz, np.ndarray):
        if (pts_xyz.ndim == 0) or (pts_xyz.ndim > 2):
            raise ValueError("Expected 1D or 2D array, got %dD array instead" % pts_xyz.ndim)
        elif pts_xyz.ndim == 1:
            pts_xyz = pts_xyz.reshape(-1, len(pts_xyz))
        else:
            pts_xyz = pts_xyz[:, :3]
    else:
        raise ValueError("Expected 2D array or DataFrame")

    # Convert the threshold
    tol = np.cos(tol)

    # Compute a normal vector with weight
    if pts_xyz.shape[0] > min_ptc:

        # Initial center and normal
        pts_p0 = np.mean(pts_xyz, axis=0)
        if (initial_normal is None):
            pts_n0 = PCA_normal_vector(pts_xyz, min_ptc=min_ptc)
        elif len(initial_normal) < 3:
            pts_n0 = PCA_normal_vector(pts_xyz, min_ptc=min_ptc)
        else:
            pts_n0 = initial_normal

        # Iterations
        flag = True
        iters = 0
        while flag & (iters <= max_iters):

            # Establish weight for each point based on a distance
            d0 = np.abs(dist_3Dpoints_3Dplane(pts_xyz, np.append(pts_p0, pts_n0)))
            w = np.exp(-(d0**2)/10.)
            # Computer new center and normal
            w_pts_xyz = (pts_xyz*npm.repmat(w, 3, 1).T)
            pts_pi = np.sum(w_pts_xyz, axis=0)/np.sum(w)
            delta_pts_pi = pts_xyz - pts_pi
            itr_cov = (delta_pts_pi.T*npm.repmat(w, 3, 1)).dot(delta_pts_pi)
            pts_ni = covariance_normal_vector(itr_cov)

            # Check deviation of the normal
            if np.sum(pts_n0*pts_ni) <= tol: # Angle deviation is still large
                pts_p0 = pts_pi
                pts_n0 = pts_ni
                iters += 1
            else:
                flag = False

        # return the result
        return pts_pi, pts_ni
    else:
        return np.mean(pts_xyz, axis=0), np.full(3, np.nan)
# In[]
def iwPCA(pts_xyz, tol=None):
    """
    Ref: Robust normal vector estimation in 3D point clouds through iterative principal component analysis
    https://doi.org/10.1016/j.isprsjprs.2020.02.018
    The function is to estimate normal vectors
    
    Parameters:
    ------------------------
        pts_xyz                 : [Nx3] - Indices
        tol                     : threshold - dictionary
            "mu_lim"            : 0.005
            "opt_lim"           : 5%
            "min_ptc"           : 5 points
    Return:
    ------------------------
         pts_p0                 : point
         pts_n0                 : a normal vector
    Demo:
    ------------------------
    pts_xyz = region_pts_xyz
    mu_lim = 0.005
    """
    # Call libs/function
        
    # Check tolerance
    if tol is None:
        tol = {"mu_lim": 0.005, "opt_lim": 0.05, "itr_descent": 1.01, "min_ptc": 5}
    if isinstance(tol, np.ndarray):
        tol = {"mu_lim": 0.005, "opt_lim": 0.05, "itr_descent": 1.01, "min_ptc": 5}
    if not isinstance(tol, dict):
        tol = {"mu_lim": 0.005, "opt_lim": 0.05, "itr_descent": 1.01, "min_ptc": 5}
    
    # Check inout parameters
    if pts_xyz.shape[0] < tol["min_ptc"]:
        return np.full(3, np.inf), np.full(3, np.inf)
    # Estimate initial plane
    pts_p0 = np.mean(pts_xyz, axis=0)
    pts_n0 = PCA_normal_vector(pts_xyz, min_ptc=5)
    # Compute the residual
    r0 = dist_3Dpoints_3Dplane(pts_xyz, np.append(pts_p0, pts_n0))
    mu0 = np.max(r0)
    w0 = weight_m_estimator(mu0, r0)
    opt0 = np.sum(w0*(r0**2))   
    #% Iteration
    flag_itr = False
    while (mu0 > tol["mu_lim"]) & (not flag_itr):
        # Compute the shif points
        shift_pt = np.sum(w0*r0)/np.sum(w0)
        pts_pi = pts_p0 + shift_pt*pts_n0
        
        # Estimate new normal
        delta_pts_p1 = pts_xyz - pts_pi
        itr_cov = (delta_pts_p1.T*npm.repmat(w0, 3, 1)).dot(delta_pts_p1)
        pts_ni = covariance_normal_vector(itr_cov)
        
        # Compute new weight
        ri = dist_3Dpoints_3Dplane(pts_xyz, np.append(pts_pi, pts_ni))
        mui = mu0/tol["itr_descent"]
        wi = weight_m_estimator(mui, ri)
        opti = np.sum(wi*(ri**2))
        # Determine shift direction
        if (opti - opt0)/opt0 <= -tol["opt_lim"]:
            flag_itr = True
        else:
            # Update for a next iteration
            mu0, w0 = mui, wi
            pts_p0, pts_n0 = pts_pi, pts_ni
            r0 = ri
    # return
    return pts_p0, pts_n0

# In[]
def weight_m_estimator(mu, r):
    """
    The function is to compute weight of each point
    
    """
    w = (mu/(mu + r**2))**2
    return w
#%%
def rPCA(pts_xyz, tol=None):
    """
    This is a robust PCA using both normal and surface distance as criteria
    Parameters
    ----------
        pts_xyz         : [N x 3] 
        tol             : tolerance

    Returns
    -------
    None.
    pts_xyz = surface_pts_xyz
    tol = fit_plane_tol
    
    pts_xyz = region_pts_xyz
    tol = threshold["vrg_threshold"]["fit_plane"]
    """

    # Constant
    if tol is None:
        tol = {"cosin_normal": np.deg2rad(1.0), "distance": 0.005, "max_itr": 100, "min_ptc": 5}
    if not isinstance(tol, dict):
        tol = {"cosin_normal": np.deg2rad(1.0), "distance": 0.005, "max_itr": 100, "min_ptc": 5}
        
    # Check input variable
    if pts_xyz.shape[0] < tol["min_ptc"]:
        return np.mean(pts_xyz, axis=0), np.full(3, np.nan)
    # Initial center and normal
    pts_p0 = np.mean(pts_xyz, axis=0)
    pts_n0 = PCA_normal_vector(pts_xyz, min_ptc=tol["min_ptc"])

    # Compute the weight and distance
    d0 = dist_3Dpoints_3Dplane(pts_xyz, np.append(pts_p0, pts_n0))
    w0 = cal_weight(np.abs(d0))

    #% Iteraction
    no_iter = 1
    flag = True
    while flag & (no_iter <= tol["max_itr"]):
        # Compute the new center and normal vector
        w_pts_xyz = pts_xyz*w0.reshape(-1,1)
        w_pts_pi = np.sum(w_pts_xyz, axis=0)/np.sum(w0)
        # Estimate normal
        delta_pts_cent = pts_xyz - w_pts_pi
        w_delta_pts_cent = delta_pts_cent*w0.reshape(-1,1)
        itr_cov = w_delta_pts_cent.T.dot(delta_pts_cent)
        pts_ni = covariance_normal_vector(itr_cov)
        
        # Compute the error
        itr_err_normal = np.abs(cosin_vectors(pts_n0, pts_ni))
        itr_err_dist = np.abs(dist_3Dpoints_3Dplane(pts_p0, np.append(w_pts_pi, pts_ni)))

        # Check
        if ((itr_err_normal <= tol["cosin_normal"]) & (itr_err_dist <= tol["distance"])):
            flag = False
        else:
            # Update the new w0
            d0 = dist_3Dpoints_3Dplane(pts_xyz, np.append(w_pts_pi, pts_ni))
            w0 = cal_weight(np.abs(d0))
            no_iter = no_iter + 1
    # Return
    return w_pts_pi, pts_ni
# In[Compute the weight
def cal_weight(data):
    """
    Ref: https://ieeexplore-ieee-org.tudelft.idm.oclc.org/abstract/document/1183771
    
    data = np.abs(d0)
    """
    h = np.max(data)/3
    w = np.exp(-data**2/(h**2))
    return w
