#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 08:04:44 2018

@author: linhtruonghong
"""

# Utilities
import numpy as np
import pandas as pd
import copy
from numpy import linalg as LA
# In[This function is to conver length unit from metric to imperical]
def imperial2meter(unit = "in"):

    """This function is to return the convert unit from si to imperical"""

    if unit.casefold() == 'in'.casefold():
        # Convert unit
        unit_convert = 0.0254
    elif (unit.casefold() == 'ft'.casefold()) or (unit.casefold() == 'foot'.casefold()):        
        unit_convert = 12.0*0.0254
    else:
        unit_convert = 1.0
        
    return unit_convert
# In[This function is to conver length unit from metric to imperical]
def meter2imperial(unit = "imperial"):

    """This function is to return the convert unit from si to imperical"""

    if unit.casefold() == 'imperial'.casefold():
        # Convert unit
        unit_convert = 1/(0.0254)
    else:        
        unit_convert = 1.0
    return unit_convert

# In[Add unit to measurement]
def num2text(number, unit = 'imperial'):
    
    if unit.casefold() == 'imperial'.casefold():
        if number >= 0:
            prefix_s = ""
        else:
            prefix_s ="-"
        number = np.abs(number)
        if number >= 12:
            s = prefix_s + str(number//12).rstrip('0').rstrip('.') + ' ft ' + str("%0.2f" % (number%12)) + " in"      
        else:
            s = prefix_s + str("%0.2f" % number) + " in" 
            
    elif unit.casefold() == 'si'.casefold():
        s = str("%.3f" % number) + " m"
    else:
        raise ValueError('Unit do not support')
    
    return s

def num2text_file_name(number, unit = 'imperial'):
    
    if unit.casefold() == 'imperial'.casefold():
        if number >= 0:
            prefix_s = ""
        else:
            prefix_s ="-"
        number = np.abs(number)
        if number >= 12:
            s = prefix_s + "{:03.0f}".format(number//12) + ' ft ' + "{:05.2f}".format(number%12) + " in" 
        else:
            if number > 1190:
                s = prefix_s + "{:05.2f}".format(number) + " in"   
            else:
                s = prefix_s + "{:04.2f}".format(number) + " in"
                
    elif unit.casefold() == 'si'.casefold():
        s = "{:05.2f}".format(number) + " m"
    else:
        raise ValueError('Unit do not support')
    
    return s

# In[This function is to check if b in a]
def ismember(a, b):
	ind = np.array([np.where(np.all(a == i, axis = 1) == True)[0] for i in zip(b)])
	if ind.ndim == 2:
		ind = np.squeeze(ind)

	return ind

# In[This function to identify if b in a]
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

# In[Convert Catersian to Polar]
def cart2pol(x, y):
    
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

# In[Convert Polar to Catersian coordinate system]
def pol2cart(rho, theta):

    rho = np.asarray(rho)
    theta = np.asarray(theta)

    if (theta.size > 1) and (rho.size == 1):
        rho = np.full(theta.size, rho)
    elif (rho.size > 1) and (theta.size == 1):
        theta = np.full(rho.size, theta)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y
# In[Convert a point cloud in Cartesian coordinate system to the polar coordinate system]
def global_cart_2_local_polar(ptc, origin):
    ''' This function is to convert the point cloud in the global Cartesian coordinate
    system to a local polar coordinate system at the origin
    ptc = cylinder_grid_ptc.copy()
    origin = grid_cell_info[int(i),1:3].copy()
    '''
    if isinstance(ptc, pd.DataFrame):
        
        ptc_xy = ptc[['x','y']].values - origin
        index = ptc.index
        
    elif isinstance(ptc, np.ndarray):
        
        ptc_xy = ptc[:,:2] - origin
        index = np.arange(0, ptc_xy.shape[0])
        
        # Set columns index
        if ptc.shape[1] == 2:
            columns = ['x', 'y']
        elif ptc.shape[1] == 3:
            columns = ['x', 'y', 'z']
        else:
            columns = np.arange(ptc.shape[1])
            
        ptc = pd.DataFrame(ptc, columns = columns, index = index)
        
    else:
        
        raise ValueError('The input type does not support')
        
        
    ptc_rho, ptc_theta = cart2pol(ptc_xy[:,0], ptc_xy[:,1])

    # Build vectors storing global and local polar corrdinates
    ptc_polar = pd.DataFrame(np.hstack((ptc_theta.reshape(-1,1), ptc_rho.reshape(-1,1))),
                                            columns = ['theta', 'rho'], index = index)

    # Build a dataframe storing both a global and local polar coordinates of the point cloud
    ptc = pd.concat([ptc, ptc_polar], axis=1, ignore_index = False)

    # Return the result
    return ptc
# In[length of the vector]
def vector_length(vect):
	vect = np.array(vect, dtype = np.float64, copy = True)
	if vect.ndim == 1:
		return np.sqrt(np.sum(vect*vect))
	else:
		return np.sqrt(np.sum(vect*vect, axis = 1))
# In[Normalize a vector]
def vector_norm(vect):
	vect = np.array(vect, dtype = np.float64, copy = True)
	if vect.ndim == 1:
		vect_norm = vect/np.sqrt(np.dot(vect,vect))
	else:
		vect_len = np.sqrt(np.sum(vect*vect, axis = 1))
		vect_norm = vect/vect_len[:, None]

	return vect_norm
# In[Calculate cosin between two vectors]
def cosin3DVector(vect1,vect2):
	#
    if len(vect1.shape) == 1:
        vect1 = np.array(vect1, dtype=np.float32, copy=True).reshape(-1, len(vect1))
    else:
        vect1 = np.array(vect1, dtype=np.float32, copy=True).reshape(-1, vect1.shape[1])
        
    if len(vect2.shape) == 1:
        vect2 = np.array(vect2, dtype=np.float32, copy=True).reshape(-1, len(vect2))
    else:
        vect2 = np.array(vect2, dtype=np.float32, copy=True).reshape(-1, vect2.shape[1])
        
    
#    vect1 = vector_norm(vect1)
#    vect2 = vector_norm(vect2)
    cosin2Vect = np.sum(vect1*vect2, axis = 1)

    return np.reshape(cosin2Vect, -1,1)
# In[Projection 3D points to a 3D parametric line]
def proj_point_on_line_3D(ptc, line):
    """This function is to project 3D points onto 3D line
    Input:
        ptc           : [Nx3] - x, y, z coordinates
        line          : [1x6] - P0 (x, y, z), t (tx, ty, tz)
    Output:

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
# In[Calculate distance from point to a line section]
def distance_to_line(p1, p2, p):
    """ This function is to calculate distances p to line p1p2

    p1 = rect_center
    p2 = sect_center
    p = sect_ptc
    """

    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = np.abs(y_diff*p[:, 0] - x_diff*p[:, 1] + p2[0]*p1[1] - p2[1]*p1[0])
    den = np.sqrt(y_diff**2 + x_diff**2)
    return num/den


# In[]
def distance_points_2_parametric_line(p, line):
    """ Calculate distances from points to a line in a parametric  form

    Demo:
        p = segment_features[neighbour_segment_ids, :3]
        line = cur_seeding_feature[:6]
    """
    from Utilities import vector_length, vector_norm
    p0 = line[:3]
    ul = line[3:]
    ul = vector_norm(ul)

    w = p - p0

    return vector_length(w - np.reshape(np.dot(w, ul), (-1,1))*ul)

# In[Cal a distance from points to the surface]
def dist_3Dpoint_surfaces(point3D, surface_center, surface_normal):

	# Demo
#	point3D = cur_out_sel_seg_ptc_xyz
#	surface_center = in_sel_seg_ptc_xyz
#	surface_normal = in_sel_seg_ptc_normal

	# Compute a distance
	dist = point3D.dot(surface_normal) - surface_normal.dot(surface_center)
	return dist
# In[Residual]
def residual(ptc_dist):
    return np.sqrt(np.sum(ptc_dist**2)/len(ptc_dist))

# In[Project points onto its fitting plane]
def proj_3Dpoints_surface(ptc):
    """This function is to porject points onto its fitting plane
    Input:
        ptc             : [Nx3]
    Output:
        proj_ptc        : [Nx3]
    Demo:
        
    
    """
    surface_normal_vector = PCA_normal_vector(ptc, min_ptc = 5)
    surface_center = np.mean(ptc, axis = 0)
    c = np.dot(surface_normal_vector, surface_center)
    a = np.sum(ptc*surface_normal_vector, axis = 1) - c
    proj_ptc = ptc - a.reshape(-1,1)*surface_normal_vector.reshape(1,3)
    
    return proj_ptc 

def proj_3Dpoints_3Dsurface(ptc, surface):
    """This function is to porject points onto its fitting plane
    Input:
        ptc             : [Nx3]
        surface         : [1x6] xc,yc,zc,nx,ny,nz
    Output:
        proj_ptc        : [Nx3]
    Demo:
        
    
    """
    surface_center = surface[:3]
    surface_normal_vector = surface[3:]
    
    c = np.dot(surface_normal_vector, surface_center)
    a = np.sum(ptc*surface_normal_vector, axis = 1) - c
    proj_ptc = ptc - a.reshape(-1,1)*surface_normal_vector.reshape(1,3)
    
    return proj_ptc
# In[Calculate a convariance matrix]
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
# In[Robust PCA]
def wPCA(ptc, tol = 1.0*np.pi/180, max_iters = 100, min_ptc = 5, initial_normal = np.array([0.0,0.0,1.0])):
    """This function is to compute a normal vector based weight of points
    ptc = shell_wall_bottom_ptc.copy()
    tol = 0.5*np.pi/180
    max_iters = 100
    """
    
    # Check input data
    import numpy.matlib as npm
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
        if (initial_normal.shape[0] < 1) or (initial_normal is None):
            ptc_normal = PCA_normal_vector(ptc, min_ptc = min_ptc)  
        else:
            ptc_normal = initial_normal
        
        # Iterations
        flag = True
        iters = 0
        while flag & (iters <= max_iters):
            
            # Establish weight for each point based on a distance
            dist_ptc_surface = np.abs(dist_3Dpoint_surfaces(ptc, ptc_center, ptc_normal))
            ptc_weight = np.exp(-dist_ptc_surface**2/(np.mean(dist_ptc_surface)/5)**2)
            ptc_weight = npm.repmat(ptc_weight, 3, 1).T
            
            # Computer new center and normal
            ptc_center_new = np.sum(ptc*ptc_weight, axis = 0)/np.sum(ptc_weight[:,0])
            cell_ptc_center_new = (ptc - ptc_center_new)
            cell_ptc_covariance_new = (cell_ptc_center_new*ptc_weight).T.dot(cell_ptc_center_new)/np.sum(ptc_weight[:,0])
            ptc_normal_new = cov_PCA(cell_ptc_covariance_new)
            
            # Check deviation of the normal
            if np.sum(ptc_normal*ptc_normal_new) <= tol: # Angle deviation is still large
                ptc_center = ptc_center_new
                ptc_normal = ptc_normal_new
                iters += 1
            else:
                flag = False
                
        # return the result
        return ptc_normal_new, ptc_center_new
    else:
                
        return np.full(3, np.nan), np.mean(ptc, axis = 0)

#%%    Compute a normal vector from  a convariance matrix  
def cov_PCA(covariance_matrix):
    """This function is to estimate a normal vector based on PCA from a covarance matrix"""
    w, v = LA.eig(covariance_matrix)
    ids = np.argsort(w)
    v = v[:,ids]
    normal_vector = v[:,0]
    
    return normal_vector

# In[Calculate a eigen vector by PCA]
def PCA_eigen_vector(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    if ptc.shape[0] >= min_ptc:
    # Calculate eigenspace
        cov_matrix = covariance_matrix(ptc, centroid = 'centroid')
        w, v = LA.eig(cov_matrix)
        ids = np.argsort(w)
    	# Eigenvalues [e1 e2, e3] e1<e2<e3
        v = v[:,ids]
    else:
        v = np.full((3,3), np.inf)

    # Return results
    return v
# In[Calculate a normal vector by PCA]
def PCA_normal_vector(ptc, min_ptc = 5):

    ''' This function is to compute a normal vector of the point based on its neighbour
    The plane fitting was used to determine the normal vector
    ptc = ptc.copy()
    min_ptc = 5
    '''
    eigen_vector = PCA_eigen_vector(ptc, min_ptc = min_ptc)
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
    eigen_vector = PCA_eigen_vector(ptc, min_ptc = min_ptc)
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
    eigen_vector = PCA_eigen_vector(ptc, min_ptc = min_ptc)
    largest_tangent_vector = eigen_vector[:, -1]   
    # Return results
    return largest_tangent_vector



# In[Rotate matrix around the direction with an angle]
def rotation_matrix(angle, direction):
    sina = np.sin(angle)
    cosa = np.cos(angle)

    direction = vector_norm(direction[:3])
    R = np.diag(np.array([cosa, cosa, cosa]))
    R += np.outer(direction, direction) *(1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])

    return R
# In[Rotation between two vectors]
def Rot_2vector(vect_1, vect_2):
    ''' This function is to identify the rotation matrix between two vectors in 3D space
    Ref: see vrrtvec.m and vrrotvec2mat.m

    vect_1 = copy.deepcopy(vect_1)
    vect_2 = copy.deepcopy(vect_2)

    vect_1 = np.array([0,0,1])
    vect_2 = np.array([0.1, 0.2, 0.8])
    '''

    # Copy data
    vect_1 = copy.deepcopy(vect_1)
    vect_2 = copy.deepcopy(vect_2)

    # Normalization
    vect_1 = vector_norm(vect_1)
    vect_2 = vector_norm(vect_2)

    # Estimate a rotation vector
    ax = vector_norm(np.cross(vect_1, vect_2))
    angle = np.arccos(np.dot(vect_1, vect_2))

    if not np.any(ax):
        abs_vect_1 = np.abs(vect_1)
        mask = np.argmin(abs_vect_1)
        c = np.zeros((1,3))
        c[0,mask] = 1
        ax = vector_norm(np.cross(vect_1, c)).flatten()

#    rotation_vector = np.append(ax, angle)

    # Rotation maxtrix
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    x = ax[0]
    y = ax[1]
    z = ax[2]

    rotation_matrix = np.full((3,3), np.inf)

    rotation_matrix[0] = [t*x*x + c, t*x*y - s*z, t*x*z + s*y]
    rotation_matrix[1] = [t*x*y + s*z, t*y*y + c, t*y*z - s*x]
    rotation_matrix[2] = [t*x*z - s*y, t*y*z + s*x, t*z*z + c]

    return rotation_matrix

def window_interp(x, xp, fp, window_size):
    """ This function is to interpolate the value in One-dimensional array
    Similarly to numpy.interp but here using window_size to calculate average values    
    
    x = elevation_section
    xp = step_sect_ptc_elevation
    fp = top_bottom_radius_error
    window_size = buffer_band
    
    """
    #% preallocation
    y = np.full(len(x), np.nan)
    
    for xi, i in zip(x, range(len(x))):
        mask = (xi - window_size/2 <= xp)&(xp <= xi + window_size/2)       
        y[i] = np.mean(fp[mask]) if sum(mask) > 0 else np.nan
    
    return y

def closest_ptc(data, pt, buffer):
    """This function is to find the closed point of the given point in one direction.
    If the given point, pt is less than mean of ptc, the searching will be ascending; otherwise descending
    
    Input:
        data    : [NxM] - Point coordinates in one direction
        pt      : scalar - A point coordinate in one direction
        buffer  : scalar - window to search the neibour points within the given point pt

    Output:
        val     : Scalar - a mean value of the point within the buffer
    
    Demo:
        data = np.vstack((section_ptc_elevation, section_ptc_radius)).T
        pt = bottom_elevation
        buffer = SECTION_THICKNESS
        
        data = ptc_settlement.copy()
        pt = measured_stat.copy()
        buffer = SECTION_THICKNESS/2
        
#    """
#    if data.shape[1] == 1:
#       col_id = 0
       
    if pt <= np.mean(data[:,0]):
        search_dir = 1      
    else:
        search_dir = -1   
    
    # Searching
    mask = np.argsort(data[:, 0])
    data = data[mask]
    
    flag = True
    
    while flag:
        
        mask = (pt - buffer <= data[:, 0])&(data[:, 0] <= pt + buffer)
        
        pt_buffer = data[mask]
        
        if pt_buffer.shape[0] >= 3:
            val = np.mean(pt_buffer, axis = 0)
            flag =  False
        else:
            mask = (data[:, 0] - pt)*search_dir > 0
            pos_ptc_ids = np.where(mask == True)[0]
            pt = data[pos_ptc_ids[0], 0]
    
    # Return result
    return  pt, val   

def closest_ptc_2(data, pt, buffer):
    """This function is to find the closed point of the given point in one direction.
    If the given point, pt is less than mean of ptc, the searching will be ascending; otherwise descending
   
    Expand the buffer
    
    Input:
        data    : [NxM] - Point coordinates in one direction
        pt      : scalar - A point coordinate in one direction
        buffer  : scalar - window to search the neibour points within the given point pt

    Output:
        val     : Scalar - a mean value of the point within the buffer
    
    Demo:
        data = np.vstack((section_ptc_elevation, section_ptc_radius)).T
        pt = bottom_elevation
        buffer = SECTION_THICKNESS
        
        data = ptc_settlement.copy()
        pt = measured_stat.copy()
        buffer = SECTION_THICKNESS/2
        
#    """
#    if data.shape[1] == 1:
#       col_id = 0
       
#    if pt <= np.mean(data[:,0]):
#        search_dir = 1      
#    else:
#        search_dir = -1   
    
    # Searching
    mask = np.argsort(data[:, 0])
    data = data[mask]
    
    flag = True
    ini_buffer = buffer
    while flag:
        
        mask = (pt - buffer <= data[:, 0])&(data[:, 0] <= pt + buffer)

        
        pt_buffer = data[mask]
        
        if pt_buffer.shape[0] >= 3:
            val = np.mean(pt_buffer, axis = 0)
            flag =  False
        else:
#            mask = (data[:, 0] - pt)*search_dir > 0
#            pos_ptc_ids = np.where(mask == True)[0]
#            pt = data[pos_ptc_ids[0], 0]
            buffer = buffer + ini_buffer/8.
    
    # Return result
    return  pt, val 
  
def moving_average_2(x, y, window_size = 10, option = {'sort': 'y'}):
    
    """This function is to calculate an average of the point based on its neighboue
    x = ptc_plumbness
    y = ptc_elevation 
    window_size = 5
    option = {'sort': 'y'}
    """
    if "sort" in option.keys():
        sort_var = option["sort"]
    
    if sort_var.casefold() == "x".casefold():
        
        mask = np.argsort(x) 
        x, y = x[mask], y[mask]
        
    elif sort_var.casefold() == "y".casefold():
        
        mask = np.argsort(y)
        
        x, y = x[mask], y[mask]
        
    else:
        raise ValueError('The %s does not support' % sort_var)
 
    # Calculate moving average for each 1d array
    xx = cal_moving_average(x, window_size = window_size)
    yy = y
        
    xx = np.append(xx, xx[0])
    yy = np.append(yy, yy[0])
    return xx, yy

def cal_moving_average(x, window_size = 3):
    """This function is to calculate moving average for 1d
    """
    
    if window_size % 2 == 1: #Odd
        front_tail = int((window_size - 1)/2)
        end_tail = int((window_size - 1)/2)
    else:
        front_tail = int((window_size)/2)
        end_tail = int((window_size)/2 - 1)
        
    # Preallocation result
    mov_average = np.full(len(x), np.inf)
    
    x_cumsum = np.cumsum(np.insert(x, 0, 0)) 
    
    if end_tail == 0:
        mov_average[front_tail:] = (x_cumsum[window_size:] - x_cumsum[:-window_size]) / float(window_size)
    else:
        mov_average[front_tail:-end_tail] = (x_cumsum[window_size:] - x_cumsum[:-window_size]) / float(window_size)
    
    #% For font edge
    
    for ids in range(front_tail):
        if window_size % 2 == 1: #Odd
            s_ids = 0 if ids - (window_size-1)/2 < 0 else ids
            e_ids = int(ids + (window_size-1)/2 + 1)
        
        else:
            s_ids = 0 if ids - window_size/2 < 0 else ids
            e_ids = int(ids + window_size/2)
        
        mov_average[ids] = np.mean(x[s_ids:e_ids])

    #% For end edge
    for ids in range(len(x) - end_tail, len(x)):
        if window_size % 2 == 1: #Odd
            s_ids = int(ids - (window_size-1)/2)
            e_ids = len(x) if ids + window_size/2 > len(x) else ids
        
        else:
            s_ids = int(ids - window_size/2)
            e_ids = len(x) if ids + window_size/2 > len(x) else ids
        
        mov_average[ids] = np.mean(x[s_ids:e_ids])
        
    return mov_average
# In[Moving average for radial distribution]
def radial_moving_average(data_x, data_y, window_size = {"Point": 9}):
    """This function is to calculate smooth line based on average method for a radial data distribution

    Input:
        data_x           : [Nx1] - to be calculate average
        data_y           : [Nx1] - to be sort 
        window_size
    Output:
        data_average_x
        data_average_y
    Demo:   
        data_x = sect_ptc_rho #rim_ptc_rho.copy()
        data_y = sect_ptc_theta#rim_ptc_theta
        window_size = {"Point": 14}
        window_size = {"Angle": 0.25*np.pi/180.0}
    """
    # Copy data
    data_x = data_x.copy()
    data_y = data_y.copy()
    
    #% In[Sorting data]
    mask = np.argsort(data_y)
    data_x = data_x[mask]
    data_y = data_y[mask]

    #In[Calculate average]
    #Ireallocate    
    data_average_x = np.zeros_like(data_x)
    data_average_y = data_y
    if 'Point' in window_size.keys():
        window_val = window_size["Point"]
        
        if window_val % 2 == 1: #Odd
            head = int((window_val - 1)/2)
            tail = int((window_val - 1)/2)
            head_data_x = data_x[0:head]
            tail_data_x = data_x[-tail:]
        else:
            head = int((window_val)/2)
            tail = int((window_val)/2 - 1)
            head_data_x = data_x[0:head]
            tail_data_x = data_x[-tail:]
        extend_data_x = np.append(tail_data_x, np.append(data_x, head_data_x))
        for count in range(tail, data_x.shape[0] + tail):
            data_average_x[count-tail] = np.mean(extend_data_x[count-tail:count+head+1])
    else:
        window_val = window_size["Angle"]
        for count, angle in enumerate(data_y):
            start_angle = angle - 0.5*window_val
            end_angle = angle + 0.5*window_val
            if start_angle < 0:
                mask = (2.0*np.pi + start_angle <= data_y)|(data_y <= end_angle)
            elif (0 <= start_angle)&(end_angle <= 2.0*np.pi):
                mask = (start_angle <= data_y)&(data_y <= end_angle)
            else:
                mask = (start_angle <= data_y)|(end_angle - 2.0*np.pi <= data_y)
            
            data_average_x[count] = np.mean(data_x[mask])
    # Return results
    return data_average_x, data_average_y
 

# In[Moving average based circular segment]
def moving_average_circular_segment(data_rho, data_theta, predefined_theta, interval_angle):
    """This function is to calculate smooth line based on average method for a circular segments

    Input:
        data_rho           : [Nx1] - a radius use to calculate a new radius for a smooth
        data_theta         : [Nx1] - a angle use to search the radius within the interval angle
        predefined_theta   : [Mx1] - A predeined angle to calculate new radius
        interval_angle     : scalar - angle to determine the points in range
    Output:
        data_average 
    Demo:   
        data_rho = rim_ptc_rho.copy()
        data_theta = rim_ptc_theta.copy()
        predefined_theta = None
        interval_angle
        
    """
    # Call libraries
    from sklearn.neighbors import KDTree
    
    # Check input
    if predefined_theta is None:
        predefined_theta = np.array(data_theta, copy = True)
        
    # Copy data
    data_theta = np.array(data_theta, copy = True)
    data_rho = np.array(data_rho, copy = True)
    
    
    # add head
    delta_data_theta = data_theta - 2.0*np.pi
    mask = delta_data_theta >= - interval_angle
    head_data_theta = delta_data_theta[mask]
    head_data_rho = data_rho[mask]
    # Add tail
    mask = data_theta <= interval_angle
    tail_data_theta = data_theta[mask] + 2.0*np.pi
    tail_data_rho = data_rho[mask]
    
    data_theta = np.concatenate((head_data_theta, data_theta, tail_data_theta))
    data_rho = np.concatenate((head_data_rho, data_rho, tail_data_rho))
    
    
    # Searching points in a range
    leaf_size = 50
    tree = KDTree(data_theta.reshape(-1,1), leaf_size = leaf_size)
    neighbour_ptcs_ids = tree.query_radius(predefined_theta.reshape(-1,1), r = 0.5*interval_angle)
    
    #In[Calculate average]     
    data_average = np.full((predefined_theta.shape[0],2), np.inf)
    data_average[:,1] = predefined_theta

    for count in range(predefined_theta.shape[0]):      
        neighbour_ptc_ids = neighbour_ptcs_ids[count]
        if neighbour_ptc_ids.shape[0] > 0:
            data_average[count, 0] = np.mean(data_rho[neighbour_ptc_ids])
  
    mask = np.isinf(data_average[:, 0])
    data_average = data_average[~mask,:]
    
    # Return results
    return data_average[:, 0], data_average[:,1]     
#%% Calculate average
def average_circular_segment(data_theta, data_rho, count, start_angle, end_angle):    
    """This function is to extract the data in a range and compute the average value
    Input:
        
    Output:
        
    Demo:
    """
    # Compute the average value
    mask = (start_angle <= data_theta)&(data_theta <= end_angle)
    local_ids = np.where(mask == True)[0]
    if local_ids.shape[0] > 0:
        data_average_rho = np.mean(data_rho[local_ids])
    else:
        data_average_rho = np.inf
    
    return (count, data_average_rho)
            
    
# In[Moving average based circular segment]
def moving_average_length_segment(data_x, data_y, predefined_x, interval_length):
    """This function is to calculate smooth line based on average method for a circular segments

    Input:
        data_x              : [Nx1] - x use to search the y within the interval length
        data_y              : [Nx1] - y use to calculate a new y for a smooth
        predefined_x        : [Mx1] - A predeined length to calculate new y
        interval_length     : scalar - length
    Output:
        data_average 
    Demo:   
        data_rho = rim_ptc_rho.copy()
        data_theta = rim_ptc_theta.copy()
        predefined_theta = None
        interval_angle
        
    """
    # Call libraries
    from sklearn.neighbors import KDTree
    
    # Check input
    if predefined_x is None:
        predefined_x = np.array(data_x, copy = True)
        
    # Copy data
    data_x = np.array(data_x, copy = True)
    data_y = np.array(data_y, copy = True)

    # Searching points in a range
    leaf_size = 50
    tree = KDTree(data_x.reshape(-1,1), leaf_size = leaf_size)
    neighbour_ptcs_ids = tree.query_radius(predefined_x.reshape(-1,1), r = 0.5*interval_length)
    
    #In[Calculate average]     
    data_average = np.full((predefined_x.shape[0],2), np.inf)
    data_average[:,0] = predefined_x
    
    for count in range(predefined_x.shape[0]):      
        neighbour_ptc_ids = neighbour_ptcs_ids[count]
        if neighbour_ptc_ids.shape[0] > 0:        
            data_average[count, 1] = np.mean(data_y[neighbour_ptc_ids])
  
    mask = np.isinf(data_average[:, 1])
    data_average = data_average[~mask,:]
    mask = np.argsort(data_average[:, 0])
    data_average = data_average[mask]
    # Return results
    return data_average[:, 0], data_average[:,1]     

#%% Filtering based distance
def stat_outlier_removal(ptc, num_kNN, std_mul):
    
    from sklearn.neighbors import KDTree
    ptc = np.array(ptc, dtype = np.float64, copy = True)
	# Create kd-tree
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

# In[Calculate features of the minimum bounding box]
def min_bounding_box_feature(ptc, long_dir):
    """This function is to calculate the features of the minimum bounding box of the data points
    Input:
        ptc        : [Nx3] - x, y, z, coordinates
        long_dir   : [Nx2] - a vector of longtudial axis of  the rafter
    Output:
        rect_features = [rect center, rect_dir, rect_width]
        
        
    Demo:
        min_bounding_box_feature(sect_ptc, long_dir)
    """
    # Call a library
    from scipy.spatial.distance import cdist
    from MinBoundingBox import minimum_bounding_box
    
    # Determine the bounding box and width of the section
    rect_corners = minimum_bounding_box(ptc[:, :2])
    rect_corners = np.array(rect_corners)
    rect_length = cdist(rect_corners, rect_corners)[0, [1, 3]]
    rect_center = np.mean(rect_corners, axis = 0)
    
    # Calculate angles of edges
    rect_edge_vectors = np.array([rect_corners[1] - rect_corners[0], rect_corners[2] - rect_corners[1]])
    rect_edge_angle = np.arctan2(*rect_edge_vectors.T[::-1])
    mask = np.where((rect_edge_angle < 0) == True)[0]
    rect_edge_angle[mask] = rect_edge_angle[mask] + 2.0*np.pi
    
    # Determine edge parallel to the rafter 
    diff_angle_rafter_rect = np.abs(np.cos(long_dir - rect_edge_angle))
    rect_width = rect_length[np.argmin(diff_angle_rafter_rect)]
    rect_long_dir = rect_edge_vectors[np.argmax(diff_angle_rafter_rect)]
     
    return np.hstack((rect_center, rect_long_dir, rect_width))


# In[UTILITIES FOR PLOT]
def polar_horizontal_align(rot_angle):
    
    '''This is to set the horizontal alignment for text
    '''
    
    if (0 < rot_angle) & (rot_angle <= np.pi):
        
        hor_align = "right"
    else:
        hor_align = "left"
    
    return hor_align

def rot_text(rot_angle, rot_origin = 0):
    """This is to return a rotation angle"""
#    if (np.pi/2 < rot_angle) & (rot_angle <= 3*np.pi/2):
    if (0 < rot_angle) & (rot_angle <= np.pi):    
        return (rot_angle + rot_origin)*180.0/np.pi + 180.0
    else:
        return (rot_angle + rot_origin)*180.0/np.pi

# In[Text origin]
def origin_loc(angle):
    
    """This function is to return a location of the origin in text"""

    if angle == 0.0:
        origin_location = "East"        
    elif angle == np.pi/2:        
        origin_location = "North"    
    elif angle == np.pi:       
        origin_location = "West"        
    elif angle == 3.0*np.pi/2.0:       
        origin_location = "South"   
    else: 
        origin_location = str("%0.2f" % (angle*180/np.pi)) + " degrees from East"
    
    return origin_location
      
def set_axes_height(ax, h):
    fig = ax.figure
    aw, ah = np.diff(ax.transAxes.transform([(0, 0), (1, 1)]), axis=0)[0]
    fw, fh = fig.get_size_inches()
    dpi = fig.get_dpi()
    scale = h / (ah / dpi)
    fig.set_size_inches(fw*scale, fh*scale, forward=True)      

def get_xtick(data, tick_interval, symmetric = None):
    """ Return tick

    data = sel_stat_radius
    tick_interval = PLT_XTICK
    sel_stat_radius, PLT_XTICK
    set_zeros = 1

    """

    min_range, max_range = np.min(data), np.max(data)

    # ticks
    num_ticks = np.ceil((max_range - min_range)/tick_interval).astype(int)
    ticks_val = np.linspace(min_range, min_range + num_ticks*tick_interval,  num_ticks + 1, dtype = np.float64)

    if symmetric is not None:
        if np.sign(min_range)*np.sign(max_range) < 0:
            # Symetric at zeros
            mask = np.argmin(np.abs(ticks_val))
            ticks_val[mask] = 0

        else:
            # Symetric at the middle
            mask = np.argmin(np.abs(ticks_val - np.mean(data)))
            ticks_val[mask] = 0

    # Fix ticks_val with max values of the data
    if ticks_val[-1] != max_range: ticks_val[-1] = max_range

    # Get tick label
    if symmetric is None:
        ticks_label = [num2text(val) for val in ticks_val]
    else:
#        ticks_val_sys = [val - min_range if val < 0 else max_range - val for val in ticks_val]
        ticks_val_sym = [val - min_range if val < 0 else val + np.abs(min_range) for val in ticks_val]
        ticks_label = [num2text(val) for val in ticks_val_sym]

    # return results
    return ticks_val, ticks_label, [min_range, max_range]

# In[get xtick]
def get_ytick(data, tick_interval, symmetric = None):
    """ Return tick

    data = y_range#[vmin, vmax]
    tick_interval = 120#0.5
    symmetric = 1
    data = y_lim
    tick_interval = PLT_YTICK

    """

    min_range, max_range = np.min(data), np.max(data)

    # ticks
    if (max_range - min_range) < tick_interval:
        ticks_val = np.array([min_range, (min_range + max_range)/2.0, max_range])
    else:

        if symmetric is None:

            num_ticks = np.ceil((max_range - min_range)/tick_interval).astype(int)
            ticks_val = np.linspace(min_range, min_range + num_ticks*tick_interval,  num_ticks + 1, dtype = np.float64)
            if ticks_val[-1] != max_range: ticks_val[-1] = max_range

        else:
            if np.sign(min_range)*np.sign(max_range) < 0:
                # Symetric at zeros
                sys_val = 0
            else:
                # Symetric at the middle
                sys_val = np.median(data)

            # Tick
            num_ticks = np.ceil((sys_val - min_range)/tick_interval).astype(int)
            ticks_val_lower = np.linspace(sys_val - num_ticks*tick_interval, sys_val, num_ticks + 1, dtype = np.float64)
            if ticks_val_lower[0] != min_range: ticks_val_lower[0] = min_range

            num_ticks = np.ceil((max_range - sys_val)/tick_interval).astype(int)
            ticks_val_upper = np.linspace(sys_val + tick_interval, sys_val + num_ticks*tick_interval, num_ticks, dtype = np.float64)
            if ticks_val_upper[-1] != max_range: ticks_val_upper[-1] = max_range

            # Combination
            ticks_val = np.append(ticks_val_lower, ticks_val_upper)

    # Remove tick closes to min and max
    if (ticks_val[-1] - ticks_val[-2]) < tick_interval/8: ticks_val = np.delete(ticks_val, -2)
    if (ticks_val[1] - ticks_val[0]) < tick_interval/8: ticks_val = np.delete(ticks_val, 1)

    # Tick label
    ticks_label = [num2text(val) for val in ticks_val]
#    if np.max(np.abs(ticks_val)) > 12:
#        ticks_label = [num2text(val) for val in ticks_val]
#    else:
#        ticks_label = [str('%.2f'%val) for val in ticks_val]

    # return results
    return ticks_val, ticks_label, [min_range, max_range]

# In[Select y-tick_internal]
def select_tick_interval(data, output_unit = 'imperial'):
    """This function is to select the interval between ticks
    
    """
    from Constant import constant
    set_constant = constant()
    data_range = np.max(data) - np.min(data)
    # Convert data range to imperial unt
    if output_unit.casefold() != 'imperial'.casefold():
        data_range = data_range/0.0254
    # Select interval based onimperial unit    
    if data_range <= 0.5:
        tick_interval = 0.125*set_constant["tick_size_level_4"] 
    elif (0.5 < data_range)&(data_range <= 1.0):    
        tick_interval = 0.25*set_constant["tick_size_level_4"] 
    elif (1.0 < data_range)&(data_range <= 2.0):
        tick_interval = 0.5*set_constant["tick_size_level_4"]         
    elif (2.0 < data_range)&(data_range <= 6.0):
        tick_interval = 1.0*set_constant["tick_size_level_4"] 
    elif (6.0 < data_range)&(data_range <= 12.0):
        tick_interval = 2.0*set_constant["tick_size_level_4"]         
    elif (12.0 < data_range)&(data_range <= 24.0):
        tick_interval = 3.0*set_constant["tick_size_level_4"] 
    elif (24.0 < data_range)&(data_range <= 36.0):
        tick_interval = 6.0*set_constant["tick_size_level_4"]
    else:
        tick_interval = 12.0*set_constant["tick_size_level_4"]
    
    if output_unit.casefold() == 'imperial'.casefold():
        tick_interval = tick_interval*meter2imperial()
    
    return tick_interval

# In[Generate a smaller images based on the current images]
def img_resize(img_path):
    """This function is to resize all images in a directory and its sub-directory

    """
    # Call library    
    import glob
    import os
    from PIL import Image

    files = glob.glob(img_path + '\**\*.jpg', recursive=True)
    
    basewidth = 750
    for file in files:
        # filename
        filename, file_extension = os.path.splitext(file)
        filename = filename + '_sm'
        newFile = filename + file_extension
        # read image
        im = Image.open(file)
        wpercent = (basewidth/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        im_resized = im.resize((basewidth,hsize), Image.ANTIALIAS)
        im_resized.save(newFile)
        del im_resized
      