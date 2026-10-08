# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:03:37 2021

@author: LinhTruongHong
"""
import numpy as np

# In[Calculate the first derivative
def first_derivative(x, y):
    """This function is to compute first derivative
    Parameters:
    ------------------------
        x                   : x-coordinate
        y                   : y - coordinate

    Return:
    ------------------------
        f_deriv             : first derivative

    Demo:
    ----
    """
    # Call libs

    # Check input variable
    x = np.array(x, copy = True).flatten()
    y = np.array(y, copy = True).flatten()

    f_deriv = np.full(x.shape[0], np.inf, dtype = np.float32)
    f_deriv[0] = (y[1] - y[0])/(x[1] - x[0])
    f_deriv[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    f_deriv[1:-1] = (y[1:-1] - y[0:-2])/(x[1:-1] - x[0:-2])

    return f_deriv

# In[Determine kernel score]
def kernel_density(data, bandwidth = 0.005, num_locs = 100):
    """This function is to determine density of data
    Parameters:
    ------------------------
        data                : 1d array
        bandwidth           : scalar
        num_locs            : scalar

    Return:
    ------------------------
        kde_data_density    : kde
        max_peak_loc        : localcation of the highest peak
        max_peak_val        : values of the higest peak

    Demo:

    """
    # Call libraries

    from sklearn.neighbors import KernelDensity

    # Generate KDE
    range_locs = np.linspace(min(data), max(data), num_locs)[:, None]
    kde_data = KernelDensity(kernel='epanechnikov', bandwidth = bandwidth).fit(data.reshape(-1, 1))
    kde_data_log = kde_data.score_samples(range_locs)
    kde_data_density = np.exp(kde_data_log)

    # Estimate the location of the highest peak
    max_peak_loc_id = np.argmax(kde_data_density)
    max_peak_loc = range_locs[max_peak_loc_id]
    max_peak_val = kde_data_density[max_peak_loc_id]

    return kde_data_density, max_peak_loc, max_peak_val

# In[Determine kernel score]
def kernel_density_2peaks(data, bandwidth = 0.005, num_locs = 100):
    """This function is to determine 2 largest peaks of data based on density
    Parameters:
    ------------------------
        data                : 1d array
        bandwidth           : scalar
        num_locs            : scalar

    Return:
    ------------------------
        kde_data_density    : kde
        max_peak_loc        : localcation of the highest peak
        max_peak_val        : values of the higest peak

    Demo:

    """
    # Call libraries
    import heapq
    from scipy.signal import find_peaks
    from sklearn.neighbors import KernelDensity

    # Generate KDE
    range_locs = np.linspace(min(data), max(data), num_locs)[:, None]
    kde_data = KernelDensity(kernel='epanechnikov', bandwidth = bandwidth).fit(data.reshape(-1, 1))
    kde_data_log = kde_data.score_samples(range_locs)
    kde_data_density = np.exp(kde_data_log)

    # Estimate the location of the peaks
    max_peak_loc_ids, _ = find_peaks(kde_data_density, height = 0)
    max_peak_loc = range_locs[max_peak_loc_ids]
    max_peak_val = kde_data_density[max_peak_loc_ids]

    # Extract the first two highest peaks

    two_peaks = heapq.nlargest(2, max_peak_val)
    mask = np.in1d(max_peak_val, two_peaks)
    max_peak_loc_ids = max_peak_loc_ids[mask]
    max_peak_loc = max_peak_loc[mask]
    max_peak_val = max_peak_val[mask]

    # Filter coincide peaks
    if max_peak_loc.shape[0] == 2:
        if np.max(max_peak_val) >= 2.0*np.min(max_peak_val): #np.abs(max_peak_loc[1] - max_peak_loc[0]) <= 2.0*bandwidth:
            max_peak_loc[0] = max_peak_loc[np.argmax(max_peak_val)]
            max_peak_val[0] = max_peak_val[np.argmax(max_peak_val)]
            max_peak_loc[1] = np.inf
            max_peak_val[1] = 0.0#np.inf
        else:
            max_peak_loc = max_peak_loc[np.argsort(max_peak_val)][::-1]
            max_peak_val = max_peak_val[np.argsort(max_peak_val)][::-1]


    return kde_data_density, max_peak_loc, max_peak_val

# In[]
def find_orthogonal_pattern(data, range_angle = [0.0, 45], delta_angle = np.deg2rad(5.0), bandwidth = 0.005, num_locs = 100):
    """This function is to determine orthogonal pattern
    Parameters:
    ------------------------
        data                : [Nx2]
        range_angle         : range angle to genrate a rotation kde
        bandwidth           : a bandwidth
        num_locs            : number of locations of kde
    Return:
    ------------------------
        data_peak_dir1      : [Nx2] [local_id, peak_id] in a direction 1
        data_peak_dir2      : [Nx2] [local_id, peak_id] in a direction 2

    Demo:
    ------------------------

    data = 0.5*(voxelgrid.bounds[voxel_linear_ids,0:2]+voxelgrid.bounds[voxel_linear_ids,3:5])
    range_angle = [0.0, 60]
    delta_angle = np.deg2rad(5.0)
    bandwidth = voxelgrid.size
    num_locs = 100

    import os
    curdir = "//Users//linhtruonghong//Dropbox//On going//Acernis//Project 1//"
    file_name = "tes_zvoxel - Cloud.txt"
    file_path = os.path.join(curdir, file_name)
    data = np.loadtxt(file_path, delimiter = " ")
    data = data[:,:2]
    range_angle = [0.0, 60]
    delta_angle = np.deg2rad(5.0)
    bandwidth = .5*0.2
    num_locs = 100

    """
    # Call libraries
    from Utils_Geometry import  cart2pol, pol2cart
    # Constant
    if delta_angle is None:
        delta_angle = np.deg2rad(2.0)

    # Generate a rotation
    range_theta = np.arange(np.deg2rad(range_angle[0]), np.deg2rad(range_angle[1]) + delta_angle, delta_angle)

    # Translate the point to a local coordinate at the center of the section
    data_cent = 0.5*(np.min(data, axis=0) + np.max(data, axis=0))
    local_data = data - data_cent
    local_rho, local_theta = cart2pol(local_data[:, 0], local_data[:, 1])

    score = np.full((range_theta.shape[0], 8), np.inf)
    # Convert to a polar coordinate system
    for count, theta in enumerate(range_theta):

        # Convert polar to cartesian
        local_incr_theta = local_theta + theta
        incr_local_x, incr_local_y = pol2cart(local_rho, local_incr_theta)
        incr_local_data = np.vstack((incr_local_x, incr_local_y)).T
        del local_incr_theta, incr_local_x, incr_local_y

        # Generate kernel density along kde_dir
        __, x_max_peak_loc, x_max_peak_val = kernel_density_2peaks(incr_local_data[:, 0], bandwidth=bandwidth, num_locs=num_locs)
        __, y_max_peak_loc, y_max_peak_val = kernel_density_2peaks(incr_local_data[:, 1], bandwidth=bandwidth, num_locs=num_locs)

        # Update the score function
        score[count, :2] = np.array([count,theta])
        score[count, [2,3]] = x_max_peak_loc.flatten()
        score[count, [4,5]] = y_max_peak_loc.flatten()
        score[count, 6] = x_max_peak_val[0]
        score[count, 7] = y_max_peak_val[0]

    # Find an optimal rot_angle
    mask = np.any(np.isinf(score), axis=1)
    score = score[~mask]
    mask = np.argmax(score[:, 6] + score[:, 7])
    opt_theta = score[mask, 1]

    # Get local ptc corresponding to the optimal location
    local_incr_theta = local_theta + opt_theta
    opt_local_x, opt_local_y = pol2cart(local_rho, local_incr_theta)
    opt_local_data = np.vstack((opt_local_x, opt_local_y)).T

#    # Find the points indices within the largest peak
    data_peak_dir1 = get_data_peak(opt_local_data[:, 0], bandwidth=bandwidth)
    data_peak_dir2 = get_data_peak(opt_local_data[:, 1], bandwidth=bandwidth)

    # return
    return data_peak_dir1, data_peak_dir2
# In[]
def get_data_peak(data, bandwidth = 0.005, num_locs = 100):
    """This function is to segment the points within the valley-peak-valley
    Parameters:
    ------------------------
        data                : [Nx1]
        bandwidth           : a bandwidth
        num_locs            : number of locations of kde
    Return:
    ------------------------
        data_peak_ids       : [Nx2] [local_id, peak_id] in a direction 1

    Demo:
    ------------------------

    """
    # Find bounds of each peaks
    peak_shapes = peak_shape_range(data, bandwidth = bandwidth, num_locs = num_locs)

    # Extract the points within the peak
    data_peak_ids = np.full((data.shape[0], 2), 0, dtype=np.uint32)
    data_peak_ids[:,0] = np.arange(data.shape[0])
    for count, peak_shape in enumerate(peak_shapes):
        # Retrieve the points within the peak
        mask = (peak_shape[1] <= data) & (data <= peak_shape[2])
        local_ids = np.where(mask == True)[0]

        # Update peak id for data
        data_peak_ids[local_ids,1] = count + 1

    return data_peak_ids

# In[Determine the range of the peak shape]
def peak_shape_range(data, bandwidth = 0.005, num_locs = 100):
    """This function is to determine a range of the peak shape, which is thee valley peak valley
    Parameters:
    ------------------------
        data                : [Nx1]
        bandwidth           : a bandwidth
        num_locs            : number of locations of kde
    Return:
    ------------------------
        peak_shape         : [Nx3] [peak location, left valley location, right valley location]

    Demo:
    ------------------------
    data = opt_local_data[:,0]

    """
    # Call libraries
    from sklearn.neighbors import KernelDensity
    from scipy.signal import find_peaks

    # Generate kde
    range_locs = np.linspace(min(data), max(data), num_locs)[:, None]
    kde_data = KernelDensity(kernel='epanechnikov', bandwidth = bandwidth).fit(data.reshape(-1, 1))
    kde_data_log = kde_data.score_samples(range_locs)
    kde_data_density = np.exp(kde_data_log)

    # Find peak
    loc_ids = np.arange(num_locs)
    peak_ids, __ = find_peaks(kde_data_density)

    # Calculate a first derivative
    f_deriv_kde = first_derivative(range_locs, kde_data_density)

    # Preallocation
    peak_shape = np.full((peak_ids.shape[0],3), np.inf)
    peak_shape[:,0] = range_locs[peak_ids].ravel()

    # Searching shape for each peak
    for count, peak_id in enumerate(peak_ids):

        # Searching points on the left
        if peak_id == loc_ids[0]:
            left_valley_id = loc_ids[0]
        else:
            # Get left ids of the points
            left_ids = loc_ids[loc_ids < peak_id]

            # Extract the points with a negative values of first derivative of kde
            left_local_ids = np.where(f_deriv_kde[left_ids] <= 0)[0]
            if left_local_ids.shape[0] > 0:
                left_valley_id = left_ids[np.max(left_local_ids)]
            else:
                left_valley_id = left_ids[0]
        if left_valley_id == loc_ids[1]: #Adjust because the first derivative of the first and second points equal
            left_valley_id = loc_ids[0]

        # Get left point -> the closest valley: the derivative is calculated by a back point f(i) = y(i) - y(i-1)
        left_peak_shape_loc = range_locs[left_valley_id]

        # Searching points on the right
        if peak_id == loc_ids[-1]:
            right_valley_id = loc_ids[-1]
        else:
            # Get left ids of the points
            right_ids = loc_ids[loc_ids > peak_id]
            # Extract the points with a negative values of first derivative of kde
            right_local_ids = np.where(f_deriv_kde[right_ids] >= 0)[0]
            if right_local_ids.shape[0] > 0:
                right_valley_id = right_ids[np.min(right_local_ids)]
            else:
                right_valley_id = right_ids[-1]

        # Determine inflexion point in a right side, which is the closest valley:
        if (right_valley_id != loc_ids[-1]) & (right_valley_id != peak_id + 1):
            right_valley_id = right_valley_id - 1

        # Get left point
        right_peak_shape_loc = range_locs[right_valley_id]

        # Assign the shape
        peak_shape[count,1] = left_peak_shape_loc
        peak_shape[count,2] = right_peak_shape_loc

    # return
    return peak_shape
