#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:57:15 2018

@author: linhtruonghong
"""
import warnings
import numpy as np
import pandas as pd
from skimage.measure import ransac, CircleModel
from scipy import optimize
import scipy.stats as sp
from .segmentation import DBscan_clustering
#%%
def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def call_error_fit_model(x, y, xc, yc, rc):
    '''This function is to calculate errors from the fitting model and data'''
    
    ri = calc_R(x, y, xc, yc)
    ri_errors = ri - rc
    rms = np.sqrt(np.sum(ri_errors**2)/len(ri))
    
    return ri_errors, rms
    
def outlier_bound_error(confidence, ri_errors):
    '''This function is to calculate the bound error to identify whether the points inlier
    or outlier'''
    
    # Check input value
    if confidence > 1:
        confidence = 0.95
        
    ri_errors = abs(ri_errors)           
    m_error, se_error = np.mean(ri_errors), sp.sem(ri_errors)
    h = se_error * sp.t._ppf((1+confidence)/2., len(ri_errors)-1)
    upper_bound_error = m_error + h
    lower_bound_error = m_error - h
    
    return upper_bound_error, lower_bound_error
    
def leastsq_fit_circle(x,y):
    ''' This function is to fit the circle by using a least square method
    Demo:
        cell_ptc_ids = large_segment_cell_ptc_xyz.index
        cell_ptc_xyz = large_segment_cell_ptc_xyz[['x','y']]
    ''' 
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
# In[Ransac]

def outlier_removal_ransac(data, ransac_para = {'residual_threshold':0.1}):
    '''This function is to remove outlier points in linear 
    data = fit_error
    residual_threshold = 0.1
    '''
    # Call functions
    from sklearn import linear_model
    
    # Initialisation
    ransac = linear_model.RANSACRegressor(residual_threshold = ransac_para['residual_threshold'])
        
    # Check input
    if data.ndim == 1:
        data = data.reshape(-1,1)
        x = np.arange(data.shape[0]).reshape(-1,1)
        ransac.fit(x, data)
        
    else:
    
        ransac.fit(x, data)
        
    # Retrieve inlier

    return np.where(ransac.inlier_mask_==True)[0]
#%%
def ransac_fit_circle(model_ptc, model_tol = 0.1, outlier_method = {'band_width': 0.1}, outlier_option = 'hard'):
    ''' This function is to fit the circle by removing outlier points.
        If the circle radius of the consecutive iteration differs less than the model_tol
        the searching is terminate

    Demo:
        model_ptc = ptc.loc[grid_cell_ptc_ids,['x', 'y']]
        model_tol = 0.1
#        outlier_method = {'band_width': 0.1}
#        outlier_method = {'probability': 0.95}
#        outlier_method = {'clustering': 0.1}
        outlier_method = {'residual_threshold': 0.1}
        
        outlier_option = 'hard'
    '''
    
    
    # Check input method
    _outlier_method_para = ('band_width', 'residual_threshold', 'clustering')
    
    if not(list(outlier_method.keys())[0] in _outlier_method_para):   
        
        warnings.warn('The input method does not support in this function', DeprecationWarning, stacklevel=2)
        return
    
    # outlier option
    _outlier_option_para = ('soft', 'hard')
    
    if not(outlier_option in _outlier_option_para):
    
        warnings.warn('The input option does not support in this function', DeprecationWarning, stacklevel=2)
        return
    
    # Criteria
    max_iters = 100 
    inlier_ratio = 0.5

    # Initial parameters
    interaction = 0
    num_ptc = len(model_ptc)
    best_xc = 0
    best_yc = 0
    best_rc = np.inf
    initial_rc = np.inf
    
    # Initial inllier points
    model_inlier_ptc_ids = model_ptc.index
    fit_model_info = np.array([])
    
    fit_inlier_ptc_ids = model_inlier_ptc_ids.copy()
    
    while interaction < max_iters:
        
        # Fit a circle by least square
        fit_ptc_x = model_ptc.loc[fit_inlier_ptc_ids, ['x']].values.flatten()
        fit_ptc_y = model_ptc.loc[fit_inlier_ptc_ids, ['y']].values.flatten()
        xc, yc, rc, __, __, __ = leastsq_fit_circle(fit_ptc_x, fit_ptc_y)
        
        # Compute errors for all data set
        model_ptc_x = model_ptc[['x']].values.flatten()
        model_ptc_y = model_ptc[['y']].values.flatten()
        test_error, test_rms = call_error_fit_model(model_ptc_x, model_ptc_y, xc, yc, rc)
        
        # Assign information for the case of divergence
        fit_model_info = np.append(fit_model_info, [xc, yc, rc, test_rms])
        
        if (abs(initial_rc - rc) <= model_tol) | (num_ptc*inlier_ratio >= len(fit_inlier_ptc_ids)):
            
            # Convergence or degenerate
            best_xc, best_yc, best_rc = xc, yc, rc
            return best_xc, best_yc, best_rc, fit_inlier_ptc_ids
            
        else:
            
            # Update a circle radius for next iteration
            initial_rc = rc
            
            # Calculate an bound error
            if 'band_width' in outlier_method.keys(): # 
                
                # All points having errors within the buffer are inlier
                upper_bound_error, lower_bound_error = outlier_method['band_width'], - outlier_method['band_width']
                
                # Determine inlier points: only applied for outlier_method = band_width
                if outlier_option.casefold() == 'hard'.casefold():
                    
                    fit_inlier_ptc_ids = model_inlier_ptc_ids[(lower_bound_error <= test_error)&(test_error <= upper_bound_error)]
                    
                else:
                    
                    fit_inlier_ptc_ids = model_inlier_ptc_ids[test_error <= upper_bound_error]
                    
            elif 'residual_threshold' in outlier_method.keys():
                
                # Using ransac
#                confidence = outlier_method[_outlier_method_para[1]]
#                upper_bound_error, lower_bound_error = outlier_bound_error(confidence, np.abs(fit_error))
                inlier_ptc_local_ids = outlier_removal_ransac(test_error, ransac_para = outlier_method)
                fit_inlier_ptc_ids = model_inlier_ptc_ids[inlier_ptc_local_ids]
            
            else: #Clustering
                
                # The DB scan was used to cluster fit_error and the largest cluster is inlier
                eps = outlier_method[_outlier_method_para[2]]
                test_error = DBscan_clustering(test_error, eps = eps, min_pts = 10)  
                largest_group = test_error['label'].value_counts().idxmax()
                
                fit_inlier_ptc_ids = model_inlier_ptc_ids[test_error.loc[test_error['label'] == largest_group].index]
                
        # Next iterations    
        interaction += 1     
        
    # For case non_convergence
    if best_rc == np.inf:
        
        # The best model has the smallest root mean square error
        fit_model_info = fit_model_info.reshape(-1,4)
        mask = np.argmin(fit_model_info[:,3])
        best_xc, best_yc, best_rc = (fit_model_info[mask,0:3])
        
        # Calculate errors and the bound error
        model_ptc_x = model_ptc[['x']].values.flatten()
        model_ptc_y = model_ptc[['y']].values.flatten()
        
        # Calculate the errors for outlier removal
        test_error, __ = call_error_fit_model(model_ptc_x, model_ptc_y, best_xc, best_yc, best_rc)
        
            # Calculate an bound error
        if 'band_width' in outlier_method.keys(): # 
            
            # All points having errors within the buffer are inlier
            upper_bound_error, lower_bound_error = outlier_method['band_width'], - outlier_method['band_width']
            
            # Determine inlier points: only apply for band_width
            if outlier_option.casefold() == 'hard'.casefold():
                
                fit_inlier_ptc_ids = model_inlier_ptc_ids[(lower_bound_error <= test_error)&(test_error <= upper_bound_error)]
                
            else:
                
                fit_inlier_ptc_ids = model_inlier_ptc_ids[test_error <= upper_bound_error]
            
        elif 'residual_threshold' in outlier_method.keys():
            
            # Points having errors in a probability by xx% inlier
#            confidence = outlier_method[_outlier_method_para[1]]
#            upper_bound_error, lower_bound_error = outlier_bound_error(confidence, fit_error)
            
            # Using Ransac
            inlier_ptc_local_ids = outlier_removal_ransac(test_error, ransac_para = outlier_method)
            fit_inlier_ptc_ids = model_inlier_ptc_ids[inlier_ptc_local_ids]
            
        else:
            # The DB scan was used to cluster fit_error and the largest cluster is inlier
            eps = outlier_method[_outlier_method_para[2]]
            test_error = DBscan_clustering(test_error, eps = eps, min_pts = 10)  
            largest_group = test_error['label'].value_counts().idxmax()
            fit_inlier_ptc_ids = model_inlier_ptc_ids[test_error['label'] == largest_group]
            
        # Return results
        return best_xc, best_yc, best_rc, fit_inlier_ptc_ids

#%%
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
    test_error, test_rms = call_error_fit_model(model_ptc_x, model_ptc_y, xc, yc, rc)
    
    # Determinate inlier points
    model_ptc_ids = np.arange(model_ptc.shape[0]) 
    upper_bound_error, lower_bound_error = outlier_tolerance, - outlier_tolerance   
    fit_inlier_ptc_ids = model_ptc_ids[(lower_bound_error <= test_error)&(test_error <= upper_bound_error)]
            
    # Return results
    return xc, yc, rc, fit_inlier_ptc_ids 

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
    '''
#    
    from Utilities import PCA_normal_vector, dist_3Dpoint_surfaces

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
        plane_normal = PCA_normal_vector(fit_ptc, min_ptc = model_min_ptc)
        
        # Compute distances from the points to the fitting plane
        plane_center = np.mean(test_ptc, axis = 0)
        test_error = dist_3Dpoint_surfaces(test_ptc, plane_center, plane_normal)
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
        model_error = dist_3Dpoint_surfaces(model_ptc, best_model[:3], best_model[3:])
        
        inlier_ptc_ids = model_inlier_ptc_ids[np.abs(model_error) <= model_tol]
             
    # Return results
    return best_model, inlier_ptc_ids

# In[Determine the best fit function of the original profile]
def fit_profile(data, estimators):
    """This fuction is to compute the settelement of points based on different between the original profile
    and measured elevation.
    The original profile can determine based on either Ransac or best fit line
    
    Input:
        data            : [Mx2] - data use to estimate the original profile
        estimators      : two fitting function: Linear regression and Ransac
        
    Output:
        
        best_fit_funct  : The best fit function of the data
    
    Demo:
        
        data = sect_profile_ptc[:, [4,2]]
        
        
    """
    from sklearn.metrics import mean_squared_error
    
    # Calculate the score for each fit function
    score = []
    for  name, estimator in estimators:
        
        model = estimator
        model.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
        
        # Calculate error        
        mse = mean_squared_error(model.predict(data[:, 0].reshape(-1, 1)), data[:, 1].reshape(-1, 1))

        if name.casefold() == "ransac".casefold():
           
            inlier_ids = np.where(model.inlier_mask_ == True)[0]
            
            if inlier_ids.shape[0] >= 0.75*data.shape[0]:
                score = np.append(score, mse)
            else:
                score = np.append(score, np.inf)
        else:
            score = np.append(score, mse)
    
    # Determine the best fit function
    mask = np.argmin(score)
    best_fit_fuct = estimators[mask]
    
    # Return results
    return best_fit_fuct
# In[Fitting the cylinder]
#def ransac_cylinder(ptc, polar_grids, fit_model_min_ptc = 100, max_iters = 500, model_tol = 1e-2, inlier_ratio = 0.9,
#                    local_model_tol = 1.5e-2, searching_neighbour = {'range_search': 0.05}):
#    
#    '''This function is to determine the best fitting cylinder through 3D point cloud.
#    The method is based on the ransac method. However, the difference here is the approach 
#    to select the random points. The approach started with the polar ggrids to extract the number of cells
#    equal to the number of required points for fitting the cylinder. For each cells, one points randomlly selected 
#    from the cell. For each point, the normal vector was claculated by PCA. The direction and centre of 
#    the cylinder can be determined from the normal vectors, where the centre is the centroid of
#    the normal vector, while the cylinder direction is the normal of the fitting plane.
#    
#    ptc = Shell_wall[['x','y','z']].values
#    polar_grids = copy.deepcopy(Polar_Grids)
#    fit_model_min_ptc = 100
#    max_iters = 500
#    model_tol = 1e-2
#    inlier_ratio = 0.9
#    local_model_tol = 1.5e-2
#    searching_neighbour = {'range_search': 0.05}
#    
#    '''
#    # Call function
#    import pandas as pd
#    from Fit_Cylinder import estimate_cylinder_model, estimate_cylinder_error
#    
#    # Check input data
#    if isinstance(ptc, pd.DataFrame):
#        if any(item in ['x','y','z'] for item in ptc.columns):
#            
#            ptc = ptc[['x','y','z']].values
#                     
#        else:
#            raise ValueError('The input data does not contain x, y, z coordinates')
#
#    if not isinstance(searching_neighbour, dict):
#        raise ValueError('A searching method must be a dict containing key and value')
#
#    # Initial paramaters for Ransac cylinder
#    
#    iteraction = 0
#    best_fit_cylinder_para = None
#    best_fit_cylinder_error = np.inf
#    fit_cylinder_info = []
#    
#    grid_cell_ids = np.arange(len(polar_grids['ids'])) 
#    total_num_inlier_ptc = np.ceil(ptc.shape[0]*inlier_ratio).astype(int)
#    
#    # Ransac process
#    while iteraction < max_iters: 
#    
#        # Random generate cell ids for extracting points to fit the cylinder
#        
#        np.random.shuffle(grid_cell_ids)
#        random_cell_ids = grid_cell_ids[:fit_model_min_ptc]
#        fit_model_cell_ids = polar_grids['ids'][random_cell_ids]
#        
#        # Fit the cylinder
#        cylinder_parameters = estimate_cylinder_model(ptc, polar_grids, fit_model_cell_ids, model_tol = local_model_tol, searching_neighbour = searching_neighbour)
#        
#        if not(cylinder_parameters is None):
#            # Calculate errors of the fitting model
#            fit_cylinder_error, __ = estimate_cylinder_error(ptc, cylinder_parameters)
#            this_error = np.mean(np.abs(fit_cylinder_error))
#            
#            # Extract inlier ptc
#            mask = fit_cylinder_error <= model_tol
#            
#        else:
#            continue
#        
#        # Assign fitting information
#        fit_cylinder_info = np.append(fit_cylinder_info, cylinder_parameters)
#        fit_cylinder_info = np.append(fit_cylinder_info, this_error)
#         
#        # Check if the fitting model satisfying the conditions
#        if (len(np.where(mask == True)[0]) >= total_num_inlier_ptc) & (this_error < best_fit_cylinder_error):
#                
#            # The final model 
#            best_fit_cylinder_para = cylinder_parameters
#       
#        else:
#            
#            best_fit_cylinder_error = this_error
#           
#        # Increase iteration
#        iteraction += 1
#    
#    # Selecting the best model
#    if best_fit_cylinder_para is None:
#        
#        # The best model has the smallest absolute mean  error
#        fit_cylinder_info = fit_cylinder_info.reshape(-1,8)
#        mask = np.argmin(fit_cylinder_info[:,7])
#        
#        best_fit_cylinder_para = fit_cylinder_info[mask,:7]
#         
#    # Return results
#    return best_fit_cylinder_para

    
    
    
    
    
    
