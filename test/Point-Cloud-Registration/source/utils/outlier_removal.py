#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:47:42 2018

@author: linhtruonghong
"""

# Noise removal
import warnings
import time
import numpy as np
import pandas as pd
import copy
from itertools import chain
import scipy.stats as sp
from .fitting import ransac_circle, leastsq_fit_circle
from .data_structures import grid1d, grid_extraction, update_data_structure
from .segmentation import segmentation1d, DBscan_clustering
from .utilities import global_cart_2_local_polar

#%%
def noise_removal_fit_circle(ptc, shell_wall, nonshell_wall, scan_type, interval_thickness = 0.2, direction = 'z',
                             fit_model_error = 0.01, outlier_tolerance = 0.1, shell_radius_tolerance = 0.15):
    ''' This function is to remove data points which are not on the shell wall.
    This is base on assumption that the points having fitting errors larger than
    the threshold or the fitting error out of the confidence level to be outlier
    points.

    Input variables:
    ptc = df_ptc.copy()
    shell_wall = shell_wall.copy()
    nonshell_cell = nonshell_wall.copy()
    scan_type = scan_type
    interval_thickness = INTERVAL_THICKNESS
    direction = 'z'
    fit_model_error = SHELL_FIT_MODEL_ERROR
    outlier_tolerance = SHELL_COARSE_OUTLIER_REMOVAL
    shell_radius_tolerance = SHELL_RADIUS_TOLERANCE
    

    

    '''
    #%% Call libraries
    from Utilities import cart2pol
    
    # Initialization
    if isinstance(ptc, pd.DataFrame):

        if any(item in ['x','y','z'] for item in ptc.columns):

            ptc_xyz = ptc[['x','y','z']].values

        else:

            raise ValueError('The input data does not contain x, y, z coordinates')
    elif isinstance(ptc, np.ndarray):

        ptc_xyz = ptc[:, :3]

    else:

        raise ValueError('The input variable ptc does not support in this function')

    # For a direction
    if direction.casefold() == 'x'.casefold():
        direction = 0
    elif direction.casefold() == 'y'.casefold():
        direction = 1
    elif direction.casefold() == 'z'.casefold():
        direction = 2
    else:
        raise ValueError('Input direction does not support')

    shell_wall = shell_wall.copy()
    nonshell_wall= nonshell_wall.copy()

    #% Extract points's coordinates in the shell_cell
    shell_wall_ptc_ids = list(chain.from_iterable(shell_wall['ptc_ids']))
    shell_wall_ptc_ids = np.asarray(shell_wall_ptc_ids, dtype = np.int32)
    shell_wall_ptc_dir = ptc_xyz[shell_wall_ptc_ids, direction]
    
    #% Extract points's coordinates in the shell_cell
    nonshell_wall_ptc_ids = list(chain.from_iterable(nonshell_wall['ptc_ids']))
    nonshell_wall_ptc_ids = np.asarray(nonshell_wall_ptc_ids, dtype = np.int32)
    nonshell_wall_ptc_dir = ptc_xyz[nonshell_wall_ptc_ids, direction]

    # Establish the grids for the shell
    ptc_dir_min = np.min(shell_wall_ptc_dir)
    ptc_dir_max = np.max(shell_wall_ptc_dir)

#    df_shell_wall_ptc_dir = pd.DataFrame(shell_wall_ptc_dir, columns = ['z'], index = shell_wall_ptc_ids)
#    shell_grids = grid1d(df_shell_wall_ptc_dir, ptc_dir_min, ptc_dir_max, interval_thickness, 'z')
    shell_grids = grid1d(shell_wall_ptc_dir, ptc_dir_min, ptc_dir_max, interval_thickness, 'z')
    shell_grids = grid_extraction(shell_grids, 1)
#    del df_shell_wall_ptc_dir

    # Establish grids for non_shell
#    df_nonshell_wall_ptc_dir = pd.DataFrame(nonshell_wall_ptc_dir, columns = ['z'], index = nonshell_wall_ptc_ids)
#    nonshell_grids = grid1d(df_nonshell_wall_ptc_dir, ptc_dir_min, ptc_dir_max, interval_thickness, 'z')
    nonshell_grids = grid1d(nonshell_wall_ptc_dir, ptc_dir_min, ptc_dir_max, interval_thickness, 'z')
#    del df_nonshell_wall_ptc_dir

    #% Remove outlier points in each grid
    # Preallocation
    shell_grid_info = np.full((shell_grids['ids'].shape[0], 5), np.inf, dtype = np.float32)
    shell_grid_info[:,0] = shell_grids['ids'].flatten()
    outlier_ptc_ids = dict()
    ground_ptc_ids = np.array([], dtype = np.int32)
    
    #%%

    for i, grid_cell_ptc_local_ids in enumerate(shell_grids['ptc_ids']):
#        print(i)
        #% Remove ground points

#        s_time = time.time()
        
        grid_cell_ptc_global_ids = shell_wall_ptc_ids[grid_cell_ptc_local_ids]
        grid_cell_ptc_xyz  = ptc_xyz[grid_cell_ptc_global_ids, :3]
        
        if grid_cell_ptc_global_ids.shape[0] >= 8000: # (0.2x1m with the sampling step of 0.005)
        
            #
            hist_rho = np.histogram(grid_cell_ptc_xyz[:, 2], bins = 10, density = False)

            if (np.max(hist_rho[0]) - np.min(hist_rho[0]) >= 0.2*np.max(hist_rho[0])) :
                
#                print("grid:", i)
#                ss_time = time.time()
                grid_cell_ground_ptc_local_ids = np.array([], dtype = np.int32)
            
                for bin_id, val in enumerate(hist_rho[0]):
                    if val >= np.mean(hist_rho[0]):
                        mask = (hist_rho[1][bin_id] <= grid_cell_ptc_xyz[:, 2])&(grid_cell_ptc_xyz[:, 2] <= hist_rho[1][bin_id + 1])
                        grid_cell_bin_ptc_local_ids = np.where(mask == True)[0]
                        grid_cell_bin_ptc = grid_cell_ptc_xyz[grid_cell_bin_ptc_local_ids] 
                        
                        # Ground point removal
                        __, bin_ground_ptc_local_ids = ground_point_removal(grid_cell_ptc_xyz, grid_cell_bin_ptc, angle_threshold = np.pi/4.0, searching_neighbour= {'knn_search': 25})
                        
                        # Update for local ids of grid_cell
                        grid_cell_bin_ground_ptc_local_ids = grid_cell_bin_ptc_local_ids[bin_ground_ptc_local_ids]
                        
                        # Update nonground and ground points of the grid_cell
                        grid_cell_ground_ptc_local_ids = np.append(grid_cell_ground_ptc_local_ids, grid_cell_bin_ground_ptc_local_ids)
                 
#                print("time for ground removal", time.time() - ss_time)
                # Update ground and wall point ids
                if grid_cell_ground_ptc_local_ids.shape[0] > 0:
                    # Ground points
                    grid_cell_ground_ptc_global_ids = grid_cell_ptc_global_ids[grid_cell_ground_ptc_local_ids]
                    ground_ptc_ids = np.append(ground_ptc_ids, grid_cell_ground_ptc_global_ids)
                    
                    # Wall point
                    grid_cell_wall_ptc_global_ids = np.setdiff1d(grid_cell_ptc_global_ids, grid_cell_ground_ptc_global_ids)
                else:
                    grid_cell_wall_ptc_global_ids = grid_cell_ptc_global_ids
    
            else:
                grid_cell_wall_ptc_global_ids = grid_cell_ptc_global_ids

            # Remove Outlier points through fitting circle
            grid_cell_wall_ptc_xyz = ptc_xyz[grid_cell_wall_ptc_global_ids, :3]
        
            # Estimate the fit circle
            xc, yc, rc, local_inlier_ptc_ids = ransac_circle(grid_cell_wall_ptc_xyz[:, :2], fit_model_error, outlier_tolerance, max_iters = 100)
            zc = np.mean(grid_cell_wall_ptc_xyz[local_inlier_ptc_ids, 2])
            
            wall_ptc_ids = grid_cell_wall_ptc_global_ids[local_inlier_ptc_ids]
            
            # Add command to retrieve outlier point indices from the ransac circle here
            outlier_ptc_ids[i] = np.setdiff1d(grid_cell_wall_ptc_global_ids, wall_ptc_ids)
            
            del grid_cell_wall_ptc_global_ids, local_inlier_ptc_ids
            
            # Retrieve the points within non_shell
            if nonshell_grids['prop'][i]:
                # Retrieve point indices: local -> global
                nonshell_grid_ptc_local_ids = nonshell_grids['ptc_ids'][i]               
                nonshell_grid_ptc_ids = nonshell_wall_ptc_ids[nonshell_grid_ptc_local_ids]
                
                # Retrieve point coordinate and convert 2 polar
                nonshell_grid_ptc = ptc_xyz[nonshell_grid_ptc_ids, :2]
                nonshell_grid_ptc_rho, __ = cart2pol(nonshell_grid_ptc[:,0] - xc, nonshell_grid_ptc[:,1] - yc)
                
                # Filtering points within the buffer
                mask = (rc - outlier_tolerance <= nonshell_grid_ptc_rho) & (nonshell_grid_ptc_rho <= rc + outlier_tolerance)
                nonshell_grid_add_ptc_ids = nonshell_grid_ptc_ids[mask]

                # Update nonShell
                nonshell_grid_remain_ptc_ids = np.setdiff1d(nonshell_grid_ptc_ids, nonshell_grid_add_ptc_ids)
                nonshell_grids['ptc_ids'][i] = nonshell_grid_remain_ptc_ids

            # If additional points are available
            if 'nonshell_grid_add_ptc_ids' in locals():
                wall_ptc_ids = np.concatenate((wall_ptc_ids, nonshell_grid_add_ptc_ids))
                
            # Update inlier points
            shell_grids['ptc_ids'][i] = wall_ptc_ids

        else:
            shell_grids['prop'][i] = False
            shell_grids['ptc_ids'][i] = None
            xc, yc, zc, rc = 0.0, 0.0, 0.0, 0.0

        # Update shell grid info
        shell_grid_info[i,1:] = [xc,yc,zc, rc]
        
#        print("time for each grid {:d} is {:.3f} s".format(i, time.time() - s_time))
    #%% Update shell grids
    shell_grid_ids = shell_grids["ids"][shell_grids["prop"]]

    # Remove all grids having the radius significantly different, in the confidence by 0.95
    mask = np.in1d(shell_grid_info[:,0], shell_grid_ids)
    shell_radius = shell_grid_info[mask, 4]
    
#    print(shell_radius)
#    print('radius tolerance', shell_radius_tolerance)

    # Clustering through DBscan
    db_shell_radius, db_nonshell_radius = DBscan_clustering(shell_radius, eps = shell_radius_tolerance, min_pts = np.ceil(0.25*shell_radius.shape[0]))
    
    # Extract large group (shell wall) and small group (non shell wall)
    largest_group = db_shell_radius['label'].value_counts().idxmax()
    mask = db_shell_radius['label'].values == largest_group
    largest_grid_ids = db_shell_radius.loc[mask].index
    
    # For a small group
    shell_grid_radius = db_shell_radius.loc[mask, ["val"]].values
    small_grid_ids = db_shell_radius.loc[~mask].index
    small_grid_ids = shell_grid_ids[small_grid_ids] # Update grid index with shell_grids

    # Update -> Shell wall
    shell_large_grid_info = shell_grid_info[largest_grid_ids]
    shell_wall_grids = update_data_structure(shell_grids, largest_grid_ids)
    
    # Update -> non Shell wall
    if small_grid_ids.shape[0] > 0:
        shell_wall_small_grid_ptc_ids = list(chain.from_iterable(shell_grids['ptc_ids'][ids] for ids in small_grid_ids))
    
    # Non-segment
#    print("Non_radius:", nonshell_radius)
    non_segment_ids = db_nonshell_radius.index
    non_segment_ids = shell_grid_ids[non_segment_ids] # Update grid index with shell_grids
    if non_segment_ids.shape[0] > 0:
        non_segment_grid_ptc_ids = list(chain.from_iterable(shell_grids['ptc_ids'][ids] for ids in non_segment_ids))
                  
    # Update all non shell wall points: Outlier, ground and small regions in the wall
    outlier_ptc_ids = list(chain.from_iterable(outlier_ptc_ids[ids] for ids in outlier_ptc_ids.keys()))
    
    all_outlier_ptc_ids = np.append(outlier_ptc_ids, ground_ptc_ids)
    if "shell_wall_small_grid_ptc_ids" in locals():
        all_outlier_ptc_ids = np.append(all_outlier_ptc_ids, shell_wall_small_grid_ptc_ids)
        all_outlier_ptc_ids = np.asarray(all_outlier_ptc_ids, dtype = np.int32)
    if "non_segment_grid_ptc_ids" in locals():
        all_outlier_ptc_ids = np.append(all_outlier_ptc_ids, non_segment_grid_ptc_ids)
        all_outlier_ptc_ids = np.asarray(all_outlier_ptc_ids, dtype = np.int32)
    
    #%% Return results
    return shell_wall_grids, nonshell_grids, shell_large_grid_info, all_outlier_ptc_ids, shell_grid_radius

# In[Noise removal based on cylinder grids]
def noise_removal_cylinder_grids(shell_cell_ptc, cylinder_cell, nbins = 10, outlier_option = 'hard'):

    ''' This function is to remove outlier points in the cylinder or elipse or circle.
    This is based on an assumption that the points out of the shell if they having the radii
    are significant difference

    shell_cell_ptc = copy.deepcopy(Shell_Cell_ptc)

    cylinder_cell = copy.deepcopy(Cylinder_Cell)

    nbins = 10
    option = 'hard'

    '''
    _option_para = ('soft', 'hard')
    if not(outlier_option in _option_para):

        warnings.warn('The input option does not support in this function', DeprecationWarning, stacklevel=2)
        return

    # Initialization
    shell_cell_ptc = copy.deepcopy(shell_cell_ptc)
    cylinder_cell = copy.deepcopy(cylinder_cell)

    # Total a number of cylinder cells
    total_grid_cells = cylinder_cell['ids'].shape[0]

    for i in range(0, total_grid_cells):

        #% Retrieve the radius of each point in the cell
        cylinder_cell_ptc_ids = cylinder_cell['ptc_ids'][i]
        cylinder_cell_ptc_rho = shell_cell_ptc.loc[cylinder_cell_ptc_ids,['rho']]

        # Difference of the radii
        diff_cylinder_cell_ptc_rho = np.max(cylinder_cell_ptc_rho.values) - np.min(cylinder_cell_ptc_rho.values)

        # Large difference -> noise data
        if diff_cylinder_cell_ptc_rho >= 0.01: #Can set as the noise level

            #% Using a histogram to detect the density
            hist_rho = np.histogram(cylinder_cell_ptc_rho, bins = nbins, density = True)

            bin_width = hist_rho[1][1] - hist_rho[1][0]
            max_bins_id = np.argmax(hist_rho[0])
            max_bins_edges = (hist_rho[1][max_bins_id]-bin_width, hist_rho[1][max_bins_id+1]+bin_width)

            # Determine outlier and inlier points
            if outlier_option.casefold() == 'hard'.casefold():

                # The points having the radius less than the upper edge of the bin having largest density
                mask = (max_bins_edges[0] <= cylinder_cell_ptc_rho)&(cylinder_cell_ptc_rho <= max_bins_edges[1])

            else:

                # The points having the radius within the bin having largest density
                mask = cylinder_cell_ptc_rho <= max_bins_edges[1]

            # Extract inllier points
            cylinder_cell_ptc_ids = cylinder_cell_ptc_rho.index[np.where(mask==True)[0]]

        # Update the cylinder cells

        if cylinder_cell_ptc_ids.shape[0] > 0:

            cylinder_cell['ptc_ids'][i] = cylinder_cell_ptc_ids

        else:

            cylinder_cell['prop'][i] = False
            cylinder_cell['ptc_ids'][i] = None

    #% Return the result
    return cylinder_cell

# In[Remove a point cloud of the rail of the Shell]

def elevation_ourlier_removal(cylinder_cell, shell_cell_ptc, length_threshold = 1.0, buffer_band = 0.1):


    '''This function is aimed to remove the rail of the Shell. This is based on
    an observation that the points of the rails often located within the cell that
    do not connect to the Shell wall. Those cells are considered as the float cells,
    which located in the short segment.

    Input:
    cylinder_cell = Shell_Cylinder.copy()
    shell_cell_ptc = Shell_ptc
    length_threshold = 1.0 #The segment length less than the length threshold is short
    buffer_band = 0.1 # band_with for estimating atop of the shell

    '''

    # Retrieve grids
    total_grid_cell = np.max(cylinder_cell['ids'][:,1])
    cylinder_grid_interval = cylinder_cell['bounds'][0,3] - cylinder_cell['bounds'][0,0]

    # Preallocation #[grid_id, z_min, z_max]
    cylinder_grid_height = np.zeros((total_grid_cell+1,3), dtype = np.float)

    # Remove the cell containg the point cloud of the rails
    for i in range(0, total_grid_cell+1):

        cylinder_grid_cell_ids = np.where(np.in1d(cylinder_cell['ids'][:,1], i)==True)[0]

        # Establish data structure stored all cells in the cylinder grid
        cylinder_grid_cell = dict()
        cylinder_grid_cell['ids'] = cylinder_cell['ids'][cylinder_grid_cell_ids,0].reshape(-1,1)
        cylinder_grid_cell['prop'] = cylinder_cell['prop'][cylinder_grid_cell_ids]
        cylinder_grid_cell['ptc_ids'] = [cylinder_cell['ptc_ids'][ids] for ids in cylinder_grid_cell_ids]

        # Segment the cells in the cylinder grid
        segment_region = segmentation1d(cylinder_grid_cell, cylinder_grid_interval, length_threshold)

        #% Remove points in the small cell
        cylinder_grid_cell_local_ids = np.arange(segment_region.shape[0])
        segment_region = np.hstack((cylinder_grid_cell_local_ids.reshape(-1,1),segment_region))
        mask = segment_region[:,2] > 0
        large_segment_cell_ids = segment_region[mask,0]

        # Retrieve the a point cloud on atop of the cylider grid
        if large_segment_cell_ids.shape[0] > 0:

            large_segment_ptc_ids = list(chain.from_iterable(cylinder_grid_cell['ptc_ids'][j] for j in large_segment_cell_ids))
            large_segment_ptc_z = shell_cell_ptc.loc[large_segment_ptc_ids,['z']].values

            # Update cell_height
            cell_z_min, cell_z_max =  np.min(large_segment_ptc_z), np.max(large_segment_ptc_z)
            cylinder_grid_height[i] = [i, cell_z_min, cell_z_max]

    # Estimate atop of the shellby using a historame
    nbins = np.ceil(np.max(cylinder_grid_height[:,2])/buffer_band)
    hist_height = np.histogram(cylinder_grid_height[:,2], bins = int(nbins), density = True)

    # Determine the peak
#    bin_width = hist_height[1][1] - hist_height[1][0]
    max_bins_id = np.argmax(hist_height[0])

    # Determine the bopund
    upper_bins_edges = hist_height[1][max_bins_id+1] #+ bin_width/2.0
    lower_bin_edges = hist_height[1][max_bins_id]
    mask = (lower_bin_edges <=cylinder_grid_height[:,2])&(cylinder_grid_height[:,2] <= upper_bins_edges)
    shell_ptc_z = cylinder_grid_height[mask,2]

    # Return the result

    return np.mean(shell_ptc_z), sp.sem(shell_ptc_z) #Mean and standard error

# In[Remove the points on the top and bottom shell]
def ground_point_removal(ptc, target_ptc, angle_threshold = np.pi/4.0, searching_neighbour= {'knn_search': 25}):

    '''This function is to remove points which is lying on nearly horizontal plane
    (not vertical), which can be on the bottom or top of the shell.
    
    Input:
        
    Output:
        
    Demo:
        
    ptc  = grid_cell_ptc_xyz.copy()

    searching_neighbour= {'range_search': 0.1}

#    searching_neighbour= {'knn_search': 25}
    '''

    from sklearn.neighbors import KDTree
    from Utilities import cosin3DVector, PCA_normal_vector

    # Check ptc
    if isinstance(ptc, pd.DataFrame):

        ptc = ptc[['x','y','z']].values

    elif isinstance(ptc, np.ndarray):

        ptc = ptc[:, :3]

    else:

        raise ValueError('Input direction does not support')
    
    # Check target ptc
    if isinstance(ptc, pd.DataFrame):

        target_ptc = target_ptc[['x','y','z']].values

    elif isinstance(ptc, np.ndarray):

        target_ptc = target_ptc[:, :3]

    else:

        raise ValueError('Input direction does not support')

    # Initialise variables
#    ptc_ids = np.arange(ptc.shape[0])
    target_ptc_ids = np.arange(target_ptc.shape[0])
    z_axis_normal_vector = np.array([0.0, 0.0, 1.0]).reshape(-1,3)
    cosin_threshold = np.cos(angle_threshold)

    # Searching neighbour points
    leaf_size = 50
    tree = KDTree(ptc, leaf_size = leaf_size)

    if 'range_search' in searching_neighbour.keys():
        # Range search
        searching_para = searching_neighbour['range_search'] if searching_neighbour['range_search'] > 0 else 0.1
        neighbour_ptc_ids = tree.query_radius(target_ptc, r = searching_para)

    elif 'knn_search' in searching_neighbour.keys():
		# kNN search
        searching_para = int(searching_neighbour['knn_search']) if int(searching_neighbour['knn_search']) > 5 else 20
        if searching_para >= target_ptc.shape[0]: 
            searching_para = target_ptc.shape[0] - 1
            
        __, neighbour_ptc_ids = tree.query(target_ptc, k = searching_para)

    else:

        raise ValueError('The searching method does not support')

    # Compute a normal vector
    target_ptc_normal_vector = np.zeros((target_ptc.shape[0], 3), dtype = float)

    for i in range(target_ptc.shape[0]):

        target_ptc_i_neighbour_ptc_ids = neighbour_ptc_ids[i]
#        ptc_i_neighbour_ptc_ids = np.setdiff1d(ptc_i_nei1zghbour_ptc_ids, i)
        normal_vector = PCA_normal_vector(ptc[target_ptc_i_neighbour_ptc_ids], min_ptc = 5)

        # Accumulative result
        target_ptc_normal_vector[i] = normal_vector

    # Remove infinite rows

    mask = np.isinf(target_ptc_normal_vector[:,0])
    target_ptc_normal_vector = target_ptc_normal_vector[~mask]
    target_ptc_ids = target_ptc_ids[~mask]

    # Compute a deviation between the normal vector and nz
    cosin_ptc_nn_ptc = cosin3DVector(z_axis_normal_vector, target_ptc_normal_vector)

    # Extract non_ground point ids in global based on index in ptc
    mask = np.abs(cosin_ptc_nn_ptc) <= cosin_threshold
#    non_ground_ptc_ids = ptc_ids[np.where(mask == True)[0]]
    non_ground_ptc_ids = target_ptc_ids[mask]
    
    ground_ptc_ids = target_ptc_ids[~mask]
    
    # Return the results
    return non_ground_ptc_ids, ground_ptc_ids 
# In[Noise removal based local analysis]
def noise_removal_grid_radius(ptc, inlier_grids, cylinder_grid_info, max_radius_deviation = 0.01, angle_interval = 2.0*np.pi/180):

    '''This function is to remove outlier points based on a local analysis
    ptc = copy.deepcopy(PTC[ptc_coord])
    inlier_grids = Shell_Grids.copy()
    cylinder_grid_info = Shell_Grid_Info.copy()
    max_radius_deviation = SHELL_FINE_OUTLIER_REMOVAL
    angle_interval = SHELL_ANGLE_INTERVAL
    '''
    from Utilities import cart2pol
    from segmentation import DBscan_clustering

    # Check input variables and initialisation

    if isinstance(ptc, pd.DataFrame):

        if any(item in ['x','y','z'] for item in ptc.columns):

            ptc_xyz = ptc[['x','y','z']].values
#            ptc_ids = ptc.index

        else:

            raise ValueError('The input data does not contain any x, y z coordinates')

    elif isinstance(ptc, np.ndarray):

        if ptc.shape[1] >= 3:
            ptc_xyz = ptc[:,:3]
#            ptc_ids = np.arange(ptc_xyz.shape[0])
        else:
            raise ValueError('The input data does not have sufficient x, y z coordinates')
    else:
        raise ValueError('The input data does not support')

    inlier_grids = inlier_grids.copy()
    cylinder_grid_info = cylinder_grid_info.copy()

    # Create new outlier_grid -. contain outlier points for each grid
    grid_outlier_ptc_ids = []
    
    #% Generate a polar grids
    angle_num_cells = int(np.ceil(2.0*np.pi/angle_interval))
    polar_grid = np.linspace(0, 2.0*np.pi, num = angle_num_cells+1, endpoint = True)

    #%% Filtering inlier points
    for i in range(len(inlier_grids['ids'])):

        #% If grid is empty -> ignore
        if not (inlier_grids['prop'][i]):
            continue
        #% Extract points in the grids
        cylinder_grid_ptc_ids = np.array(inlier_grids['ptc_ids'][i])
        
        if (cylinder_grid_ptc_ids.shape[0] > 100):

            cylinder_grid_ptc = ptc_xyz[cylinder_grid_ptc_ids]

            # Extract the circle center of the grids and convert points to a local polar coordinate system
            cylinder_grid_local_ptc_xy = cylinder_grid_ptc[:,:2] - cylinder_grid_info[i,1:3]
            cylinder_grid_ptc_rho, cylinder_grid_ptc_theta = cart2pol(cylinder_grid_local_ptc_xy[:,0], cylinder_grid_local_ptc_xy[:,1])
    
    #        # Build vectors storing global and local polar corrdinates
    #        cylinder_local_ptc_polar = pd.DataFrame(np.hstack((cylinder_grid_ptc_theta.reshape(-1,1), cylinder_grid_ptc_rho.reshape(-1,1))),
    #                                                columns = ['theta', 'rho'], index = cylinder_grid_ptc_ids)
    #        cylinder_grid_ptc = pd.concat([cylinder_grid_ptc, cylinder_local_ptc_polar], axis=1, ignore_index = False)
    #
            # Find inlier points in each grid
            grid_inlier_ptc_ids = dict()
            grid_rho = []
            for j in range(len(polar_grid)-1):
    
                #% Retrieve the radius of each point in the cell
                mask = (polar_grid[j] <= cylinder_grid_ptc_theta) & (cylinder_grid_ptc_theta < polar_grid[j+1])
                local_ptc_ids = np.where(mask == True)[0]
    
                if len(local_ptc_ids) >= 10:
    
                    cylinder_polar_grid_ptc_rho = cylinder_grid_ptc_rho[local_ptc_ids]
                    cylinder_polar_grid_ptc_rho_ids = cylinder_grid_ptc_ids[local_ptc_ids]
    
                    # Difference of the radii
                    diff_cylinder_grid_ptc_rho = np.max(cylinder_polar_grid_ptc_rho) - np.min(cylinder_polar_grid_ptc_rho)
    
                    # Large difference -> noise data
                    if diff_cylinder_grid_ptc_rho >= max_radius_deviation: #Can set as the noise level
    
                        #% Using a histogram to detect the inlier points
                        nbins = np.max((7, int(np.ceil(diff_cylinder_grid_ptc_rho/max_radius_deviation))))
                        hist_rho = np.histogram(cylinder_polar_grid_ptc_rho, bins = nbins, density = True)
    
                        # Find the peak and points within the peak
                        max_bins_id = np.argmax(hist_rho[0])
                        bound_edges = (hist_rho[1][max_bins_id] - max_radius_deviation/2, hist_rho[1][max_bins_id+1] + max_radius_deviation/2)
                        mask = (bound_edges[0] <= cylinder_polar_grid_ptc_rho)&(cylinder_polar_grid_ptc_rho <= bound_edges[1])
    
                        inlier_ptc_ids = cylinder_polar_grid_ptc_rho_ids[mask]
                        outlier_ptc_ids = cylinder_polar_grid_ptc_rho_ids[~mask]
    
                        # Assign the inlier point ids for the grid
                        grid_inlier_ptc_ids[j] = inlier_ptc_ids
                        
                        # Assign the outlier point ids for the grid
                        grid_outlier_ptc_ids = np.append(grid_outlier_ptc_ids, outlier_ptc_ids)
    
                        # Estimate the radius of the grid
                        grid_rho = np.append(grid_rho, [j, np.mean(cylinder_polar_grid_ptc_rho[mask])])
                        
                        del nbins, hist_rho, max_bins_id, bound_edges,inlier_ptc_ids    
                    else: 
                       # Fro small difference: inlier points and radius of the grid
                       grid_inlier_ptc_ids[j] = cylinder_polar_grid_ptc_rho_ids
                       grid_rho = np.append(grid_rho, [j, np.mean(cylinder_polar_grid_ptc_rho)])
                    
                    # Del temporate variables
                    del cylinder_polar_grid_ptc_rho, cylinder_polar_grid_ptc_rho_ids, diff_cylinder_grid_ptc_rho
    
                del mask, local_ptc_ids 
                
    
            # Convert radius of the grid to an array
            grid_rho = grid_rho.reshape(-1,2)
    
            # Clustering the grid based on its radius by using DB scan
            grid_cluster, __ = DBscan_clustering(grid_rho[:,1], eps = max_radius_deviation, min_pts = 20)
    
            # Find the bigest clusters
            if grid_cluster.shape[0] >= 20:
    
                grid_ids = grid_rho[grid_cluster['label'].index, 0]
            else:
                grid_ids = []
    
            # Accumulate all inlier points in the grids
            cylinder_grid_inlier_ptc_ids = list(chain.from_iterable(grid_inlier_ptc_ids[j] for j in grid_ids))
    
            # Update new inlier points
            inlier_grids['ptc_ids'][int(i)] = cylinder_grid_inlier_ptc_ids
        else:
            inlier_grids['ptc_ids'][i] = []

    # Return the result
    return inlier_grids, grid_outlier_ptc_ids


# In[Stair removal]
def removal_rail(ptc, cylinder_grids, cylinder_grid_info, interval_thickness = 0.2,
                 interval_angle = 2*np.pi/180, rail_gap = 0.025):
    '''This function is to remove the point cloud of the rail attached on a top
    of the shell wall.
    ptc = PTC.copy()
    cylinder_grids = Shell_Grids.copy()
    cylinder_grid_info = Shell_Grid_Info.copy()
    interval_thickness = INTERVAL_THICKNESS
    interval_angle = SHELL_ANGLE_INTERVAL
    rail_gap = SHELL_RAIL_GAP
    '''

    from .utilities import cart2pol, wPCA, dist_3Dpoint_surfaces
    from .data_structures import grid_extraction, cal_cell_height

    # Copy input variables
    ptc = ptc.copy()
    cylinder_grids = cylinder_grids.copy()
    cylinder_grid_info = cylinder_grid_info.copy()

    # Searching the grid containing the point cloid of a top of the Shell
    candidate_grid_ids = cylinder_grids['ids'][cylinder_grids['ids'] >= 0.50*np.max(cylinder_grids['ids'])]

    # print(f'candidate {candidate_grid_ids}')
    lower_grid_num_cell = 0
    for counter, val in enumerate(candidate_grid_ids):
        # if ((val + 1) in candidate_grid_ids):
        mask = np.in1d(cylinder_grids['ids'], val)
        grid_order = np.asscalar(np.where(mask == True)[0])
        if cylinder_grids['ptc_ids'][grid_order] is None:
            cylinder_top_grid_id = candidate_grid_ids[counter - 1]
            break
        cylinder_grid_ptc_ids = cylinder_grids['ptc_ids'][grid_order]
        cylinder_grid_ptc = ptc[cylinder_grid_ptc_ids, :3]
        cylinder_grid_center = cylinder_grid_info[1:3] #cylinder_grid_info[grid_order,1:3]

        # Convert coordinates in the global to local polar
        __, cylinder_ptc_theta = cart2pol(cylinder_grid_ptc[:,0] - cylinder_grid_center[0], cylinder_grid_ptc[:,1] - cylinder_grid_center[1])
        polar_grid_cell = grid1d(cylinder_ptc_theta, s_coord = 0.0, e_coord = 2.0*np.pi, interval_size = interval_angle, direction = 'theta')
        # polar_grid_num_cell = len(polar_grid_cell['ids'])
        polar_grid_num_cell = sum(polar_grid_cell['prop'])
        # Calculate the cell height
        polar_grid_cell = grid_extraction(polar_grid_cell, 1)
        cell_height = cal_cell_height(polar_grid_cell, ptc)
        mask = cell_height[:,0] >= interval_thickness/2.0
        # Determine the cylinder grid id which contain
        flag = cell_height[mask]

        # print(val, flag.shape[0], polar_grid_num_cell)
        # if (flag.shape[0] <= 0.75*polar_grid_num_cell):
        # if (flag.shape[0] <= 0.2*polar_grid_num_cell):
        if polar_grid_num_cell < lower_grid_num_cell * 0.7:
            cylinder_top_grid_id = candidate_grid_ids[counter - 1]
            print(f'top grid {cylinder_top_grid_id}')
            break
        # else:
        #     cylinder_top_grid_id = candidate_grid_ids[counter]
        #     break
        all_top_pid = cylinder_grid_ptc_ids

        # the number of cell in the grid below
        lower_grid_num_cell = polar_grid_num_cell
    # print(f'top grid {cylinder_top_grid_id}')
    # return None, all_top_pid
    # Remove outlier points
    if "cylinder_top_grid_id" not in locals():
        cylinder_top_grid_id = candidate_grid_ids[-1]       # Rail is not available

    print(f'top grid {cylinder_top_grid_id}')

    #%% Extract points in a top grids to remove outlier points
#    mask = np.in1d(cylinder_grids['ids'], cylinder_top_grid_id)
    mask = np.in1d(cylinder_grids['ids'], [cylinder_top_grid_id, cylinder_top_grid_id-1])
    grid_order = np.where(mask == True)[0]
    cylinder_grid_ptc_global_ids = list(chain.from_iterable(cylinder_grids['ptc_ids'][ids] for ids in grid_order))
    cylinder_grid_ptc_global_ids = np.asarray(cylinder_grid_ptc_global_ids, dtype = np.int32)
    cylinder_grid_ptc = ptc[cylinder_grid_ptc_global_ids, :3]
    
    if grid_order.shape[0] == 1:
        cylinder_grid_center = cylinder_grid_info[1:3] #cylinder_grid_info[grid_order, 1:3]
    else:
        grid_xc, grid_yc, __, __, __, __ = leastsq_fit_circle(cylinder_grid_ptc[:,0], cylinder_grid_ptc[:,1])
        cylinder_grid_center = np.array([grid_xc, grid_yc])

    # Convert coordinates in the global to local polar
#    cylinder_ptc_polar = global_cart_2_local_polar(cylinder_grid_ptc, cylinder_grid_center)
    __, cylinder_ptc_theta = cart2pol(cylinder_grid_ptc[:,0] - cylinder_grid_center[0], cylinder_grid_ptc[:,1] - cylinder_grid_center[1])
            
    polar_grid_cell = grid1d(cylinder_ptc_theta, s_coord = 0.0, e_coord = 2.0*np.pi, interval_size = interval_angle, direction = 'theta')

    # Remove all empty cells
    polar_grid_cell = grid_extraction(polar_grid_cell, 1)

    # Determine inlier points
    cylinder_grid_inlier_ptc_ids = []
    cylinder_grid_ptc_zmax_ids = []

    print(f"polar grid cell: {len(polar_grid_cell['ids'])}")
    top_grid = np.empty((0,4))
    for j in range(len(polar_grid_cell['ids'])):

        polar_grid_cell_ptc_local_ids = polar_grid_cell['ptc_ids'][j]
        polar_grid_cell_ptc_global_ids = cylinder_grid_ptc_global_ids[polar_grid_cell_ptc_local_ids]
        polar_grid_cell_ptc_z = ptc[polar_grid_cell_ptc_global_ids, 2]

        _p = ptc[polar_grid_cell_ptc_global_ids, :]
        top_grid = np.r_[top_grid, np.c_[_p, j * np.ones(_p.shape[0])]]

        # Clustering and extracting the ptc_ids in the largest cluster
        if polar_grid_cell_ptc_z.shape[0] >= 10:
            try:
                polar_grid_cell_cluster, __ = DBscan_clustering(polar_grid_cell_ptc_z.reshape(-1,1), eps = rail_gap, min_pts = 10)

                polar_grid_cell_cluster_id = polar_grid_cell_cluster['label'].value_counts().idxmax()
                polar_grid_cluster_local_ptc_ids = polar_grid_cell_cluster.loc[polar_grid_cell_cluster['label'] == polar_grid_cell_cluster_id].index
                polar_grid_cluster_ptc_ids = polar_grid_cell_ptc_global_ids[polar_grid_cluster_local_ptc_ids]

                # Cumulate all inlier points
    #            cylinder_grid_inlier_ptc_ids = np.append(cylinder_grid_inlier_ptc_ids, polar_grid_cluster_ptc_ids)
                # Extract the number of the points having z min = 5 points
                mask = np.argsort(polar_grid_cell_ptc_z[polar_grid_cluster_local_ptc_ids])[::-1]
                cylinder_grid_ptc_zmax_ids = np.append(cylinder_grid_ptc_zmax_ids, polar_grid_cluster_ptc_ids[mask[:5]])
            except Exception as exc:
                pass
                # print(exc)
                # print(j)
                # print(polar_grid_cell_cluster)
                # print(polar_grid_cell_ptc_z)
                # return None, None

    # Fiting the surface through the
    cylinder_grid_ptc_zmax_ids = np.asarray(cylinder_grid_ptc_zmax_ids, dtype = np.int32)
    shell_top_ptc = ptc[cylinder_grid_ptc_zmax_ids, :3]

    all_high_ptc_ids = cylinder_grid_ptc_zmax_ids
    
    # Filter outlier points from top points: Points have a distance to a fitting surfave larger than a rail_gap / 2
    cylinder_grid_outlier_ptc_zmax_ids = []
    flag = True
    while flag:
        # Compute normal and center
        shell_top_normal, shell_top_center = wPCA(shell_top_ptc, tol = 0.5*np.pi/180)
        
        # Adjust the direction
        if np.sum(shell_top_normal*np.array([0,0,1])) < 0: shell_top_normal = -shell_top_normal
        
        # Compute distance from the points to the fitting surface and filter outlier points
        shell_top_ptc_dist = dist_3Dpoint_surfaces(shell_top_ptc, shell_top_center, shell_top_normal)
        mask = shell_top_ptc_dist <= rail_gap/2
        
        outlier_ptc_zmax_ids = cylinder_grid_ptc_zmax_ids[~mask]
        
        if outlier_ptc_zmax_ids.shape[0] > 0:
            cylinder_grid_ptc_zmax_ids = cylinder_grid_ptc_zmax_ids[mask]
            shell_top_ptc = ptc[cylinder_grid_ptc_zmax_ids, :3]
            cylinder_grid_outlier_ptc_zmax_ids = np.append(cylinder_grid_outlier_ptc_zmax_ids, outlier_ptc_zmax_ids)
        else:
            flag = False

    top_surface = {
        'normal': shell_top_normal,
        'center': shell_top_center,
        'ptc': shell_top_ptc,
        'grid_ptc_ids': cylinder_grid_ptc_global_ids
    }
#     # Calculate distances of point clouds to the fitting surface
# #    cylinder_grid_inlier_ptc_ids = np.asarray(cylinder_grid_inlier_ptc_ids, dtype= np.int32)
#     mask = np.in1d(cylinder_grids['ids'], cylinder_top_grid_id)
#     grid_order = np.asscalar(np.where(mask == True)[0])
#     cylinder_top_grid_ptc_global_ids = cylinder_grids['ptc_ids'][grid_order] 
    
#     cylinder_top_grid_ptc_global_ids = np.asarray(cylinder_top_grid_ptc_global_ids, dtype = np.int32)
#     cylinder_top_grid_ptc = ptc[cylinder_top_grid_ptc_global_ids, :3]
    
#     shell_top_ptc_dist = dist_3Dpoint_surfaces(cylinder_top_grid_ptc, shell_top_center, shell_top_normal)

    
    
# #    shell_top_grid_ptc = ptc[cylinder_grid_inlier_ptc_ids, :3]
# #    shell_top_ptc_dist = dist_3Dpoint_surfaces(shell_top_grid_ptc, shell_top_center, shell_top_normal)

#     # Remove the point above the fitting surface
#     mask = shell_top_ptc_dist <= 0
#     shell_top_inlier_ptc_ids = cylinder_top_grid_ptc_global_ids[mask]
#     shell_top_outlier_ptc_ids = cylinder_top_grid_ptc_global_ids[~mask]
#     # Return the result
#     mask = np.in1d(cylinder_grid_ptc_zmax_ids, cylinder_top_grid_ptc_global_ids)
#     cylinder_grid_ptc_zmax_ids = cylinder_grid_ptc_zmax_ids[mask]
    
    # return cylinder_top_grid_id, shell_top_inlier_ptc_ids, shell_top_outlier_ptc_ids, cylinder_grid_ptc_zmax_ids, all_high_points
    # return top_surface, top_grid
    return top_surface, all_high_ptc_ids

# In[Remove outlier points at the shell bottom in an elevation direction]
def bottom_outlier_points(ptc, cylinder_grids, interval_angle = 2.0*np.pi/180, rail_gap = 0.025):
    """This function is to remove outlier points at the bottom of the shell
    Demo:
       ptc = ptc
       cylinder_grids = Shell_Grids.copy()
       cylinder_grid_info = Shell_Grid_Info.copy()
       interval_angle = SHELL_ANGLE_INTERVAL
       rail_gap = SHELL_RAIL_GAP
        

    """
    # Call functions
    from Utilities import cart2pol, wPCA, dist_3Dpoint_surfaces
    from data_structures import grid_extraction
    from Fitting import leastsq_fit_circle
    
    # Copy input variables
    ptc = ptc.copy()
    cylinder_grids = cylinder_grids.copy()

    # Identify the grid at the bottom
#    cylinder_grid_bottom_id = np.hstack((cylinder_grids['ids'][0], cylinder_grids['ids'][1]))
#    mask = np.in1d(cylinder_grids['ids'], cylinder_grid_bottom_id)
#    grid_order = np.asscalar(np.where(mask == True)[0])

    # Extract the points of the grid at the bottom of the shell
#    cylinder_grid_ptc_ids = cylinder_grids['ptc_ids'][grid_order]
#    cylinder_grid_ptc_ids = np.hstack((cylinder_grids['ptc_ids'][0], cylinder_grids['ptc_ids'][1]))
    cylinder_grid_ptc_ids = np.hstack((cylinder_grids['ptc_ids'][0], cylinder_grids['ptc_ids'][1]))
    grid_id = 2
    while cylinder_grid_ptc_ids.shape[0] <= 100:
        cylinder_grid_ptc_ids = np.append(cylinder_grid_ptc_ids, cylinder_grids['ptc_ids'][grid_id])
        cylinder_grid_ptc_ids = np.asarray(cylinder_grid_ptc_ids, dtype = np.int32)
        grid_id += 1
    # Extract coordinate of the bottom points
#    print(cylinder_grid_ptc_ids.shape)
    cylinder_grid_ptc = ptc[cylinder_grid_ptc_ids, :3]
    xc, yc, __, __, __, __ = leastsq_fit_circle(cylinder_grid_ptc[:, 0], cylinder_grid_ptc[:, 1])
#    cylinder_grid_center = cylinder_grid_info[grid_order,1:3]

    # Convert coordinates in the global to local polar: Note ptc_ids here is local
    __, cylinder_ptc_theta = cart2pol(cylinder_grid_ptc[:, 0] - xc, cylinder_grid_ptc[:, 1] - yc)
    polar_grid_cell = grid1d(cylinder_ptc_theta, s_coord = 0, e_coord = 2.0*np.pi, interval_size = interval_angle, direction = 'theta')

    # Remove all empty cells
    polar_grid_cell = grid_extraction(polar_grid_cell, 1)

    # Determine inpier points
#    cylinder_grid_inlier_ptc_ids = []
    cylinder_grid_ptc_zmin_ids = []
    
    for j in range(len(polar_grid_cell['ids'])):

        polar_grid_cell_ptc_local_ids = polar_grid_cell['ptc_ids'][j]
        polar_grid_cell_ptc_global_ids = cylinder_grid_ptc_ids[polar_grid_cell_ptc_local_ids]
        polar_grid_cell_ptc_z = ptc[polar_grid_cell_ptc_global_ids, 2]

        # Clustering and extracting the ptc_ids in the largest cluster
        if polar_grid_cell_ptc_z.shape[0] >= 50:
            
            polar_grid_cell_cluster, __ = DBscan_clustering(polar_grid_cell_ptc_z.reshape(-1,1), eps = rail_gap, min_pts = 50)
            
            if polar_grid_cell_cluster.shape[0] >= 5:
                polar_grid_cell_cluster_id = polar_grid_cell_cluster['label'].value_counts().idxmax()
                polar_grid_cluster_local_ptc_ids = polar_grid_cell_cluster.loc[polar_grid_cell_cluster['label'] == polar_grid_cell_cluster_id].index
    #            polar_grid_cluster_ptc_ids = polar_grid_cell_ptc_z.iloc[polar_grid_cluster_local_ptc_ids].index
                polar_grid_cluster_ptc_ids = polar_grid_cell_ptc_global_ids[polar_grid_cluster_local_ptc_ids]
    
                # Accumulate all inlier points
    #            cylinder_grid_inlier_ptc_ids = np.append(cylinder_grid_inlier_ptc_ids, polar_grid_cluster_ptc_ids)
                # Extract the number of the points having z min = 5 points
                mask = np.argsort(polar_grid_cell_ptc_z[polar_grid_cluster_local_ptc_ids])
                cylinder_grid_ptc_zmin_ids = np.append(cylinder_grid_ptc_zmin_ids, polar_grid_cluster_ptc_ids[mask[:5]])
    
    # Fiting the surface through the bottom points
    cylinder_grid_ptc_zmin_ids = np.asarray(cylinder_grid_ptc_zmin_ids, dtype = np.int32)
    shell_bottom_ptc = ptc[cylinder_grid_ptc_zmin_ids, :3]
    shell_bottom_normal, shell_bottom_center = wPCA(shell_bottom_ptc, tol = 0.5*np.pi/180)

    if np.sum(shell_bottom_normal*np.array([0,0,1])) < 0:
        shell_bottom_normal = -shell_bottom_normal

    # Calculate distances of point clouds to the fitting surface
    outlier_bottom_ptc_ids = []
    for i in range(grid_id):
        # Retrieve the points within the grid
        bottom_grid_ptc_ids = cylinder_grids['ptc_ids'][i]
        bottom_grid_ptc_ids = np.asarray(bottom_grid_ptc_ids, dtype = np.int32)
        if bottom_grid_ptc_ids.shape[0] > 1:
            bottom_grid_ptc = ptc[bottom_grid_ptc_ids, :3]
            
            # Calculate the distance from the poins to to bottom surface
            shell_bottom_ptc_dist = dist_3Dpoint_surfaces(bottom_grid_ptc, shell_bottom_center, shell_bottom_normal)
    
            # Remove the point above the fitting surface
            mask = shell_bottom_ptc_dist >= 0
            bottom_grid_inlier_ptc_ids = bottom_grid_ptc_ids[mask]
            bottom_grid_outlier_ptc_ids = bottom_grid_ptc_ids[~mask]
            
            # Update Shell grids
            cylinder_grids['ptc_ids'][i] = bottom_grid_inlier_ptc_ids
            outlier_bottom_ptc_ids = np.append(outlier_bottom_ptc_ids, bottom_grid_outlier_ptc_ids)
        
    # Return the result
    return cylinder_grids, cylinder_grid_ptc_zmin_ids, np.asarray(outlier_bottom_ptc_ids, dtype = np.int32)

# In[Refine the bottom grid]
def refine_bottom_grids(ptc, cylinder_grids, shell_grid_info, outlier_ptc_ids, interval_angle = 2*np.pi/180, option = "Bottom"):
    
    """This function is to remove outlier points of the bottom and th top grid based on its adjacent grid
    
    Demo:
        ptc = ptc.copy()
        cylinder_grids =  copy.deepcopy(Shell_Grids) 
        shell_grid_info = copy.deepcopy(Shell_Grid_Info)
        outlier_ptc_ids = copy.deepcopy(Outlier_Ptc_Ids)
        interval_angle = actual_angle_interval
        option = "Top"
    """
    
    # Copy input data
    ptc = ptc.copy()
    cylinder_grids =  cylinder_grids.copy()
    shell_grid_info = shell_grid_info.copy()
    outlier_ptc_ids = outlier_ptc_ids.copy()
    
    # Grid refinement
    if option.casefold() == "Bottom".casefold():
        grid_id = shell_grid_info[1,0].astype(int)
        rem_grid_id = shell_grid_info[0,0].astype(int)
    else:
        grid_id = shell_grid_info[-2, 0].astype(np.int32)
        rem_grid_id = shell_grid_info[-1,0].astype(int)
     
    # Grid 1: Reference points for filtering 
    mask = np.in1d(cylinder_grids["ids"], grid_id)
    cur_grid_id = np.where(mask == True)[0]
    cur_grid_id = np.asscalar(cur_grid_id)
    reference_grid_ptc_ids =  cylinder_grids["ptc_ids"][cur_grid_id]
    reference_grid_ptc_ids = np.asarray(reference_grid_ptc_ids)
    
    # Grid to be filtered
    mask = np.in1d(cylinder_grids["ids"], rem_grid_id)
    rem_grid_id = np.where(mask == True)[0]
    rem_grid_id = np.asscalar(rem_grid_id)
    rem_grid_ptc_ids =  cylinder_grids["ptc_ids"][rem_grid_id]
    rem_grid_ptc_ids = np.asarray(rem_grid_ptc_ids)
    
    if reference_grid_ptc_ids.shape[0] > 500:
        reference_grid_ptc =  ptc[reference_grid_ptc_ids, :3]
        
        # Extract a center of the reference 
        xc, yc, rc = shell_grid_info[cur_grid_id, 1], shell_grid_info[cur_grid_id, 2], shell_grid_info[cur_grid_id, 3]
        
#        # Grid 0: The points within the Grid to be filtered
#        sample_grid_ptc_ids = cylinder_grids["ptc_ids"][0].astype(np.int32)
#        sample_grid_ptc = ptc[sample_grid_ptc_ids, :3]
        
        # Extract potential ptc in Outlier points (non_shell wall points) to filter
        rem_grid_ptc = ptc[rem_grid_ptc_ids, :3]
    
        # Convert points in Catersian coord. to a polar coordinate system
        reference_grid_ptc = global_cart_2_local_polar(reference_grid_ptc, np.array([xc, yc]))
#        sample_grid_ptc = global_cart_2_local_polar(sample_grid_ptc, np.array([xc, yc]))
        rem_grid_ptc = global_cart_2_local_polar(rem_grid_ptc, np.array([xc, yc]))
        
        reference_grid_ptc = reference_grid_ptc[["theta", "rho"]].values
#        sample_grid_ptc = sample_grid_ptc[["theta", "rho"]].values
        rem_grid_ptc = rem_grid_ptc[["theta", "rho"]].values
    
        # Generate a polar grids
        angle_num_cells = int(np.ceil(2.0*np.pi/interval_angle))
        polar_grid = np.linspace(0, 2.0*np.pi, num = angle_num_cells+1, endpoint = True)
        
        # Preallocation
#        sample_grid_cell_inlier_ptc_ids = []
#        sample_grid_cell_outlier_ptc_ids = []
        rem_add_ptc_ids = []
        
        # Filter inlier and outlier points 
        for j in range(len(polar_grid)-1):
            
            # Retrieve the radius of each point in the cell
            mask = (polar_grid[j] <= reference_grid_ptc[:, 0]) & (reference_grid_ptc[:, 0] < polar_grid[j + 1])
            reference_grid_cell_ptc_local_ids = np.where(mask == True)[0]
    
            # Retrieve sample_ptc in the cell
#            mask = (polar_grid[j] <= sample_grid_ptc[:, 0]) & (sample_grid_ptc[:, 0] < polar_grid[j + 1])
#            sample_grid_cell_ptc_local_ids = np.where(mask == True)[0]
               
            # Retrieve outlier_pos_ptc in Cell
            mask = (polar_grid[j] <= rem_grid_ptc[:, 0]) & (rem_grid_ptc[:, 0] < polar_grid[j + 1])
            rem_grid_cell_ptc_local_ids = np.where(mask == True)[0]
            
            # Filtering
            if reference_grid_cell_ptc_local_ids.shape[0] >= 10:
                reference_grid_cell_ptc_radius = reference_grid_ptc[reference_grid_cell_ptc_local_ids, 1]
                reference_grid_mean_radius = np.mean(reference_grid_cell_ptc_radius)
                reference_grid_std_radius = np.std(reference_grid_cell_ptc_radius)
                lower_bound, upper_bound = reference_grid_mean_radius - 3*reference_grid_std_radius, reference_grid_mean_radius + 3*reference_grid_std_radius
            else:
                lower_bound, upper_bound = rc - 1.0*0.005, rc + 1.0*0.005
            
#                # Check radius of points in the cell less than the bounds from a top grid
#                if sample_grid_cell_ptc_local_ids.shape[0] > 0:
#                    mask = (lower_bound <= sample_grid_ptc[sample_grid_cell_ptc_local_ids, 1]) & (sample_grid_ptc[sample_grid_cell_ptc_local_ids, 1] < upper_bound)
#                    
#                    sample_grid_cell_inlier_ptc_ids = np.append(sample_grid_cell_inlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids[mask]])
#                    sample_grid_cell_outlier_ptc_ids = np.append(sample_grid_cell_outlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids[~mask]])
                
            # Check radius of points in outlier_pos_ptc
            if rem_grid_cell_ptc_local_ids.shape[0] > 0:

                mask = (lower_bound <= rem_grid_ptc[rem_grid_cell_ptc_local_ids, 1]) & (rem_grid_ptc[rem_grid_cell_ptc_local_ids, 1] < upper_bound)
                rem_add_ptc_ids = np.append(rem_add_ptc_ids, rem_grid_ptc_ids[rem_grid_cell_ptc_local_ids[mask]])
            
#            else:
#                sample_grid_cell_inlier_ptc_ids = np.append(sample_grid_cell_inlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids])
        
        # Update shell grid       
        cylinder_grids["ptc_ids"][rem_grid_id] = np.setdiff1d(rem_grid_ptc_ids, rem_add_ptc_ids).astype(np.int32)  
        
        # Update outlier ptc
        outlier_ptc_ids = np.union1d(outlier_ptc_ids, rem_add_ptc_ids) 
        
    # return results
    return cylinder_grids, outlier_ptc_ids
## In[Refine the bottom grid]
#def refine_bottom_grids(ptc, cylinder_grids, shell_grid_info, outlier_ptc_ids, interval_angle = 2*np.pi/180, option = "Bottom"):
#    
#    """This function is to refine the bottom grid, which may contain a points of the floor or ground
#    
#    Demo:
#        ptc = ptc.copy()
#        cylinder_grids =  copy.deepcopy(Shell_Grids) 
#        shell_grid_info = copy.deepcopy(Shell_Grid_Info)
#        outlier_ptc_ids = copy.deepcopy(Outlier_Ptc_Ids)
#        interval_angle = actual_angle_interval
#        option = "Top"
#    """
#    
#    # Copy input data
#    ptc = ptc.copy()
#    cylinder_grids =  cylinder_grids.copy()
#    shell_grid_info = shell_grid_info.copy()
#    outlier_ptc_ids = outlier_ptc_ids.copy()
#    
#    # Grid refinement
#    if option.casefold() == "Bottom".casefold():
#        grid_ids = shell_grid_info[:2,0][::-1].astype(int)
#    else:
#        grid_ids = np.hstack((shell_grid_info[-2, 0], shell_grid_info[-1, 0])).astype(np.int32)
#     
#    for i in grid_ids:     
#        # Grid 1: Reference points for filtering 
#        mask = np.in1d(cylinder_grids["ids"], i)
#        cur_grid_id = np.where(mask == True)[0]
#        if cur_grid_id.shape[0] < 1: continue
#        
##        cur_grid_id = np.asscalar(np.where(mask == True)[0])
#        cur_grid_id = np.asscalar(cur_grid_id)
#        reference_grid_ptc_ids =  cylinder_grids["ptc_ids"][cur_grid_id]
#        reference_grid_ptc_ids = np.asarray(reference_grid_ptc_ids)
#        if reference_grid_ptc_ids.shape[0] > 50:
#            reference_grid_ptc =  ptc[reference_grid_ptc_ids, :3]
#            min_elevation, max_elevation = np.min(reference_grid_ptc[:,2]), np.max(reference_grid_ptc[:,2])
#            
#            # Extract a center of the reference 
#            xc, yc, rc = shell_grid_info[cur_grid_id, 1], shell_grid_info[cur_grid_id, 2], shell_grid_info[cur_grid_id, 3]
#            
#    #        # Grid 0: The points within the Grid to be filtered
#    #        sample_grid_ptc_ids = cylinder_grids["ptc_ids"][0].astype(np.int32)
#    #        sample_grid_ptc = ptc[sample_grid_ptc_ids, :3]
#            
#            # Extract potential ptc in Outlier points (non_shell wall points) to filter
#            if option.casefold() == "Bottom".casefold():
#                mask = (min_elevation - (max_elevation - min_elevation) <= ptc[outlier_ptc_ids, 2]) & (ptc[outlier_ptc_ids, 2] <= min_elevation) 
#            else:
#                mask = (max_elevation <= ptc[outlier_ptc_ids, 2]) & (ptc[outlier_ptc_ids, 2] <= max_elevation + (max_elevation - min_elevation)) 
#            
#            outlier_pos_ptc_ids = outlier_ptc_ids[mask]
#            outlier_pos_ptc = ptc[outlier_pos_ptc_ids, :3]
#        
#            # Convert points in Catersian coord. to a polar coordinate system
#            reference_grid_ptc = global_cart_2_local_polar(reference_grid_ptc, np.array([xc, yc]))
#    #        sample_grid_ptc = global_cart_2_local_polar(sample_grid_ptc, np.array([xc, yc]))
#            outlier_pos_ptc = global_cart_2_local_polar(outlier_pos_ptc, np.array([xc, yc]))
#            
#            reference_grid_ptc = reference_grid_ptc[["theta", "rho"]].values
#    #        sample_grid_ptc = sample_grid_ptc[["theta", "rho"]].values
#            outlier_pos_ptc = outlier_pos_ptc[["theta", "rho"]].values
#        
#            # Generate a polar grids
#            angle_num_cells = int(np.ceil(2*np.pi/interval_angle))
#            polar_grid = np.linspace(0, 2*np.pi, num = angle_num_cells+1, endpoint = True)
#            
#            # Preallocation
#    #        sample_grid_cell_inlier_ptc_ids = []
#    #        sample_grid_cell_outlier_ptc_ids = []
#            outlier_add_ptc_ids = []
#            
#            # Filter inlier and outlier points 
#            for j in range(len(polar_grid)-1):
#                
#                # Retrieve the radius of each point in the cell
#                mask = (polar_grid[j] <= reference_grid_ptc[:, 0]) & (reference_grid_ptc[:, 0] < polar_grid[j + 1])
#                reference_grid_cell_ptc_local_ids = np.where(mask == True)[0]
#        
#                # Retrieve sample_ptc in the cell
#    #            mask = (polar_grid[j] <= sample_grid_ptc[:, 0]) & (sample_grid_ptc[:, 0] < polar_grid[j + 1])
#    #            sample_grid_cell_ptc_local_ids = np.where(mask == True)[0]
#                   
#                # Retrieve outlier_pos_ptc in Cell
#                mask = (polar_grid[j] <= outlier_pos_ptc[:, 0]) & (outlier_pos_ptc[:, 0] < polar_grid[j + 1])
#                outlier_grid_cell_ptc_local_ids = np.where(mask == True)[0]
#                
#                # Filtering
#                if reference_grid_cell_ptc_local_ids.shape[0] >= 10:
#                    reference_grid_cell_ptc_radius = reference_grid_ptc[reference_grid_cell_ptc_local_ids, 1]
#                    reference_grid_mean_radius = np.mean(reference_grid_cell_ptc_radius)
#                    reference_grid_std_radius = np.std(reference_grid_cell_ptc_radius)
#                    lower_bound, upper_bound = reference_grid_mean_radius - 3*reference_grid_std_radius, reference_grid_mean_radius + 3*reference_grid_std_radius
#                else:
#                    lower_bound, upper_bound = rc - 1*0.005, rc + 1*0.005
#                
#    #                # Check radius of points in the cell less than the bounds from a top grid
#    #                if sample_grid_cell_ptc_local_ids.shape[0] > 0:
#    #                    mask = (lower_bound <= sample_grid_ptc[sample_grid_cell_ptc_local_ids, 1]) & (sample_grid_ptc[sample_grid_cell_ptc_local_ids, 1] < upper_bound)
#    #                    
#    #                    sample_grid_cell_inlier_ptc_ids = np.append(sample_grid_cell_inlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids[mask]])
#    #                    sample_grid_cell_outlier_ptc_ids = np.append(sample_grid_cell_outlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids[~mask]])
#                    
#                # Check radius of points in outlier_pos_ptc
#                if outlier_grid_cell_ptc_local_ids.shape[0] > 0:
#    
#                    mask = (lower_bound <= outlier_pos_ptc[outlier_grid_cell_ptc_local_ids, 1]) & (outlier_pos_ptc[outlier_grid_cell_ptc_local_ids, 1] < upper_bound)
#                    outlier_add_ptc_ids = np.append(outlier_add_ptc_ids, outlier_pos_ptc_ids[outlier_grid_cell_ptc_local_ids[mask]])
#                
#    #            else:
#    #                sample_grid_cell_inlier_ptc_ids = np.append(sample_grid_cell_inlier_ptc_ids, sample_grid_ptc_ids[sample_grid_cell_ptc_local_ids])
#            
#            # Update shell grid       
#            cylinder_grids["ptc_ids"][cur_grid_id] = np.hstack((reference_grid_ptc_ids, outlier_add_ptc_ids)).astype(np.int32) 
#                    
#            # Update outlier ptc
#            outlier_ptc_ids = np.setdiff1d(outlier_ptc_ids, outlier_add_ptc_ids) 
#            
#    #        outlier_ptc_ids = np.hstack((outlier_ptc_ids, sample_grid_cell_outlier_ptc_ids.astype(np.int32)))
#        else:
#            cylinder_grids["ptc_ids"][cur_grid_id] = []
#    # return results
#    return cylinder_grids, outlier_ptc_ids
# In[Filter manway]
#def manway_filter(ptc, shell_grids, cylinder_grid_info,  grid_ids = [0, 10], cell_size = 0.05):
#    
#    '''This function is to remove the ourlier points within the full cells
#    1. connected to empty cell, known as the full cells on the boundary. This was done by comparing the point radius to ones from neighbour full cells
#    2. Isolated cells
#    
#    Input:
#        ptc                 : [Nx3] x, y, z coordinates
#        shell_grids         : dict - all grids in an elevation
#        cylinder_grid_info  : [Mx4] - Center and radius of the fitting circle of each grid
#        grid_ids            : [kx1] - List of the grids to be checked
#        cell_size           : scalar - the cell size
#    
#    Output:
#        shell_grids         : dict - all grids in an elevation, where the outlier points to be removed
#        new_outlier_ptc_ids : [Mx1] - New outlier points
#        
#    shell_grids = Shell_Grids.copy()
#    cylinder_grid_info = Shell_Grid_Info.copy()
#
#    '''
#    from data_structures import grid2D
#
#    # Preallocation   
#    new_outlier_ptc_ids = []
#    
#    for i in grid_ids:
#        
#        # [1. Extract points and center of each grid]
#        shell_grid_ptc_ids = np.asarray(shell_grids["ptc_ids"][i])
#        shell_grid_ptc = ptc[shell_grid_ptc_ids, :3]
#        shel_grid_center = cylinder_grid_info[i, 1:]
#
#        # In[2. Convert xyz coordinates to polar coordinate and create a polar grids]
#        shell_grid_polar_ptc = global_cart_2_local_polar(shell_grid_ptc, shel_grid_center[:2])
#        shell_grid_polar_ptc_radius = shell_grid_polar_ptc["rho"].values
#    
#        # Compute paramaters for the grid
#        interval_angle = cell_size/shel_grid_center[-1]
#        major_dir = {'z': cell_size}
#        minor_dir = {'theta':interval_angle}
#    
#        # Generate the polar grid
#        polar_grids, __ = grid2D(shell_grid_polar_ptc, major_dir = major_dir, minor_dir = minor_dir)
#        
#        # In[3. Check if the cell connected to the empty -> filter based on points radius from neighbour full cells]
#        # Convert all grid to array
#        all_cell_order = np.arange(len(polar_grids["ids"]))
#        all_cell_ids = polar_grids["ids"]
#        all_cell_ptc_ids = polar_grids["ptc_ids"]
#        
#        # Extract full cells in array
#        full_cell_order = np.where(polar_grids["prop"] == True)[0]
#        full_cell_ids = polar_grids["ids"][full_cell_order]
#        
#        # Filtering 
#        grid_outlier_ptc_local_ids = []
#        for cell_ids, cell_order in zip(full_cell_ids, full_cell_order):
#            
#            # Search neighbour cell order
#            mask = np.all((cell_ids - 1 <= all_cell_ids) & (all_cell_ids <= cell_ids + 1), axis = 1)
#            neighbour_cell_order = all_cell_order[mask]
#            
#            # Extract full and empty neighbour
#            neighbour_cell_prop = polar_grids["prop"][neighbour_cell_order]
#            mask = np.where(neighbour_cell_prop == True)[0]
#            neighbour_full_cell = neighbour_cell_order[mask]
#            neighbour_empty_cell = np.setdiff1d(neighbour_cell_order, neighbour_full_cell)
#            
#            # Remove a current full cell
#            neighbour_full_cell = np.setdiff1d(neighbour_full_cell, cell_order)
#            
#            # Check a condition for filtering
#            if neighbour_full_cell.shape[0] == 0:
#                # All point within the cell is ourlier
#                grid_outlier_ptc_local_ids = np.append(grid_outlier_ptc_local_ids, all_cell_ptc_ids[cell_order])
#                    
#            else:
#                if neighbour_empty_cell.shape[0] >= 1:
#                    
#                    # Current cell
#                    current_full_cell_ptc_ids = all_cell_ptc_ids[cell_order]
#                    current_full_cell_ptc_radius = shell_grid_polar_ptc_radius[current_full_cell_ptc_ids]
#                    
#                    # Search neighbour of the checked cell
#                    neighbour_full_cell_ptc_ids = list(chain.from_iterable(all_cell_ptc_ids[ids] for ids in neighbour_full_cell)) 
#                    neighbour_full_cell_ptc_radius = shell_grid_polar_ptc_radius[neighbour_full_cell_ptc_ids]
#                    
#                    # Filter ourlier
#                    neighbour_full_cell_ptc_radius_mean = np.mean(neighbour_full_cell_ptc_radius)
#                    neighbour_full_cell_ptc_radius_std = np.std(neighbour_full_cell_ptc_radius)
#                    error_lower_bound = neighbour_full_cell_ptc_radius_mean - 3*neighbour_full_cell_ptc_radius_std
#                    error_upper_bound = neighbour_full_cell_ptc_radius_mean + 3*neighbour_full_cell_ptc_radius_std
#                    
#                    mask = ((error_lower_bound <= current_full_cell_ptc_radius)&
#                            (current_full_cell_ptc_radius <= error_upper_bound))
#                    
#                    # Update outlier point in local indices
#                    grid_outlier_ptc_local_ids = np.append(grid_outlier_ptc_local_ids, current_full_cell_ptc_ids[~mask] )
#            
#        # Update outlier points in a global indices
#        grid_outlier_ptc_global_ids = shell_grid_ptc_ids[np.asarray(grid_outlier_ptc_local_ids, dtype = np.int32)]
#        
#        # Update points of the grid
#        shell_grids["ptc_ids"][i] = np.setdiff1d(shell_grid_ptc_ids, grid_outlier_ptc_global_ids)
#        
#        # Update new outlier ptc
#        new_outlier_ptc_ids = np.append(new_outlier_ptc_ids, grid_outlier_ptc_global_ids)
#   
#    #Return the result
#    return shell_grids, new_outlier_ptc_ids

# In[]
def manway_filter(ptc, shell_grids,  manway_radius, location = 2.5,cell_size = 0.05, dist_threshold = 0.01):
    
    '''This function is to remove the ourlier points within the full cells
    1. connected to empty cell, known as the full cells on the boundary. This was done by comparing the point radius to ones from neighbour full cells
    2. Isolated cells
    
    Input:
        ptc                 : [Nx3] x, y, z coordinates
        shell_grids         : dict - all grids in an elevation
        cylinder_grid_info  : [Mx4] - Center and radius of the fitting circle of each grid
        grid_ids            : [kx1] - List of the grids to be checked
        cell_size           : scalar - the cell size
    
    Output:
        
        shell_grids         : dict - all grids in an elevation, where the outlier points to be removed
        new_outlier_ptc_ids : [Mx1] - New outlier points
     
    Demo:
        ptc
        shell_grids = Shell_Grids.copy()
        manway_radius = MANWAY_RADIUS
        location = MANYWAY_LOCATION
        cell_size = CELL_SIZE
        dist_threshold = DIST_THRESHOLD

    '''
    # Load functions
    from data_structures import grid2D, grid_extraction
    from segmentation import segmentation2d
    from Fitting import leastsq_fit_circle
    from Utilities import wPCA, dist_3Dpoint_surfaces, vector_norm, pol2cart, cart2pol
    
    # Calcualate manway area
    manway_num_cells = np.floor([((radius/np.sqrt(2))/cell_size)**2 for radius in manway_radius]).astype(int) # The largest square inside the circle rc/sqrt(2)/cell_size     
    #% Extract the points
    shell_wall_ptc_ids = np.array(list(chain.from_iterable(shell_grids['ptc_ids'])), dtype = np.int32)
    shell_wall_ptc = ptc[shell_wall_ptc_ids]
    
    mask = shell_wall_ptc[:, 2] <= np.min(shell_wall_ptc[:, 2]) + location
    shell_bottom_wall_ptc_ids = shell_wall_ptc_ids[mask]
    shell_bottom_wall_ptc = shell_wall_ptc[mask]
    
    xc, yc, rc, __, __, __ = leastsq_fit_circle(shell_bottom_wall_ptc[:, 0], shell_bottom_wall_ptc[:, 1])
    shell_bottom_wall_ptc_rho, shell_bottom_wall_ptc_theta = cart2pol(shell_bottom_wall_ptc[:,0] - xc, shell_bottom_wall_ptc[:,1] - yc)    
    
    # Compute paramaters for the grid
    interval_angle = cell_size/rc
    major_dir = {'z': cell_size}
    minor_dir = {'theta':interval_angle}

    # Generate the polar grid
    polar_grids, __ = grid2D(np.hstack((shell_bottom_wall_ptc, shell_bottom_wall_ptc_rho.reshape(-1,1), shell_bottom_wall_ptc_theta.reshape(-1,1))), major_dir = major_dir, minor_dir = minor_dir)
        
    # Extract full cells
    full_cell = grid_extraction(polar_grids, 1)
    full_cell_grid_ids = full_cell["ids"].astype(np.int32)
    full_cell_ids = np.arange(0, full_cell_grid_ids.shape[0])
    
    # Segment empty cells
    empty_cell_regions = segmentation2d(polar_grids, cell_type = 0)
    empty_cell_region_ids = np.unique(empty_cell_regions[:, 2])
    
    #% In[5. Calculate the centroid of the empty cells]
    empty_cell = grid_extraction(polar_grids, 0)
    empty_cell_centroid = np.full((len(empty_cell['ids']), 4), np.nan)
    centroid_z = (empty_cell['bounds'][:,0] + empty_cell['bounds'][:,2])/2.0    
    empty_cell_centroid[:,3] = (empty_cell['bounds'][:,1] + empty_cell['bounds'][:,3])/2.0
    
    centroid_x, centroid_y = pol2cart(rc, empty_cell_centroid[:,3])
    empty_cell_centroid[:,0] = centroid_x + xc               
    empty_cell_centroid[:,1] = centroid_y + yc
    empty_cell_centroid[:,2] = centroid_z
    
    del centroid_x, centroid_y, centroid_z
    #% Extract outlier points in the full cells connected to empty cell regions
    new_outlier_ptc_local_ids = []
    for j in empty_cell_region_ids:

        mask = empty_cell_regions[:, 2] == j
        region_empty_cell_order = np.where(mask == True)[0]
        region_empty_cell_ids = empty_cell_regions[mask, :2].astype(np.int32)
        
        if (manway_num_cells[0] <= region_empty_cell_order.shape[0])&(region_empty_cell_order[0] <= manway_num_cells[1]):
            current_region_cell_centroid = empty_cell_centroid[region_empty_cell_order]
            
            bound_width = (np.max(current_region_cell_centroid[:, 3]) - np.min(current_region_cell_centroid[:, 3]))*rc + cell_size
            bound_height = np.max(current_region_cell_centroid[:, 2]) - np.min(current_region_cell_centroid[:, 2]) + cell_size
            
            if np.all((manway_radius[0] <= np.array([bound_width, bound_height]))&
                      (np.array([bound_width, bound_height]) <= manway_radius[1])):
            
                # Extract full cells around the empty
                region_bound_full_cell_ids = []
                for k in range(region_empty_cell_ids.shape[0]):
                    
                    # Search neighbour cell order
                    mask = np.all((region_empty_cell_ids[k] - 1 <= full_cell_grid_ids) & (full_cell_grid_ids <= region_empty_cell_ids[k] + 1), axis = 1)
                    neighbour_cell_ids = full_cell_ids[mask]
                    
                    # Cummulate full cells around the region
                    region_bound_full_cell_ids = np.append(region_bound_full_cell_ids, neighbour_cell_ids)
                         
                # Extract points within the full cells connected to the empty region
                region_bound_full_cell_ids = np.asarray(np.unique(region_bound_full_cell_ids), dtype = np.int32)

                # Filtering outlier points within the full bound cell
                for region_bound_full_cell_id in region_bound_full_cell_ids:
                    
                    # Search neighbour cell of full bound cells
                    mask = np.all((full_cell_grid_ids[region_bound_full_cell_id] - 1 <= full_cell_grid_ids) & (full_cell_grid_ids <= full_cell_grid_ids[region_bound_full_cell_id] + 1), axis = 1)
                    region_neigbour_bound_cell_ids = full_cell_ids[mask]
                     
                    # Estimate the surface through full cells around the manway by wPCA
                    region_full_neigbour_cell_ptc_ids = np.array(list(chain.from_iterable(full_cell['ptc_ids'][ids] for ids in region_neigbour_bound_cell_ids)), dtype = np.int32)
                    region_full_neigbour_cell_ptc = shell_bottom_wall_ptc[region_full_neigbour_cell_ptc_ids]
                    region_full_neigbour_cell_ptc_normal, region_full_neigbour_cell_ptc_center = wPCA(region_full_neigbour_cell_ptc, tol = 0.5*np.pi/180)
                    centroid_direction = vector_norm(region_full_neigbour_cell_ptc_center - np.array([xc, yc, region_full_neigbour_cell_ptc_center[2]]))
                    if np.sum(region_full_neigbour_cell_ptc_normal*centroid_direction) < 0:
                        region_full_neigbour_cell_ptc_normal = -region_full_neigbour_cell_ptc_normal
     
                    # Filter outlier points of the bound cell
                    region_bound_full_cell_ptc_ids = full_cell['ptc_ids'][region_bound_full_cell_id]
                    region_bound_full_cell_ptc = shell_bottom_wall_ptc[region_bound_full_cell_ptc_ids]
                    dist_ptc_surface = dist_3Dpoint_surfaces(region_bound_full_cell_ptc, region_full_neigbour_cell_ptc_center, region_full_neigbour_cell_ptc_normal)
                    mask = dist_ptc_surface > dist_threshold#np.max((dist_threshold/2, 0.005))
                    region_bound_full_cell_outlier_ptc_ids = region_bound_full_cell_ptc_ids[mask]
                    
                    # Cummulate outlier points 
                    new_outlier_ptc_local_ids = np.append(new_outlier_ptc_local_ids, region_bound_full_cell_outlier_ptc_ids)

    #% Update outlier points and in a global
    new_outlier_ptc_local_ids = np.asarray(new_outlier_ptc_local_ids, dtype = np.int32)
    new_outlier_ptc_ids = shell_bottom_wall_ptc_ids[new_outlier_ptc_local_ids]
    
    # Shell wall points
    new_shell_wall_ptc_ids = np.setdiff1d(shell_wall_ptc_ids, new_outlier_ptc_ids)
    
    #%Return the result
    return new_shell_wall_ptc_ids, new_outlier_ptc_ids

# In[Find the points on the top or bottom shell]

def shell_top_bottom_ptc_ids(ptc, shell_wall_ptc_ids, cylinder_fit_parameters, interval_angle = 2*np.pi/180, option = "top"):
    '''This function is to remove the point cloud on the top or bottom of the shell.
    Input:
    Output:
        
    Demo
        ptc = PTC.copy()
        shell_wall_ptc_ids = copy.deepcopy(Shell_Wall_Ptc_Ids)
        cylinder_fit_parameters = Shell_Wall_Best_Fit['vertical_cylinder_parameters']
        interval_angle = 1.2*np.pi/180
    '''
    # Call library
    from Utilities import wPCA, dist_3Dpoint_surfaces, cart2pol
    
    # Copy input variables
    ptc = ptc.copy()
    if isinstance(ptc, pd.DataFrame):
        ptc = ptc[["x", "y", "z"]].values
    elif isinstance(ptc, np.ndarray):
        ptc = ptc[:, :3]
        
    if interval_angle is None:
        from Constant import constant
        set_constant = constant()
        interval_angle = set_constant["shell_angle_interval"]
    if option.casefold() == "top".casefold():
        sort_dir = -1
    else:
        sort_dir = 1
    # Extract the shell wall
    shell_wall_ptc = ptc[shell_wall_ptc_ids]
    del ptc
    # convert points from Catersian to Polar
    __, shell_wall_ptc_theta = cart2pol(shell_wall_ptc[:,0] - cylinder_fit_parameters[0], shell_wall_ptc[:,1] - cylinder_fit_parameters[1])
    num_grids = np.ceil(2.*np.pi/interval_angle).astype(np.int16)
    grid_values = np.linspace(0, 2.0*np.pi, num = num_grids)
    
    shell_outmost_ptc_ids = []
    for i in range(grid_values.shape[0] - 1):
        mask = (grid_values[i] <= shell_wall_ptc_theta)&(shell_wall_ptc_theta <= grid_values[i+1])
        grid_ptc_ids = shell_wall_ptc_ids[mask]
        if grid_ptc_ids.shape[0] > 100:
            # Search top 5 points
            grid_ptc_z = shell_wall_ptc[mask, 2]
            mask = np.argsort(grid_ptc_z)[::sort_dir][:10]
            grid_ptc_z_ids = grid_ptc_ids[mask]
            shell_outmost_ptc_ids = np.append(shell_outmost_ptc_ids, grid_ptc_z_ids)
            del grid_ptc_z_ids, grid_ptc_z
            
    # Filter outlier points
    shell_outmost_ptc_ids = np.unique(np.asarray(shell_outmost_ptc_ids, dtype = np.int32))
    mask = np.in1d(shell_wall_ptc_ids, shell_outmost_ptc_ids)
    shell_outmost_ptc = shell_wall_ptc[mask]
    shell_outmost_ptc_normal, shell_outmost_ptc_center = wPCA(shell_outmost_ptc, tol = 0.5*np.pi/180)
    shell_outmost_ptc_dist = dist_3Dpoint_surfaces(shell_outmost_ptc, shell_outmost_ptc_center, shell_outmost_ptc_normal)
    mask = np.abs(shell_outmost_ptc_dist) <= 0.01
    shell_outmost_ptc_local_ids = np.where(mask == True)[0]
    shell_outmost_ptc_ids = shell_outmost_ptc_ids[shell_outmost_ptc_local_ids]
    shell_outmost_ptc = shell_outmost_ptc[shell_outmost_ptc_local_ids]
    
    
    return shell_outmost_ptc_ids, shell_outmost_ptc


# In[1D outlier filtering]
def filtering1D(data, cell_size, min_ptc = 10):
    """This fuction is to filtering ourlier points in 1 dimension, which is based on scalar value
    
    Input:
        data      : [Nx1]
        cell_size : the cell size to subdivide the scalar value
        min_ptc   : The number of points within the cell considered as the full cell
   
    Output:
        ptc_ids   : [Mx1] - Local indices of inlier points
    
    Demo:
    """
    # Check data
    if isinstance(data, pd.DataFrame):
        data = data.values
        data = data[:,0]
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:,0]
    elif isinstance(data, list):
        data = np.asarray(data)
    else:
        raise ValueError("Data type of the variable data %s does not support " %type(data))
        
        
    # Generate grid based on fitting errors
    data_grid1d = grid1d(data.reshape(-1,1), np.min(data), np.max(data), cell_size, "x", min_ptc = min_ptc)
 
    # Determine the large segment    
    data_grid1d_segment = segmentation1d(data_grid1d, cell_size, cell_size)
    
    # Extract the largest segment and ptc_ids in this segment
    num_seg = np.max(data_grid1d_segment[:,1])
    data_grid1d_segment_info = np.full((num_seg, 2), np.inf)
    data_grid1d_segment_info[:,0] = np.unique(data_grid1d_segment[:,1])
    for count, seg_id in enumerate(np.unique(data_grid1d_segment[:,1])):
        mask = data_grid1d_segment[:,1] == seg_id
        data_grid1d_segment_cell_ids = data_grid1d_segment[mask,0]
        data_grid1d_segment_ptc_ids = list(chain.from_iterable(data_grid1d['ptc_ids'][ids] for ids in data_grid1d_segment_cell_ids))       
        data_grid1d_segment_info[count,1] = len(data_grid1d_segment_ptc_ids)
    
    largest_seg_id = data_grid1d_segment_info[np.argmax(data_grid1d_segment_info[:,1]),0].astype(np.int8)
    mask = np.in1d(data_grid1d_segment[:,1], largest_seg_id)
    largest_seg_cell_ids = data_grid1d_segment[mask,0]
    largest_seg_ptc_ids = list(chain.from_iterable(data_grid1d['ptc_ids'][ids] for ids in largest_seg_cell_ids))       
    largest_seg_ptc_ids = np.asarray(largest_seg_ptc_ids, dtype = np.int32)
    
    return largest_seg_ptc_ids
    
# In[Remove points around floating roof and top and bottom]
def floating_roof_edge_points_filter(shell_wall_ptc, set_constant = None, subtask_info = None):
    
    """This function is to remove points around the edge: bottom, roof, floating roof or any hole
    
    Input:
    
    Output:
    
    Demo:
        shell_wall_ptc_ids = list(chain.from_iterable(Shell_Grids['ptc_ids']))
        shell_wall_ptc_ids = np.asarray(shell_wall_ptc_ids, dtype = np.int32)
        shell_wall_ptc = ptc[shell_wall_ptc_ids]
        
        elevation_interval = 0.5*0.5
        circular_angle_interval = 1.0*np.pi/180.0 
        sampling_step = 0.005
        
        
        
    """
    #%% Call libraries
    import os
    from Constant import constant
    from data_structures import grid2D
    from utilities import cart2pol
    from fitting import leastsq_fit_circle
    from WriteOut import write_array2txt
    
    # Check input variable
    if isinstance(shell_wall_ptc, pd.DataFrame):
        shell_wall_ptc_ids = shell_wall_ptc.index.values.astype(np.int32)
        shell_wall_ptc = shell_wall_ptc.values
        shell_wall_ptc = shell_wall_ptc[:,:3]
    elif isinstance(shell_wall_ptc, np.ndarray):
        shell_wall_ptc_ids = np.arange(shell_wall_ptc.shape[0], dtype = np.int32)
        shell_wall_ptc = shell_wall_ptc[:,:3]
    else:
        raise ValueError("Input data MUST BE DataFrame or Array")
        
    # Set up constant
    if set_constant is None:
        set_constant = constant()
    sampling_step = set_constant["sampling_step"]   
    elevation_interval = 0.5*set_constant["interval_thickness_z_dir"]
    circular_angle_interval = set_constant["shell_angle_interval"] 
    
    # Convert points to polar coordindate
    xc, yc, rc, __, __, __ = leastsq_fit_circle(shell_wall_ptc[:,0], shell_wall_ptc[:,1])
    shell_wall_ptc_rho, shell_wall_ptc_theta = cart2pol(shell_wall_ptc[:,0] - xc, shell_wall_ptc[:,1] - yc)
    
    shell_wall_ptc = np.hstack((shell_wall_ptc, shell_wall_ptc_rho.reshape(-1,1), shell_wall_ptc_theta.reshape(-1,1)))
    del shell_wall_ptc_rho, shell_wall_ptc_theta
    
    # Create 2D grid in z and theta
    grid2D_z_theta, __ = grid2D (shell_wall_ptc, major_dir = {'z': elevation_interval}, minor_dir = {'theta':circular_angle_interval}, min_ptc = 10)
    
    # Check 
    for grid_theta_id in np.unique(grid2D_z_theta['ids'][:,1]):
        
        #% Extract empty cells for each grids
#        grid_theta_id = 140
        mask = grid2D_z_theta['ids'][:,1] == grid_theta_id
        local_ids = np.where(mask == True)[0]
        grid_theta_prop = grid2D_z_theta['prop'][local_ids]
        empty_cell_local_ids = np.where(grid_theta_prop == False)[0]

        # Add virtual grids on the bottom and top
#        empty_cell_local_ids = np.insert(empty_cell_local_ids, [0, empty_cell_local_ids.shape[0]],[-1,local_ids.shape[0] + 1])

        #% Filter for each cell connected either to the bottom, empty cell, top
        for i in empty_cell_local_ids:
            
            # Determine target and ref grids
            if i < grid_theta_prop.shape[0] - 2:
                if i < 0: 
                    target_grid_local_ids, ref_grid_local_ids = np.array([0], dtype = int), np.array([1], dtype = int)
                    bandwidth = 2.0*sampling_step
                elif i > local_ids.shape[0]:
                    target_grid_local_ids, ref_grid_local_ids = np.array([local_ids.shape[0]-1], dtype = int), np.array([local_ids.shape[0]-2], dtype = int)
                    bandwidth = 2.0*sampling_step
                else:
                    target_grid_local_ids, ref_grid_local_ids = np.array([i-1, i+1], dtype = np.int32), np.array([i-2, i+2], dtype = np.int32)
                    bandwidth = 1.0*sampling_step
                # Remove target and ref cell is empty
                target_grid_prop, ref_grid_prop = grid_theta_prop[target_grid_local_ids], grid_theta_prop[ref_grid_local_ids]
                mask = (target_grid_prop == True)&(ref_grid_prop == True)
                mask = np.where(mask == True)[0]
                target_grid_local_ids, ref_grid_local_ids = target_grid_local_ids[mask], ref_grid_local_ids[mask]
                
    #            print(grid_theta_id, target_grid_local_ids, ref_grid_local_ids)
                
                if (target_grid_local_ids.shape[0] > 0)&(ref_grid_local_ids.shape[0] > 0):
                    
                    for (target_grid_local_id, ref_grid_local_id) in zip(target_grid_local_ids, ref_grid_local_ids):
                        
                        # Retrieve ptc_ids in target and ref grids
                        target_grid_id, ref_grid_id = local_ids[target_grid_local_id], local_ids[ref_grid_local_id]
                        target_grid_ptc_ids = grid2D_z_theta['ptc_ids'][target_grid_id]
                        ref_grid_ptc_ids = grid2D_z_theta['ptc_ids'][ref_grid_id]
                        target_grid_ptc = shell_wall_ptc[target_grid_ptc_ids]
                        ref_grid_ptc = shell_wall_ptc[ref_grid_ptc_ids]
                        
                        # Filtering outlier points
                        comb_grid_ptc = np.concatenate((target_grid_ptc, ref_grid_ptc), axis = 0)
                        
                        # Find the range of the peak shape, where the data have dense distribution
                        
                        range_ptc_rho = peak_shape_range(comb_grid_ptc[:,3], bandwidth = bandwidth, num_ptc = 100)
                        
                        print(range_ptc_rho)
                        # Filter inlier points
                        mask = (range_ptc_rho[0] <= target_grid_ptc[:,3])&(target_grid_ptc[:,3] <= range_ptc_rho[1])
                        temp_local_ids = np.where(mask == True)[0]
                        print("before {:d} vs after {:d}".format(target_grid_ptc_ids.shape[0], temp_local_ids.shape[0]))
                        # Update the inlier points for the reference grid
                        grid2D_z_theta['ptc_ids'][target_grid_id] = target_grid_ptc_ids[temp_local_ids]
        
    #%% gathering data
    # Filter 
    grid2D_z_theta = grid_extraction(grid2D_z_theta, 1)
    
    shell_wall_filter_ptc_ids = list(chain.from_iterable(grid2D_z_theta['ptc_ids']))
    shell_wall_filter_ptc_ids = np.asarray(shell_wall_filter_ptc_ids, dtype = np.int32)
    shell_wall_filter_ptc = shell_wall_ptc[shell_wall_filter_ptc_ids,:3]
    
    # Update data 
    file_caption = "Final Ptc of Shell Wall Filtering Floating Roof"
    file_name = file_caption + ".txt"
    file_path = os.path.join(subtask_info["subtask_path"], file_name)
    write_array2txt(file_path, shell_wall_filter_ptc)
    
    #%%
    return shell_wall_ptc_ids[shell_wall_filter_ptc_ids], shell_wall_filter_ptc
# In[Determine the range of the peak shape]
def peak_shape_range(data, bandwidth = 0.005, num_ptc = 100):
    """This function is to determine a range of the largest peak shape of data
    Input:
        data            : [Nx1]
        bandwidth

    Output:

    Demo:
                    
                    
    data = comb_grid_ptc[:,3]                   
    num_ptc = 100
    bandwidth = bandwidth
    """
    # Call libraries
#    from scipy.signal import find_peaks
    from sklearn.neighbors import KernelDensity                    

    # Generate kde
    range_locs = np.linspace(min(data), max(data), num_ptc)[:, None]
    kde_data = KernelDensity(kernel='epanechnikov', bandwidth = bandwidth).fit(data.reshape(-1, 1))
    kde_data_log = kde_data.score_samples(range_locs)
    kde_data_density = np.exp(kde_data_log)
    
    # Estimate the location of the highest peak
    max_peak_loc_id = np.argmax(kde_data_density)

    # Calculate a first derivative
    f_deriv_kde = first_derivative(range_locs, kde_data_density)
    s_deriv_kde = first_derivative(range_locs, f_deriv_kde)

    # Searching points on the left
    left_ids = np.arange(0, max_peak_loc_id-1, dtype = np.int16)
    # Extract the points with a negative values of first derivative of kde
    left_local_ids = np.where(f_deriv_kde[:max_peak_loc_id-1] <= 0)[0]    
    if left_local_ids.shape[0] > 0:
        start_peak = left_ids[np.max(left_local_ids)]
    else:
        start_peak = left_ids[0]
    
    # a location of a left range of the peak shape
    left_range_locs = range_locs[start_peak:max_peak_loc_id-1]
    left_s_deriv_kde = s_deriv_kde[start_peak:max_peak_loc_id-1]
    left_peak_shape_loc = left_range_locs[np.argmax(left_s_deriv_kde)]
    
    
    # Searching points on the right
    right_ids = np.arange(max_peak_loc_id+1, range_locs.shape[0], dtype = np.int16)
    # Extract the points with a negative values of first derivative of kde
    right_local_ids = np.where(f_deriv_kde[max_peak_loc_id+1:] >= 0)[0]    
    if right_local_ids.shape[0] > 0:
        end_peak = right_ids[np.min(right_local_ids)]
    else:
        end_peak = right_ids[-1]
    
    # a location of a left range of the peak shape
    right_range_locs = range_locs[max_peak_loc_id+1:end_peak]
    right_s_deriv_kde = s_deriv_kde[max_peak_loc_id+1:end_peak]
    right_peak_shape_loc = right_range_locs[np.argmax(right_s_deriv_kde)]
    
    # return 
    return np.append(left_peak_shape_loc, right_peak_shape_loc)
    
# In[Calculate the first derivative
def first_derivative(x, y):
    """This function is to compute first derivative of vector 
    
    
    # Demo:
    x = range_locs
    y = kde_data_density
    """
    # Check input variable
    x = np.array(x, copy = True).flatten()
    y = np.array(y, copy = True).flatten()
    
    f_deriv = np.full(x.shape[0], np.inf, dtype = np.float32)
    
    f_deriv[0] = (y[1] - y[0])/(x[1] - x[0])
    f_deriv[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    
    f_deriv[1:-2] = (y[2:-1] - y[0:-3])/(x[2:-1] - x[0:-3])
    
    return f_deriv