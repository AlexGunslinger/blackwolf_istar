#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 23:14:12 2018

@author: linhtruonghong
"""
import pandas as pd
import numpy as np
import copy
from itertools import chain  
from .data_structures import grid_extraction    
from .utilities import inNd, cosin3DVector, dist_3Dpoint_surfaces, wPCA

# In[Region growing]
def point_features(ptc, searching_neighbour):
    """This function is to compute features of points including normal vectors and residual
    for a point based region growing

    Input:
    Output:
    Demo:
        searching_neighbour = {'knn_search': 25}
    """
    # Call libraries
    from sklearn.neighbors import KDTree
    from Utilities import PCA_normal_vector, dist_3Dpoint_surfaces, residual

    # Check ptc
    if isinstance(ptc, pd.DataFrame):
        ptc = ptc[['x','y','z']].values
    elif isinstance(ptc, np.ndarray):
        ptc = ptc[:, :3]
    else:
        raise ValueError('Input direction does not support')

    # Searching neighbour points
    leaf_size = 50
    tree = KDTree(ptc, leaf_size = leaf_size)

    if 'range_search' in searching_neighbour.keys():
        # Range search
        searching_para = searching_neighbour['range_search'] if searching_neighbour['range_search'] > 0 else 0.1
        neighbour_ptc_ids = tree.query_radius(ptc, r = searching_para)
    elif 'knn_search' in searching_neighbour.keys():
		# kNN search
        searching_para = int(searching_neighbour['knn_search']) if int(searching_neighbour['knn_search']) > 5 else 20
        __, neighbour_ptc_ids = tree.query(ptc, k = searching_para)
    else:
        raise ValueError('The searching method does not support')

    # Compute a normal vector
    ptc_features = np.full((ptc.shape[0], 4), np.inf, dtype = float) #[nx, ny, nz, res]

    for i in range(ptc.shape[0]):
        ptc_i_neighbour_ptc_ids = neighbour_ptc_ids[i]
        ptc_i_neighbour_ptc = ptc[ptc_i_neighbour_ptc_ids]
        ptc_normal = PCA_normal_vector(ptc_i_neighbour_ptc, min_ptc = 5)
        if not any(np.isinf(ptc_normal)):
            ptc_center = np.mean(ptc_i_neighbour_ptc, axis = 0)
            ptc_dist = dist_3Dpoint_surfaces(ptc_i_neighbour_ptc, ptc_center, ptc_normal)
            ptc_residual = residual(ptc_dist)

            # Gathering results
            ptc_features[i] = np.hstack((ptc_normal, ptc_residual))

    # Remove singularity points
    # Remove singularity points
    mask = np.isinf(ptc_features[:, 0])
    ptc_local_ids = np.where(mask == False)[0]
    ptc_features = ptc_features[ptc_local_ids]
    ptc = ptc[ptc_local_ids]

    # Retrun
    return ptc, ptc_local_ids, ptc_features
# In[DB scan]
def DBscan_clustering(x_train, eps, min_pts):
    from sklearn.cluster import DBSCAN
    import pandas as pd
    '''
    x_train = cylinder_grid_rho[:,1]
    eps = 0.05
    min_pts = 5
    
    '''
    x_train = x_train.copy()
    
     
     
    # Convert to dataframe
    if not isinstance(x_train, pd.DataFrame):
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1,1)  
            
        x_train = pd.DataFrame(x_train, columns = ['val'])
    
    
    db = DBSCAN(eps = eps, min_samples = min_pts).fit(x_train)
#    labels = db.labels_
#    labels = labels + (1 if -1 in labels else 0)
    x_train['label'] = db.labels_
    x_train_segment = x_train[x_train['label'] >= 0]
    x_train_nonsegment = x_train[x_train['label'] < 0]
    

    # Return the results
    return x_train_segment, x_train_nonsegment    
# In[Segmentation] find all cells connection together in 1 dimension
def segmentation1d(cell, cell_size, length_threshold):
    
    '''This is the function to segment cells connected to together in 1 dimension.
    The region is considered if its length is larger than the threshold
    
    cell = grid_cell.copy()
    length_threshold = 2.0
    cell_size = 0.1
    '''
    
    cell = cell.copy()
    
    # Remove all empty cells
    cell = grid_extraction(cell, 1)
    
    # Intial region growing process 
    num_cells = cell['ids'].shape[0]
    
    # Result
    result_region = np.zeros((num_cells,2), dtype = int)
    result_region[:,0] = cell['ids'].flatten()
    
    # Check region
    check_region = result_region[:,0]
    
    region_order = 1
    
    while check_region.shape[0]>0:
        
        # Set up a current region
        current_region = np.array([check_region[0]], dtype = np.uint32)
        
        # Set up a seeding_region
        seed_region = np.array([check_region[0]], dtype = np.uint32)
       
        # Remove the first cell
        check_region = np.setdiff1d(check_region, check_region[0])
        
        while (seed_region.shape[0]>0) & (check_region.shape[0]>0):
            
            current_searching_cell_id = seed_region[0]
            
            # Searching a neighbour cell, next cell of current_searching_cell
            # Note the cell in ascend order
            neighbour_cell_id = check_region[0]
            
            if neighbour_cell_id == current_searching_cell_id + 1:
                
                # Update the current region
                current_region = np.append(current_region, neighbour_cell_id)
                
                # Update a seed region by adding a new cell
                seed_region = np.append(seed_region, neighbour_cell_id)
                
                # Update the check_region
                check_region = np.setdiff1d(check_region, check_region[0])
                
            # Update a seed region
            seed_region = np.setdiff1d(seed_region, current_searching_cell_id)
                
        # Assign the region order to a result_region
        if current_region.shape[0] >= np.ceil(length_threshold/cell_size):
            mask = np.in1d(result_region[:,0],current_region)
            result_region[mask,1] = region_order
            region_order += 1
    
    # Return results
    return result_region

# In[Segmentation in 2d]   
def segmentation2d(cell, cell_type = 1):
    
    '''This function is to segment cells connected in 2D
    cell = Shell_Cell.copy()
    '''
    
    if not isinstance(cell_type, int):
            raise ValueError('Cell_type must be an integer > 1')
    #Remove all empty cell
    cell = cell.copy()
    cell = grid_extraction(cell, cell_type)
    
    #%
    # Intial region growing process 
    num_cells = cell['ids'].shape[0]
    
    # Preallocation of result region
    
    result_region = np.zeros((num_cells,3), dtype = np.uint32)
    result_region[:,0:2] = cell['ids']#np.arange(0, num_cells)
    
    # Check region
    check_region = np.array(cell['ids'], dtype = np.uint32)
    
    # Reion growing process
    region_order = 1
    #%
    while check_region.shape[0] > 0:
        # Set up a current region
        current_region = np.array([check_region[0]], dtype = np.uint32)
        
        # Set up a seeding_region
        seed_region = np.array([check_region[0,:]], dtype = np.uint32)
       
        # Remove the first cell
        mask = inNd(check_region, check_region[0])
        check_region = np.delete(check_region, np.where(mask==True), axis = 0)
        
        #%
        while (seed_region.shape[0]>0)& (check_region.shape[0] > 0):
            
            seeding_cell_id = seed_region[0]
            
            # Searching the connected cells
            mask = (seeding_cell_id[0] - 1 <= check_region[:,0]) & (check_region[:,0] <= seeding_cell_id[0] + 1)&\
                   (seeding_cell_id[1] - 1 <= check_region[:,1]) & (check_region[:,1] <= seeding_cell_id[1] + 1)
            neighbour_cell_ids = check_region[mask]
            
            if neighbour_cell_ids.shape[0] > 0:
                
                # Update the current region
                current_region = np.vstack((current_region, neighbour_cell_ids))
                
                # Update a seed region by adding a new cell
                seed_region = np.vstack((seed_region, neighbour_cell_ids))
                
                # Update the check_region
                mask = inNd(check_region, neighbour_cell_ids)
                check_region = np.delete(check_region, np.where(mask==True), axis = 0)
                
            # Remove seeding_cell_id out of seed_region
            mask = inNd(seed_region, seeding_cell_id)
            seed_region = np.delete(seed_region, np.where(mask==True), axis = 0)
            
        # Assign the region order to a result_region
        mask = inNd(result_region[:,0:2], current_region)        
        result_region[mask,2] = region_order
        region_order += 1
    
    #% Return the results
    return result_region

# In[Cell-based segmentation]
def cell_based_segmentation(cell, cell_ptc, cell_features, 
                            ANGLE_THRESHOLD = 5*np.pi/180, 
                            ELEVATION_THRESHOLD = 0.01, RESIDUAL_THRESHOLD = 0.01):
    
    
    """This function is a cell-based region growing segmentation
    cell                    : a dictionary, [ids], [bounds], [ptc_ids]
    cell_ptc                : [Nx3] - List of the points used to create the cell
    cell_features           : [Nx8] - Features of the cell: [cell_id, cell_center, cell_normal, cell_residual]
    SEARCH_RANGE            : Scalar [m] - The value for searching points in cells around where two adjacent cells connected
    ANGLE_THRESHOLD         : Scalar[rad] - The maximum deviation between two normal vectors between two adjacent cells
    ELEVATION_THRESHOLD     : Scalar [m] - Maximum difference in the elevation of the point cloud in a connected region between two cells
    RESIDUAL_THRESHOLD      : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration
    
    # Demo:
        
    cell = shell_floor_cell
    cell_ptc = shell_floor_candidate_ptc
    cell_features = shell_floor_info
    SEARCH_RANGE = CELL_SIZE
    ANGLE_THRESHOLD = ANGLE_THRESHOLD
    RESIDUAL_THRESHOLD = RESIDUAL_THRESHOLD
    ELEVATION_THRESHOLD = ELEVATION_THRESHOLD
    
    """
#    import time

    # Translate angle threshold to cosin value
    COSIN_THRESHOLD = np.cos(ANGLE_THRESHOLD)

    # Retrieve from cell
    cell_ptc_ids = cell["ptc_ids"]
    cell_grid_ids = cell["ids"].astype(int)
    cell_bounds = cell["bounds"]
    CELL_SIZE = np.round(cell_bounds[0,2] - cell_bounds[0,0], 3)
    
    # Retrieve cell_info
    cell_ids = cell_features[:, 0]
#    cell_center = cell_features[:, 1:4]
    cell_normal = cell_features[:, 4:7]
    cell_residual = cell_features[:, -1]
    
    # Initialize variable for a region growing
    new_cell_ids = np.arange(0, len(cell_ids))
    global_cell_region = np.zeros([len(cell_ids), 2], dtype = np.uint32)
    global_cell_region[:, 0] = new_cell_ids
    exam_cell_info = np.zeros([len(cell_ids), 2])
    exam_cell_info[:, 0], exam_cell_info[:,1] = new_cell_ids, cell_residual
    
    region_no = 1    
    while exam_cell_info.shape[0] != 0:
        
        # Searching the initial seeding cell
        mask = np.argmin(exam_cell_info[:, 1])
        cell_seeding_id = exam_cell_info[mask, 0].astype(int)
        seeding_cell_region = np.array([cell_seeding_id], dtype = np.int64)        
		
        # Current region
        current_cell_region = np.array([cell_seeding_id], dtype = np.int64) 
        
        # Remove point with a minimum residual
        mask = np.in1d(exam_cell_info[:, 0], cell_seeding_id)
        match_cell_id = np.where(mask == True)[0]
        exam_cell_info = np.delete(exam_cell_info, match_cell_id, axis = 0)
		#%
        while seeding_cell_region.shape[0] > 0:
            
            # Initia seeding cell
            init_cell_id = seeding_cell_region[0]
            init_cell_grid_id = cell_grid_ids[init_cell_id]
            
            # Extract neighbour cell
            mask = np.all((init_cell_grid_id - 1 <= cell_grid_ids) & (cell_grid_ids <= init_cell_grid_id + 1), axis = 1)
            neighbour_cell_ids = new_cell_ids[mask]
            neighbour_cell_ids = np.setdiff1d(neighbour_cell_ids, init_cell_id)
            
            # Check neighbour cell in exam_cell_info
            mask = np.in1d(neighbour_cell_ids, exam_cell_info[:, 0])
            neighbour_cell_ids = neighbour_cell_ids[mask]
            
            # Compare normals
            if neighbour_cell_ids.shape[0] >= 1:
                
                # Angle between normals
                init_cell_normal = cell_normal[init_cell_id]
                neighbour_cell_normals = cell_normal[neighbour_cell_ids]
                cosin_init_neighbour_cells = cosin3DVector(init_cell_normal, neighbour_cell_normals)
                
                # Check conditions for selecting cells of the region
                mask = (np.abs(cosin_init_neighbour_cells) >= COSIN_THRESHOLD) 
                add_cell_ids = neighbour_cell_ids[mask]

                if add_cell_ids.shape[0] >= 1:
                    
                    # Check connectivity in elevation
                    flag = surface_continuity(cell_ptc, cell_ptc_ids, cell_bounds, cell_grid_ids, 
                                              init_cell_id, add_cell_ids, SEARCH_RANGE = CELL_SIZE, ELEVATION_THRESHOLD = ELEVATION_THRESHOLD)
                    
                    add_cell_ids = add_cell_ids[flag]
                    
                    if add_cell_ids.shape[0] >= 1:
                        # Update current region
                        current_cell_region = np.append(current_cell_region, add_cell_ids)
                        
                        # Update exam_cell_ids
                        mask = np.in1d(exam_cell_info[:, 0], add_cell_ids)
                        match_cell_id = np.where(mask == True)[0]
                        exam_cell_info = np.delete(exam_cell_info, match_cell_id, axis = 0)
        		
                        # Update seeding cell region
                        add_cell_residual = cell_residual[add_cell_ids]
                        mask = add_cell_residual <= RESIDUAL_THRESHOLD
                        if sum(mask) >= 1:
                            seeding_cell_region = np.append(seeding_cell_region, add_cell_ids[mask])
            
            # Remove check seeding cell
            seeding_cell_region = np.setdiff1d(seeding_cell_region, init_cell_id)   
        
        # Assign the region
        mask = np.in1d(global_cell_region[:, 0], current_cell_region)
        global_cell_region[mask, 1] = region_no + 1
        
        region_no += 1
    
    # Return results
    return global_cell_region

# In[Statistic region]
def region_summary(region_info, num_region = 1):
    
    """This fuction is to return the cell ids in the number of the first largest regions
    region_info     : [Nx2] - The array contains cell_ids and its region
    num_region      : Scalar - The first number of largest reion to be extracted.
    
    Demo:
        
    region_info = floor_cell_region
    num_region = 1
    
    """
    if not isinstance(region_info, np.ndarray):
        raise AttributeError('The format of the variable region_info does not support') 
 
    if num_region > np.max(region_info[:,1]):
        raise AttributeError('The requested number of the region by %s larger then the available region' % num_region)
    # Statistic regions
    region_count = np.bincount(region_info[:,1])
    region_id = np.nonzero(region_count)[0]
    region_stat = np.vstack((region_id, region_count[region_id])).T #[region_id, num_cell_in_region]
    
    # Extract the points of the largest region
    
    large_region_order = np.argsort(region_stat[:,1])[::-1][:num_region]
    large_region_ids = region_stat[large_region_order, 0]
#    large_region_ids = np.asscalar(large_region_ids)
    mask = np.in1d(region_info[:,1],large_region_ids)
    large_region_cell_ids = region_info[mask,0]
    small_region_cell_ids = region_info[~mask,0]
    
    return large_region_cell_ids, small_region_cell_ids

# In[Refinement merging points from small segment to large segment]   

def small_2_large_regions(cell, cell_ptc, large_region_cell_ids, small_region_cell_ids,
                              RESIDUAL_THRESHOLD = 0.01,
                              ANGLE_DEVIATION = 0.5*np.pi/180, MIN_PTS = 5):
    
    """This function is to merge points in cells of the small regions to the large region
    cell                    : a dictionary, [ids], [bounds], [ptc_ids]
    cell_ptc                : [Nx3] - List of the points used to create the cell
    large_region_cell_ids   : [Mx1] - The array contains the list of cells in the large region
    small_region_cell_ids   : [Kx1] - The array contains the list of cells in all small regions
    RESIDUAL_THRESHOLD      : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration
    ANGLE_DEVIATION         : Rad - The angle threshold for estimating the normal of the cell by using weighted PCA
    MIN_PTS                 : scalar - The minimum number of points for considering the cell as well as estimating a normal vector
    
    large_region_cell_ids = large_region[0]
    
    """
    
    # Retrieve from cell
    cell_ptc_ids = cell["ptc_ids"]
    cell_grid_ids = cell["ids"].astype(int)

    # Refinement step from small -> large
    small_region_add_ptc_ids = []
    
    large_region_cell_grid_id = cell_grid_ids[large_region_cell_ids]
    for i in small_region_cell_ids:
        
        small_region_current_cell_grid_id = cell_grid_ids[i]
        # Search neighbour
        mask = np.all((small_region_current_cell_grid_id - 1 <= large_region_cell_grid_id) & 
                      (large_region_cell_grid_id <= small_region_current_cell_grid_id + 1), axis = 1)
        neighbour_cell_ids = large_region_cell_ids[mask]
        
        if neighbour_cell_ids.shape[0] >= 1:
            
            # Retrieve points in the neighbour cells
            neighbour_cell_ptc_ids = list(chain.from_iterable(cell_ptc_ids[ids] for ids in neighbour_cell_ids))
            neighbour_cell_ptc = cell_ptc[neighbour_cell_ptc_ids]
    
            # Compute a new center and normal for these points
            ptc_normal, ptc_center = wPCA(neighbour_cell_ptc, tol = ANGLE_DEVIATION, min_ptc = MIN_PTS)
        
            # Compute distances from the points in small region cell to this local surface
            small_region_cell_ptc_ids = cell_ptc_ids[i]
            small_region_cell_ptc = cell_ptc[small_region_cell_ptc_ids]
            dist_ptc_cell = dist_3Dpoint_surfaces(small_region_cell_ptc, ptc_center, ptc_normal)
             
            # Extract the points within the buffer
            mask = np.abs(dist_ptc_cell) <= RESIDUAL_THRESHOLD
            merging_ptc_ids = small_region_cell_ptc_ids[mask]
            
            if merging_ptc_ids.shape[0] >= 1:
                small_region_add_ptc_ids = np.append(small_region_add_ptc_ids, merging_ptc_ids)
    
    # Return the results
    return small_region_add_ptc_ids.astype(int), cell_ptc[small_region_add_ptc_ids.astype(int)]

# In[Refinement for non_cell points -> large region]
def ptc_2_large_regions(cell, cell_ptc, cell_features, large_region_cell_ids, RESIDUAL_THRESHOLD = 0.01):
    
    """ This function is to merge the points no located any cell to the large region
    cell                    : a dictionary, [ids], [bounds], [ptc_ids]
    cell_ptc                : [Nx3] - List of the points used to create the cell
    cell_features           : [Nx8] - Features of the cell: [cell_id, cell_center, cell_normal, cell_residual]
    large_region_cell_ids   : [Mx1] - The array contains the list of cells in the large region
    RESIDUAL_THRESHOLD      : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration

    """
    # Retrieve from cell
    cell_ptc_ids = cell["ptc_ids"]
    cell_bounds = cell["bounds"]
    CELL_SIZE = np.round(cell_bounds[0,2] - cell_bounds[0,0], 3)
    
    # Retrieve cell_info
    cell_center = cell_features[:, 1:4]
    cell_normal = cell_features[:, 4:7]
    
    # Extract float points
    non_cell_ptc_ids = np.setdiff1d(np.arange(cell_ptc.shape[0]), list(chain.from_iterable(cell_ptc_ids)))
    non_cell_ptc = cell_ptc[non_cell_ptc_ids]
    
    # Initialisa points not located any cell to be merged
    non_cell_add_ptc_ids = []
    
    for i in large_region_cell_ids:
        
        # Current cell
        large_region_current_cell_bounds = cell_bounds[i] 
        
        # Searching points from non_cell within the window
        mask = np.all((large_region_current_cell_bounds[:2] - CELL_SIZE <= non_cell_ptc[:,:2]) & 
                      (non_cell_ptc[:,:2] <= large_region_current_cell_bounds[2:] + CELL_SIZE), axis = 1)
        neighbour_ptc_ids = non_cell_ptc_ids[mask]
        
        if neighbour_ptc_ids.shape[0] >= 1:
            
            # Compute distance from the points to the lcal surface
            large_region_current_cell_center = cell_center[i]
            large_region_current_cell_normal = cell_normal[i]
            dist_ptc_cell = dist_3Dpoint_surfaces(cell_ptc[neighbour_ptc_ids], large_region_current_cell_center, large_region_current_cell_normal)
             
            # Extract the points within the buffer
            mask = np.abs(dist_ptc_cell) <= RESIDUAL_THRESHOLD
            merging_ptc_ids = neighbour_ptc_ids[mask]
                
            if merging_ptc_ids.shape[0] >= 1:
                non_cell_add_ptc_ids = np.append(non_cell_add_ptc_ids, merging_ptc_ids)

    # Return result    
    return non_cell_add_ptc_ids.astype(int), cell_ptc[non_cell_add_ptc_ids.astype(int)]
# In[Check if the elevation between two closes points]
def surface_continuity(cell_ptc, cell_ptc_ids, cell_bounds, cell_grid_ids, source_cell_id, target_cell_ids, 
                       SEARCH_RANGE = 0.5/4, ELEVATION_THRESHOLD = 0.01):

    """This function is to whether two adjacent cells parts of the surface, where their boundaries are continuity
    cell_ptc                : [Nx3] x, y, z coordinates of all points potential the shell floor
    cell_ptc_ids            : [Nx...] - Array contains point ids in each cells
    cell_bounds             : [Nx4] - [x_min, y_min, x_max_y_max] of all cells
    cell_grid_ids           : [Nx2] - [i , j] 
    source_cell_id          : The cell id 
    target_cell_ids         : The list of the cell adjacent to the source cell
    SEARCH_RANGE            : Scalar [m] - The value for searching points in cells around where two adjacent cells connected
    ELEVATION_THRESHOLD     : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration

    """
     
    # Call a functions
    from Utilities import vector_length
    
    # Retrieve points in a source cell
    source_cell_ptc_ids = cell_ptc_ids[source_cell_id]
    source_cell_ptc = cell_ptc[source_cell_ptc_ids]
    
    # Check through each target_cell
    flag = np.zeros(target_cell_ids.shape[0], dtype = bool)
    
    for i in range(len(target_cell_ids)):
    
        target_cell_id = target_cell_ids[i]
        # Check a connection type
        pos, pos_vert = cell_type_connection(cell_bounds, cell_grid_ids, source_cell_id, target_cell_id)

        # Retrieve points in a target cell
        target_cell_ptc_ids = cell_ptc_ids[target_cell_id]
        target_cell_ptc = cell_ptc[target_cell_ptc_ids]

        # Extract points in the cell
        
        if pos % 2:         # Vertice connection
            
            # Search points in sources 
            source_to_post_vert_length = vector_length(source_cell_ptc[:,:-1] - pos_vert)
            mask = source_to_post_vert_length <= SEARCH_RANGE
            source_cell_ptc_z = source_cell_ptc[mask, 2]
            
            # Search points in target 
            target_to_post_vert_length = vector_length(target_cell_ptc[:,:-1] - pos_vert)
            mask = target_to_post_vert_length <= SEARCH_RANGE
            target_cell_ptc_z = target_cell_ptc[mask, 2]
            
        else:               # Side connection

            # Window search            
            search_window_vert_left = pos_vert[:2] - SEARCH_RANGE
            search_window_vert_right = pos_vert[2:] + SEARCH_RANGE

            # Search points in sources
            mask = np.all((search_window_vert_left <= source_cell_ptc[:,:-1]) & (source_cell_ptc[:,:-1] <= search_window_vert_right), axis = 1)
            source_cell_ptc_z = source_cell_ptc[mask, 2]
            
            # Search points in target
            mask = np.all((search_window_vert_left <= target_cell_ptc[:,:-1]) & (target_cell_ptc[:,:-1] <= search_window_vert_right), axis = 1)
            target_cell_ptc_z = target_cell_ptc[mask, 2]
        
        # Check a continuity condition
        if  (source_cell_ptc_z.shape[0] >= 2) & (target_cell_ptc_z.shape[0] >= 2): 
            if ((np.abs(np.min(source_cell_ptc_z) - np.max(target_cell_ptc_z)) <= ELEVATION_THRESHOLD) or
                (np.abs(np.min(target_cell_ptc_z) - np.max(source_cell_ptc_z)) <= ELEVATION_THRESHOLD)):
                
                flag[i] = True
    # Return results
    return flag
# In[Determine types of the connection between two adjacent cells]
def cell_type_connection(cell_bounds, cell_grid_ids, source_cell_id, target_cell_id): 
    
    """This function is to determine how two adjacent cells connected each orther
    
    cell_bounds     : [Nx4] - [x_min, y_min, x_max_y_max] of all cells
    cell_grid_ids   : [Nx2] - [i , j] 
    source_cell_id  : The cell id 
    target_cell_ids : The list of the cell adjacent to the source cell
    
    Demo:
        
    cell_bounds
    source_cell_id = init_cell_id
    target_cell_id = j
    
    """
    
    source_cell_grid_id = cell_grid_ids[source_cell_id]
    target_cell_grid_id = cell_grid_ids[target_cell_id]
    
    source_cell_bounds = cell_bounds[source_cell_id]

    if source_cell_grid_id[0] == target_cell_grid_id[0] + 1:
        if source_cell_grid_id[1] == target_cell_grid_id[1] + 1:
            pos = 1
            pos_vert = source_cell_bounds[:2]
        elif source_cell_grid_id[1] == target_cell_grid_id[1]:
            pos = 2
            pos_vert = [source_cell_bounds[0], source_cell_bounds[1], source_cell_bounds[0], source_cell_bounds[3]]
        else:
            pos = 3
            pos_vert = [source_cell_bounds[0], source_cell_bounds[3]]
            
    elif source_cell_grid_id[0] == target_cell_grid_id[0]:
        if source_cell_grid_id[1] == target_cell_grid_id[1] + 1:
            pos = 8
            pos_vert = [source_cell_bounds[0], source_cell_bounds[1], source_cell_bounds[2], source_cell_bounds[1]]
        else:
            pos = 6
            pos_vert = [source_cell_bounds[0], source_cell_bounds[3], source_cell_bounds[2], source_cell_bounds[3]]
    
    else:  
        if source_cell_grid_id[1] == target_cell_grid_id[1] - 1:
            pos = 5
            pos_vert = source_cell_bounds[2:]
        elif source_cell_grid_id[1] == target_cell_grid_id[1]:
            pos = 6
            pos_vert = [source_cell_bounds[2], source_cell_bounds[1], source_cell_bounds[2], source_cell_bounds[3]]
        else:
            pos = 7
            pos_vert = [source_cell_bounds[2], source_cell_bounds[1]]            

    # Return the results
    
    return pos, np.asarray(pos_vert)
   
# In[Segmentation in 2d]   
def cell_connectivity(cell, cell_type = 1):
    
    '''This function is to segment cells connected in 2D
    cell = shell_wall.copy()
    '''
    
    if not isinstance(cell_type, int):
            raise ValueError('Cell_type must be an integer > 1')
    #Remove all empty cell
    cell = cell.copy()
    cell = grid_extraction(cell, cell_type)

    # Retrieve from cell
    cell_grid_ids = cell["ids"].astype(int)

    # Initialize variable for a region growing
    num_cell = cell['ids'].shape[0]
   
    # Preallocation of result region
    global_cell_region = np.zeros([num_cell, 2], dtype = np.uint32)
    global_cell_region[:, 0] = np.arange(0, num_cell)
    exam_cell_info = np.arange(0, num_cell)
    
    region_no = 0    
    while exam_cell_info.shape[0] != 0:
        
        # Searching the initial seeding cell
        seeding_cell_region = np.array([exam_cell_info[0]], dtype = np.int64)        
		
        # Current region
        current_cell_region = np.array([exam_cell_info[0]], dtype = np.int64) 
        
        # Remove point with a minimum residual
        exam_cell_info = np.delete(exam_cell_info, 0, axis = 0)
		#%
        while seeding_cell_region.shape[0] > 0:
            
            # Initia seeding cell
            init_cell_id = seeding_cell_region[0]
            init_cell_grid_id = cell_grid_ids[init_cell_id]
            
            # Extract neighbour cell
            mask = np.all((init_cell_grid_id - 1 <= cell_grid_ids[exam_cell_info]) & (cell_grid_ids[exam_cell_info] <= init_cell_grid_id + 1), axis = 1)
            neighbour_cell_ids = exam_cell_info[mask]
           
            
            if neighbour_cell_ids.shape[0] >= 1:
                
                # Update current region
                current_cell_region = np.append(current_cell_region, neighbour_cell_ids)
                        
                # Update exam_cell_ids
                exam_cell_info = np.setdiff1d(exam_cell_info, neighbour_cell_ids)
                
                # Update seeding cell region
                seeding_cell_region = np.append(seeding_cell_region, neighbour_cell_ids)
            
            # Remove check seeding cell
            seeding_cell_region = np.setdiff1d(seeding_cell_region, init_cell_id)   
        
        # Assign the region
        global_cell_region[current_cell_region, 1] = region_no + 1
        
        region_no += 1
    
    # Return results
    return global_cell_region   
    
    
# In[2D cell based]
def cell2D_based_segmentation(cell, cell_features, 
                            ANGLE_THRESHOLD = 5*np.pi/180, 
                            ELEVATION_THRESHOLD = 0.01, RESIDUAL_THRESHOLD = 0.01):
    
    
    """This function is a cell-based region growing segmentation
    cell                    : a dictionary, [ids], [bounds], [ptc_ids]
    cell_features           : [Nx8] - Features of the cell: [cell_id, cell_center, cell_normal, cell_residual]
    SEARCH_RANGE            : Scalar [m] - The value for searching points in cells around where two adjacent cells connected
    ANGLE_THRESHOLD         : Scalar[rad] - The maximum deviation between two normal vectors between two adjacent cells
    ELEVATION_THRESHOLD     : Scalar [m] - Maximum difference in the elevation of the point cloud in a connected region between two cells
    RESIDUAL_THRESHOLD      : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration
    
    # Demo:
        
    cell = floor_cell
    cell_features = floor_info
    ANGLE_THRESHOLD = ANGLE_THRESHOLD
    RESIDUAL_THRESHOLD = RESIDUAL_THRESHOLD
    ELEVATION_THRESHOLD = ELEVATION_THRESHOLD
    
    """
#%%    import time
    from Utilities import dist_3Dpoint_surfaces
    # Translate angle threshold to cosin value
    COSIN_THRESHOLD = np.cos(ANGLE_THRESHOLD)

    # Retrieve cell_info
    cell_ids = cell_features[:, 0].astype(np.int32)
    cell_grid_ids = cell["ids"][cell_ids]
    
    # Initialize variable for a region growing
#    new_cell_ids = np.arange(0, len(cell_ids))
    global_cell_region = np.full([cell_ids.shape[0], 2], np.inf, dtype = np.int32)
    global_cell_region[:, 0] = cell_ids
    exam_cell_info = np.full([cell_ids.shape[0], 2], np.inf, dtype = np.float32)
    exam_cell_info[:, 0], exam_cell_info[:,1] = cell_features[:, 0], cell_features[:, -1]
    
    region_no = 1    
    while exam_cell_info.shape[0] > 0:
        
        # Searching the initial seeding cell
        mask = np.argmin(exam_cell_info[:, 1])
        cell_seeding_id = exam_cell_info[mask, 0].astype(np.int32)
        seeding_cell_region = np.array([cell_seeding_id], dtype = np.int32)        
		
        # Current region
        current_cell_region = np.array([cell_seeding_id], dtype = np.int32) 
        
        # Remove point with a minimum residual
        mask = np.in1d(exam_cell_info[:, 0].astype(np.int32), cell_seeding_id)
        match_cell_id = np.where(mask == True)[0]
        exam_cell_info = np.delete(exam_cell_info, match_cell_id, axis = 0)
		#%
        while seeding_cell_region.shape[0] > 0:
            
            # Initia seeding cell
            init_cell_id = seeding_cell_region[0]
            init_cell_grid_id = cell["ids"][init_cell_id]
            
            # Extract neighbour cell
            mask = np.all((init_cell_grid_id - 1 <= cell_grid_ids) & (cell_grid_ids <= init_cell_grid_id + 1), axis = 1)
            neighbour_cell_ids = cell_ids[mask]
            neighbour_cell_ids = np.setdiff1d(neighbour_cell_ids, init_cell_id)
            
            # Check neighbour cell in exam_cell_info
            mask = np.in1d(neighbour_cell_ids, exam_cell_info[:, 0].astype(np.int32))
            neighbour_cell_ids = neighbour_cell_ids[mask]
            
            # Compare normals
            if neighbour_cell_ids.shape[0] >= 1:
                
                # Angle between normals between initial cell and neighbour
                mask = np.in1d(cell_features[:,0].astype(np.int32), init_cell_id)
                init_cell_center = cell_features[mask, 1:4].flatten()
                init_cell_normal = cell_features[mask, 4:7].flatten()
                mask = np.in1d(cell_features[:,0].astype(np.int32), neighbour_cell_ids)
                neighbour_cell_centers = cell_features[mask, 1:4]
                neighbour_cell_normals = cell_features[mask, 4:7]
                cosin_init_neighbour_cells = cosin3DVector(init_cell_normal, neighbour_cell_normals)
                
                dist_cell_cells = dist_3Dpoint_surfaces(neighbour_cell_centers, init_cell_center, init_cell_normal)
                
                
                # Check conditions for selecting cells of the region
                mask = (np.abs(cosin_init_neighbour_cells) >= COSIN_THRESHOLD)&(np.abs(dist_cell_cells) <= ELEVATION_THRESHOLD) 
                add_cell_ids = neighbour_cell_ids[mask]

                if add_cell_ids.shape[0] >= 1:

                    # Update current region
                    current_cell_region = np.append(current_cell_region, add_cell_ids)
                    
                    # Update exam_cell_ids
                    mask = np.in1d(exam_cell_info[:, 0].astype(np.int32), add_cell_ids)
                    match_cell_id = np.where(mask == True)[0]
                    exam_cell_info = np.delete(exam_cell_info, match_cell_id, axis = 0)
    		
                    # Update seeding cell region
                    mask = np.in1d(cell_features[:,0].astype(np.int32), add_cell_ids)
                    add_cell_residual = cell_features[mask,-1]
                    mask = add_cell_residual <= RESIDUAL_THRESHOLD
                    if sum(mask) >= 1:
                        seeding_cell_region = np.append(seeding_cell_region, add_cell_ids[mask])
            
            # Remove check seeding cell
            seeding_cell_region = np.setdiff1d(seeding_cell_region, init_cell_id)   
        
        # Assign the region
        mask = np.in1d(global_cell_region[:, 0], current_cell_region)
        global_cell_region[mask, 1] = region_no
        
        region_no += 1
    
    #% Update the region number
    region_count = np.bincount(global_cell_region[:,1])
    region_id = np.nonzero(region_count)[0]
    region_stat = np.vstack((region_id, region_count[region_id])).T #[region_id, num_cell_in_region]
    
    # Extract the points of the largest region
    mask = np.argsort(region_stat[:, 1])[::-1]
    region_stat = region_stat[mask]
    
    #% Update the global region by new region
    global_region_new_ids = np.zeros_like(global_cell_region[:, 1], dtype = np.int32)
    for new_region_id, old_region_id in enumerate(region_stat[:, 0]):
        mask = np.in1d(global_cell_region[:, 1], old_region_id)
        
        global_region_new_ids[mask] = new_region_id+1
    global_cell_region[:, 1] = global_region_new_ids

    # Return results
    return global_cell_region
 
# In[Merging points in the smaller region into large]

def add_ptc_2_large_regions(cell, cell_ptc, cell_features, large_region_cell_ids, small_region_cell_ids, MIN_PTS = 10):
    
    """This function is to merge points in cells of the small regions to the large region
    cell                    : a dictionary, [ids], [bounds], [ptc_ids]
    cell_ptc                : [Nx3] - List of the points used to create the cell
    large_region_cell_ids   : [Mx1] - The array contains the list of cells in the large region
    small_region_cell_ids   : [Kx1] - The array contains the list of cells in all small regions
    RESIDUAL_THRESHOLD      : Scalar[m] - The maximum residual of the cell to be seeding cells for next iteration
    ANGLE_DEVIATION         : Rad - The angle threshold for estimating the normal of the cell by using weighted PCA
    MIN_PTS                 : scalar - The minimum number of points for considering the cell as well as estimating a normal vector
    
    large_region_cell_ids = large_region[0]
    
    cell = origin_floor_cell.copy()
    cell_ptc = floor_candidate_ptc.copy()
    """
    #%% Call library
    from Utilities import PCA_normal_vector
    
    # Refinement step from small -> large
    small_region_add_ptc_ids = []
    
    check_cell_ids = np.array(large_region_cell_ids, copy = True, dtype = np.int32)
    seed_cell_ids = np.array(small_region_cell_ids, copy = True, dtype = np.int32)
    while check_cell_ids.shape[0] > 0:
        
        check_initial_cell_ids = check_cell_ids[0]
        
        # Remove check_initial_cell_ids
        check_cell_ids = np.setdiff1d(check_cell_ids, check_initial_cell_ids)
        
        # Searching neighbour cell
        mask = np.all((cell["ids"][check_initial_cell_ids] - 1 <= cell["ids"][seed_cell_ids]) & 
                      (cell["ids"][seed_cell_ids] <= cell["ids"][check_initial_cell_ids] + 1), axis = 1)
        neighbour_cell_ids = seed_cell_ids[mask]
        
        if neighbour_cell_ids.shape[0] > 0:
            # Retrieve the features of the check cell
            mask = np.in1d(cell_features[:, 0].astype(np.int32), check_initial_cell_ids)
            check_initial_cell_center = cell_features[mask, 1:4].flatten()
            check_initial_cell_normal = cell_features[mask, 4:7].flatten()
            check_initial_cell_residual = cell_features[mask, -1].flatten()
            # Check if any point in the cell satisfy a distance condition
            for neighbour_cell_id in neighbour_cell_ids:
                # Retrieve the points within the neighbour cell
                neighbour_cell_ptc_ids = cell["ptc_ids"][neighbour_cell_id]
                neighbour_cell_ptc = cell_ptc[neighbour_cell_ptc_ids]
                # Check distance
                dist_ptc_cell = dist_3Dpoint_surfaces(neighbour_cell_ptc, check_initial_cell_center, check_initial_cell_normal)
                mask = np.abs(dist_ptc_cell) <= 5.0*check_initial_cell_residual
                neighbour_cell_add_ptc_ids = neighbour_cell_ptc_ids[mask]
                neighbour_cell_add_ptc = neighbour_cell_ptc[mask]
                
                if neighbour_cell_add_ptc_ids.shape[0] >= MIN_PTS:
                    # Update add ptc
                    small_region_add_ptc_ids = np.append(small_region_add_ptc_ids,neighbour_cell_add_ptc_ids)
                    
                    # Add neighbour cell to check cell ids
                    check_cell_ids = np.union1d(check_cell_ids,neighbour_cell_id)
                    
                    # Remove seed_cell_ids
                    seed_cell_ids = np.setdiff1d(seed_cell_ids,neighbour_cell_id)
                    
                    # Recompute new features and update cell_features but the residual use from the large segment
                    new_ptc_center = np.mean(neighbour_cell_add_ptc, axis = 0)
                    new_ptc_normal = PCA_normal_vector(neighbour_cell_add_ptc, min_ptc = 5)
                    mask = np.in1d(cell_features[:, 0].astype(np.int32), neighbour_cell_id)
                    
                    cell_features[mask, 1:7] = np.concatenate((new_ptc_center, new_ptc_normal))
                    cell_features[mask, -1] = check_initial_cell_residual 

    #% Return the results
    small_region_add_ptc_ids = np.asarray(small_region_add_ptc_ids, dtype = np.int32)
    return small_region_add_ptc_ids, cell_ptc[small_region_add_ptc_ids]
           