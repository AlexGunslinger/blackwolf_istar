#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:00:37 2018

@author: linhtruonghong
"""
import pandas as pd
import numpy as np
import copy
# In[Update cell - data structure as dict]
def update_data_structure(cell, cell_ids):
    '''This function is to update the data structure stored as dict. Only ids in
    the list to be updated'''

    # Update the cell
    update_cell = dict()
    for key,__ in cell.items():
    
        if 'ptc_ids' in key:
            
            update_cell[key] = [cell[key][ids] for ids in cell_ids]
        else:
            update_cell[key] = cell[key][cell_ids]
    
    # Return 
    return update_cell

# In[Function to extract the type of the cells]

def grid_extraction(cell, cell_type):
    
    if cell_type == 1:
        
        # Full cell
        mask = np.where(cell['prop'] == True)[0]
        
    else:    
        # Empty cell
        mask = np.where(cell['prop'] == False)[0]
     
    
    filter_cell = dict()
    for key,__ in cell.items():
        
        if 'ptc_ids' in key:
            
            if cell_type != 0:
                # This option is only available for full cells
                filter_cell[key] = [cell[key][ids] for ids in mask]
                
        else:
            
            filter_cell[key] = cell[key][mask]
    
    
    return filter_cell

# In[Compute a height of the cells]

def cal_cell_height(cell, ptc):
    
    ''' The function is to calculate the cell's height and maximum gap between
    the point cloud in cell. An efficiency way can be obtained by filtering 
    the long cell firstly, which has difference of the bound in z coordinate.
    Demo
    The function
    cell = Cell.copy()
    ptc = PTC.copy()
    
    '''
    # Convert dataframe into array
    if isinstance(ptc, pd.DataFrame):
        ptc = ptc.iloc[:,:3].values
    else:
        ptc = ptc[:,:3]
        
    # Retrieve cell information    
    num_cell = len(cell['ids'])
    cell_ptc_ids = cell['ptc_ids']

    # Prelocation     
    cell_height = np.zeros((num_cell, 2), dtype = float) #[]
#%    
    for i in range(num_cell):
        
        # Extract ptc_ids in the cell
        current_cell_ptc_ids = cell_ptc_ids[i]
        
        if current_cell_ptc_ids.shape[0] > 10:
            # Retrieve z coordinates of the points
        
            cell_ptc_z = ptc[current_cell_ptc_ids, 2]
            cell_ptc_z = np.sort(cell_ptc_z)
        
            # Calculate the height and different gaps
        
            z = cell_ptc_z[-1] - cell_ptc_z[0]
            diff_z = np.diff(cell_ptc_z, n = 1, axis = 0)
            max_diff_z = np.max(abs(diff_z))
            
        else: # Consider as the small cell in term of the number of the point
            z = 0
            max_diff_z = 0
        
        # Assign the values
        cell_height[i] = [z, max_diff_z]
#        
    return cell_height   
# In[1D grid]

def grid1d(ptc, s_coord, e_coord, interval_size, direction, min_ptc = 1):
    ''' This function is to subdive the cell into small cells in 1 direction
    ptc = cell_ptc_dir.copy()
    
    s_coord = z_min
    e_coord = z_max
    interval_size = cell_size
    direction = 'z'
    
    cell_ptc = nonshell_cell_ptc_dir
    ptc = df_shell_cell_ptc_dir.copy()
    s_coord = ptc_dir_min
    e_coord = ptc_dir_max
    interval_size = interval_thickness
    direction = 'z'
       
    '''
    # Check input paramaters and initialisation
    
    if isinstance(ptc, pd.DataFrame):

        if direction in ptc.columns:
            ptc_ids = ptc.index
            ptc = ptc[direction].values
        
        else:
            
            raise ValueError('The input data does not contain points in direction')
            
    elif isinstance(ptc, np.ndarray):
        
        if ptc.ndim == 1:
            # Point index
            ptc_ids = np.arange(ptc.shape[0])            
        else:
            if direction.casefold() == 'x'.casefold():
                ptc = ptc[:, 0]
            elif direction.casefold() == 'y'.casefold():            
                ptc = ptc[:, 1]            
            elif direction.casefold() == 'z'.casefold():            
                ptc = ptc[:, 2]
            elif (direction.casefold() == 'rho'.casefold()):
                ptc = ptc[:, 3]
            elif (direction.casefold() == 'theta'.casefold()):    
                ptc = ptc[:, 4]
            else:
                raise ValueError('Input direction does not support')
            # Point index
            ptc_ids = np.arange(ptc.shape[0])
    else:
        raise ValueError('The input data does not support')
               
    # Check bounds
    if isinstance(s_coord, np.ndarray):
        s_coord = np.asscalar(s_coord)
        
    if isinstance(e_coord, np.ndarray):
        e_coord = np.asscalar(e_coord)
     
    # Create the grid   
    num_cells = np.ceil((e_coord - s_coord)/interval_size).astype(int)  
    grid_coord = np.linspace(s_coord, s_coord + num_cells*interval_size, num = num_cells+1, endpoint = True)
    if grid_coord[-1] != e_coord: grid_coord[-1] = e_coord
        
    # Preallocation
    grid_cell_ids = np.zeros((num_cells,1), dtype = np.int32)
    grid_cell_prop = np.zeros((num_cells,1), dtype = bool)
    grid_cell_ptc_id = dict()

    # Determine properties of the cells

    for i in range(0, num_cells):

        # Extract point indices in a range
        if i+1 == num_cells:
            mask = (grid_coord[i] <= ptc)&(ptc <= grid_coord[i+1])
        else:
            mask = (grid_coord[i] <= ptc)&(ptc < grid_coord[i+1])

        cell_ptc_ids = ptc_ids[mask]
        if cell_ptc_ids.shape[0] >= min_ptc:
            grid_cell_ids[i] = i
            grid_cell_prop[i] = True
            grid_cell_ptc_id[i] = cell_ptc_ids
        else:
    
            grid_cell_ids[i] = i              
            grid_cell_prop[i] = False
            grid_cell_ptc_id[i] = None
                            
    # Assign the grid to grid_cell
    grid_cell = dict()
    grid_cell['ids'] = grid_cell_ids        
    grid_cell['prop'] = grid_cell_prop  
    grid_cell['ptc_ids'] = [grid_cell_ptc_id[ids] for ids in range(len(grid_cell_ptc_id))]          

    return grid_cell
#%%	Create 2D cell grid
def rectangle_grids(ptc, cell_size):

    
    '''This function is to subdive the space into 2D adjoin gird
    
    
    ptc = PTC[['x','y','z']].copy() 
    cell_size = 0.5
    '''
    # Initialisation
    
#    ptc = copy.deepcopy(ptc)
#    ptc_id = np.array(range(len(ptc))) #Test index from data frame

    # Compute a bounding box of a point cloud
    bounding_box = np.array([np.min(ptc, axis = 0), np.max(ptc, axis = 0)])

    # Computing a number of cells in x and y direction
    
    bounding_box_length = np.diff(bounding_box, n=1, axis = 0)
    num_cells = np.ceil(bounding_box_length/cell_size).astype(int)
    
    # Update a bounding box
    
    bounding_box_center = np.mean(bounding_box, axis = 0)
    bounding_box[0] = bounding_box_center - (num_cells*cell_size)/2
    bounding_box[1] = bounding_box_center + (num_cells*cell_size)/2
                
                                    
    # Coordinates of grid
    grid_coord = dict()
    grid_coord['x'] = np.linspace(bounding_box[0,0], bounding_box[1,0], num = num_cells[0,0]+1, endpoint = True)
    grid_coord['y'] = np.linspace(bounding_box[0,1], bounding_box[1,1], num = num_cells[0,1]+1, endpoint = True)

    # Create cell grid
    total_cells = num_cells[0,0]*num_cells[0,1]
    cell_ids = np.zeros((int(total_cells), 2), dtype = np.int32) # Cell id
    cell_prop = np.zeros((int(total_cells), 1), dtype = bool)
    cell_bounds = np.zeros((int(total_cells), 6), dtype = float)
    cell_ptc_ids = dict()
    

    # Process to create a rectangle grid
    for i in range(0, num_cells[0,0]):  # x direction

		# Extract the points within the grid_x
        if i+1 == num_cells[0,0]:
            
            grid_x_ptc_y = ptc.loc[(grid_coord['x'][i] <= ptc['x'])&(ptc['x'] <= grid_coord['x'][i+1]),['y']]
        else:
            
            grid_x_ptc_y = ptc.loc[(grid_coord['x'][i] <= ptc['x'])&(ptc['x'] < grid_coord['x'][i+1]),['y']]
        
        for j in range (0, int(num_cells[0,1])):  # y direction

#            lower_left = np.array([grid_coord['x'][i], grid_coord['y'][j]])
#            upper_right = np.array([grid_coord['x'][i+1], grid_coord['y'][j+1]])
#            
#            mask = np.all((lower_left <= ptc[['x', 'y']]) & (ptc[['x', 'y']] <= upper_right), axis = 1)
#            in_pt_ids = ptc.loc[mask].index
            # Determine 
            if j+1 == num_cells[0,0]:
                in_pt_ids = grid_x_ptc_y[(grid_coord['y'][j] <= grid_x_ptc_y['y'])&(grid_x_ptc_y['y'] <= grid_coord['y'][j+1])].index                 
            else:
                in_pt_ids = grid_x_ptc_y[(grid_coord['y'][j] <= grid_x_ptc_y['y'])&(grid_x_ptc_y['y'] < grid_coord['y'][j+1])].index                 
                  
    			# Assign cell attributes
            
            cell_order = int(i*num_cells[0,1] + j) 
            
            cell_ids[cell_order] = [i,j]
              
            if in_pt_ids.shape[0] >= 1:
                
                cell_prop[cell_order] = True
                z_min = np.min(ptc.loc[in_pt_ids, ['z']].values)
                z_max = np.max(ptc.loc[in_pt_ids, ['z']].values)
                             
#                cell_bounds[cell_order,:] = np.append(np.append(lower_left, z_min.values), np.append(upper_right, z_max.values))
                cell_ptc_ids[cell_order] = in_pt_ids
                            
#                ptc = ptc.drop(in_pt_ids, axis = 0)
            else:
                
                cell_prop[cell_order] = False
                z_min = bounding_box[0,2]
                z_max = bounding_box[1,2]
#                cell_bounds[cell_order,:] = np.append(np.append(lower_left, bounding_box[0,2]), np.append(upper_right, bounding_box[1,2]))
                cell_ptc_ids[cell_order] = None
                            
            # Update cell bounds
            cell_bounds[cell_order,:] = [grid_coord['x'][i], grid_coord['y'][j], z_min, grid_coord['x'][i+1], grid_coord['y'][j+1], z_max]
                       
    # Gathering all information
    cell = dict()
    cell['ids'] = cell_ids
    cell['prop'] = cell_prop
    cell['bounds'] = cell_bounds
    cell['ptc_ids'] = cell_ptc_ids
        
        
    # Return results
    return cell

# In[Create 2D grid in a cylinder coordinates]
def cylinder_grids(polar_ptc, angle_interval = np.pi/180, elevation_interval = 4.0*2.54/100):
    ''' This function is to generate cylinder grid


    polar_ptc = Shell_Cell_ptc[['z','theta']]
    angle_interval = np.pi/180
    elevation_interval = 4.0*2.54/100
    '''
    # Initialisation
    
    polar_ptc = copy.deepcopy(polar_ptc)
	

    # Compute the number of grids 
    elevation_num_cells = int(np.ceil((polar_ptc[['z']].max() - polar_ptc[['z']].min())/elevation_interval))
    angle_num_cells = int(np.ceil(2.0*np.pi/angle_interval))
                      
    # Coordinates of grid
    grid_coord = dict()
    grid_coord['z'] = np.linspace(polar_ptc[['z']].min(), polar_ptc[['z']].max(), num = elevation_num_cells+1, endpoint = True)
    grid_coord['theta'] = np.linspace(0, 2.0*np.pi, num = angle_num_cells+1, endpoint = True)

    # Create cell grid
    total_cells = elevation_num_cells*angle_num_cells
    cell_ids = np.zeros((int(total_cells), 2), dtype = np.int32) # Cell id
    cell_prop = np.zeros((int(total_cells), 1), dtype = bool)
    cell_bounds = np.zeros((int(total_cells), 4), dtype = float) #[z_min, theta_min, z_max, theta_max]
    cell_ptc_ids = dict()

    # Process to create a cylinder grid
    for i in range(0, elevation_num_cells):  # z direction
        # Extract point indices in a range of elevation
        if i == elevation_num_cells:
            mask = (grid_coord['z'][i] <=polar_ptc['z']) &(polar_ptc['z'] <= grid_coord['z'][i+1])
        
        else:
            mask = (grid_coord['z'][i] <=polar_ptc['z']) &(polar_ptc['z'] < grid_coord['z'][i+1])
        
        elevation_grid_ptc = polar_ptc.loc[mask,['theta']]
		
    
        for j in range (0, angle_num_cells):  # in polar
            # Extract points indices in a range of angle 
            if j == angle_num_cells:
                
                mask = (grid_coord['theta'][j] <=elevation_grid_ptc['theta']) &(elevation_grid_ptc['theta'] <= grid_coord['theta'][j+1])
            
            else:
                mask = (grid_coord['theta'][j] <=elevation_grid_ptc['theta']) &(elevation_grid_ptc['theta'] < grid_coord['theta'][j+1])
            
            elevation_angle_grid_ptc_ids = elevation_grid_ptc.loc[mask].index

    			# Assign cell attributes
            
            cell_order = int(i*angle_num_cells + j) 
            
            cell_ids[cell_order] = [i,j]
            cell_bounds[cell_order,:] = [grid_coord['z'][i], grid_coord['theta'][j], grid_coord['z'][i+1], grid_coord['theta'][j+1]]
              
            if elevation_angle_grid_ptc_ids.shape[0] >= 1:
                
                cell_prop[cell_order] = True              
                
                cell_ptc_ids[cell_order] = elevation_angle_grid_ptc_ids
                            
            else:
                cell_prop[cell_order] = False
                cell_ptc_ids[cell_order] = None
   
    # Gathering all information
    cylinder_cell = dict()
    cylinder_cell['ids'] = cell_ids
    cylinder_cell['prop'] = cell_prop
    cylinder_cell['bounds'] = cell_bounds
    cylinder_cell['ptc_ids'] = cell_ptc_ids
            
    # Return results
    return cylinder_cell

# In[Create grid in 2D]
def grid2D(ptc, major_dir = {'z': 2*2.54/100}, minor_dir = {'theta':1.0*np.pi/180.0}, min_ptc = 20):


    '''This function is to subdive the space into 2D adjoin gird


    ptc = Shell_ptc.copy()
    ptc = shell_wall_ptc.copy()
    major_dir = {'z': CELL_SIZE}
    minor_dir = {'theta':interval_angle}
    '''
    # Check input data
    if isinstance(ptc, pd.DataFrame):
        ptc_ids = ptc.index
#        ptc_columns = ptc.columns.astype(str)
        ptc = ptc.values
    else:
        ptc = np.asarray(ptc)
        ptc_ids = np.arange(ptc.shape[0])
#        ptc_columns = np.array(['x', 'y', 'z', 'rho', 'theta'])

    # Extract major and minor direction
    major_dir_key = list(major_dir.keys())[0]
    major_dir_val = major_dir[major_dir_key]
    minor_dir_key = list(minor_dir.keys())[0]
    minor_dir_val = minor_dir[minor_dir_key]
    
    # Check input array if the direction is rho or theta
    num_columns = ptc.shape[1]
    if ((major_dir_key.casefold() == "rho".casefold())|(minor_dir_key.casefold() == "theta".casefold()))&(num_columns <= 3):
        raise ValueError("Input data must be at least 4 columns: x, y, z, rho and the current data has only {:d} columns".format(num_columns))
    
    if ((major_dir_key.casefold() == "theta".casefold())|(minor_dir_key.casefold() == "theta".casefold()))&(num_columns <= 4):
        raise ValueError("Input data must be at least 5 columns: x, y, z, rho, theta and the current data has only {:d} columns".format(num_columns))
    
    if (((major_dir_key.casefold() == "rho".casefold())|(minor_dir_key.casefold() == "rho".casefold())|
        (major_dir_key.casefold() == "theta".casefold())|(minor_dir_key.casefold() == "theta".casefold()))&(num_columns <= 3)):
        raise ValueError("Input data must be 5 columns: x, y, z, rho, theta and the current data has only {:d} columns".format(num_columns))
    del num_columns

    # Check grid direction: major_direction
#    mask = major_dir_key.casefold() == ptc_columns
#    major_dir_id = np.asscalar(np.where(mask == True)[0])
#    major_ptc = ptc[:, major_dir_id]
    if major_dir_key.casefold() == "x".casefold():
        major_ptc = ptc[:, 0]
    elif major_dir_key.casefold() == "y".casefold():
        major_ptc = ptc[:, 1]
    elif major_dir_key.casefold() == "z".casefold():
        major_ptc = ptc[:, 2]
    elif major_dir_key.casefold() == "rho".casefold():
        major_ptc = ptc[:, 3]
    elif major_dir_key.casefold() == "theta".casefold():
        major_ptc = ptc[:, 4]
    else:
        raise ValueError('Input directional division {:s} does not support'.format(major_dir_key))

    # Check grid direction: minor_direction
#    mask = minor_dir_key.casefold() == ptc_columns
#    minor_dir_id = np.asscalar(np.where(mask == True)[0])
#    minor_ptc = ptc[:, minor_dir_id]
    
    if minor_dir_key.casefold() == "x".casefold():
        minor_ptc = ptc[:, 0]
    elif minor_dir_key.casefold() == "y".casefold():
        minor_ptc = ptc[:, 1]
    elif minor_dir_key.casefold() == "z".casefold():
        minor_ptc = ptc[:, 2]
    elif minor_dir_key.casefold() == "rho".casefold():
        minor_ptc = ptc[:, 3]
    elif minor_dir_key.casefold() == "theta".casefold():
        minor_ptc = ptc[:, 4]
    else:
        raise ValueError('Input directional division {:s} does not support'.format(minor_dir_key))

    #%% Calculate grid number
    major_dir_min_val, major_dir_max_val = np.min(major_ptc), np.max(major_ptc)
    minor_dir_min_val, minor_dir_max_val = np.min(minor_ptc), np.max(minor_ptc)
    
#    if major_dir_key.casefold() == 'theta'.casefold():
#        major_dir_min_val, major_dir_max_val = 0, 2.0*np.pi
#    else:
#        major_dir_min_val, major_dir_max_val = np.min(major_ptc), np.max(major_ptc)
#
#    if minor_dir_key.casefold() == 'theta'.casefold():
#        minor_dir_min_val, minor_dir_max_val = 0, 2.0*np.pi
#    else:
#        minor_dir_min_val, minor_dir_max_val = np.min(minor_ptc), np.max(minor_ptc)

    # Calculate the number of the grids
    major_dir_num_grids = np.ceil((major_dir_max_val - major_dir_min_val)/major_dir_val).astype(int)
    minor_dir_num_grids = np.ceil((minor_dir_max_val - minor_dir_min_val)/minor_dir_val).astype(int)

    # Generate mesh grids
    grids = dict()

    # Major
    grids[major_dir_key] = np.linspace(major_dir_min_val, major_dir_min_val + major_dir_val*(major_dir_num_grids), major_dir_num_grids + 1, dtype = np.float)
    if grids[major_dir_key][-1] != major_dir_max_val:
        grids[major_dir_key][-1] = major_dir_max_val

    grids[minor_dir_key] = np.linspace(minor_dir_min_val, minor_dir_min_val + minor_dir_val*(minor_dir_num_grids), minor_dir_num_grids + 1, dtype = np.float)
    if grids[minor_dir_key][-1] != minor_dir_max_val:
        grids[minor_dir_key][-1] = minor_dir_max_val

    # Create cell grid
    total_grids = major_dir_num_grids*minor_dir_num_grids
    grid_ids = np.zeros((total_grids, 2), dtype = np.int32) # Cell id
    grid_prop = np.zeros((total_grids, 1), dtype = bool)
    grid_bounds = np.zeros((total_grids, 4), dtype = float)

#    linked_list = np.zeros((ptc.shape[0], 2), dtype = np.uint64) #Connection grid id to point id: Check local index

    grid_ptc_ids = dict()

    #%% Process to create a rectangle grid
    for i in range(0, major_dir_num_grids):  # major direction

		# Extract the points within the grid_x
        if i+1 == major_dir_num_grids:
            mask = (grids[major_dir_key][i] <= major_ptc)&(major_ptc <= grids[major_dir_key][i+1])
        else:
            mask = (grids[major_dir_key][i] <= major_ptc)&(major_ptc < grids[major_dir_key][i+1])

        major_grid_ptc = minor_ptc[mask]
        major_grid_ptc_ids = ptc_ids[mask]

        for j in range (0, minor_dir_num_grids):  # a minor direction
            if j+1 == minor_dir_num_grids:
                mask = (grids[minor_dir_key][j] <= major_grid_ptc)&(major_grid_ptc <= grids[minor_dir_key][j+1])
            else:
                mask = (grids[minor_dir_key][j] <= major_grid_ptc)&(major_grid_ptc < grids[minor_dir_key][j+1])

            in_ptc_ids = major_grid_ptc_ids[mask]

    			# Assign cell attributes

            grid_order = i*minor_dir_num_grids + j
            grid_ids[grid_order] = [i,j]
            grid_bounds[grid_order,:] =[grids[major_dir_key][i], grids[minor_dir_key][j], grids[major_dir_key][i+1], grids[minor_dir_key][j+1]]

            if in_ptc_ids.shape[0] >= min_ptc:
                grid_prop[grid_order] = True
                grid_ptc_ids[grid_order] = in_ptc_ids
#                linked_list[in_ptc_ids, :] = [i,j]
            else:
                grid_prop[grid_order] = False

    # Gathering all information
    cell_grid_2D = dict()
    cell_grid_2D['ids'] = grid_ids
    cell_grid_2D['prop'] = grid_prop
    cell_grid_2D['bounds'] = grid_bounds
    cell_grid_2D['ptc_ids'] = grid_ptc_ids

    # Return results
    return cell_grid_2D, grids #, linked_list

# In[radial 2D grid for 2 data sets]
def radial_grid2D(sr_ptc, tr_ptc, major_dir = {'rho': 0.2}, minor_dir = {'theta':0.5*np.pi/180.0}):


    '''This function is to subdive the space into 2D adjoin gird


    sr_ptc = rafter_candidate_ptc.copy()
    tr_ptc = roof_ptc.copy()
    major_dir = major_dir = {'rho': 0.2}
    minor_dir = minor_dir = {'theta':0.5*np.pi/180.0}
    

    '''
    #% Check input source data
    if isinstance(sr_ptc, pd.DataFrame):
        sr_ptc_ids = sr_ptc.index
        sr_ptc_columns = sr_ptc.columns.astype(str)
        sr_ptc = sr_ptc.values
    else:
        sr_ptc = np.asarray(sr_ptc)
        sr_ptc_ids = np.arange(sr_ptc.shape[0])
        sr_ptc_columns = np.array(['x', 'y', 'z', 'rho', 'theta'])
    
    # Check input target data
    if tr_ptc is not None:
        if isinstance(tr_ptc, pd.DataFrame):
            tr_ptc_ids = tr_ptc.index
    #        tr_ptc_columns = tr_ptc.columns.astype(str)
            tr_ptc = tr_ptc.values
        else:
            tr_ptc = np.asarray(tr_ptc)
            tr_ptc_ids = np.arange(tr_ptc.shape[0])
    #        tr_ptc_columns = np.array(['x', 'y', 'z', 'rho', 'theta'])

    # Extract major and minor direction
    major_dir_key = list(major_dir.keys())[0]
    major_dir_val = major_dir[major_dir_key]
    minor_dir_key = list(minor_dir.keys())[0]
    minor_dir_val = minor_dir[minor_dir_key]

    # Check input array if the direction is rho or theta
    num_columns = sr_ptc.shape[1]
    if (((major_dir_key.casefold() == "rho".casefold())|(minor_dir_key.casefold() == "rho".casefold())|
        (major_dir_key.casefold() == "theta".casefold())|(minor_dir_key.casefold() == "theta".casefold()))&(num_columns <= 3)):

        raise ValueError("Input data must be 5 columns: x, y, z, rho, theta")
    del num_columns

    # Check grid direction: major_direction
    mask = major_dir_key.casefold() == sr_ptc_columns
    major_dir_id = np.asscalar(np.where(mask == True)[0])
    major_sr_ptc = sr_ptc[:, major_dir_id]
    if tr_ptc is not None: major_tr_ptc = tr_ptc[:, major_dir_id]
    # Check grid direction: minor_direction
    mask = minor_dir_key.casefold() == sr_ptc_columns
    minor_dir_id = np.asscalar(np.where(mask == True)[0])
    minor_sr_ptc = sr_ptc[:, minor_dir_id]
    if tr_ptc is not None: minor_tr_ptc = tr_ptc[:, minor_dir_id]
    
    #% Calculate grid number

    if major_dir_key.casefold() == 'theta'.casefold():
        major_dir_min_val, major_dir_max_val = 0, 2.0*np.pi
    else:
        major_dir_min_val, major_dir_max_val = np.min(major_sr_ptc), np.max(major_sr_ptc)

    if minor_dir_key.casefold() == 'theta'.casefold():
        minor_dir_min_val, minor_dir_max_val = 0, 2.0*np.pi
    else:
        minor_dir_min_val, minor_dir_max_val = np.min(minor_sr_ptc), np.max(minor_sr_ptc)

    # Calculate the number of the grids
    major_dir_num_grids = np.ceil((major_dir_max_val - major_dir_min_val)/major_dir_val).astype(int)
    minor_dir_num_grids = np.ceil((minor_dir_max_val - minor_dir_min_val)/minor_dir_val).astype(int)

    # Generate mesh grids
    grids = dict()

    # Major
    grids[major_dir_key] = np.linspace(major_dir_min_val, major_dir_min_val + major_dir_val*(major_dir_num_grids), major_dir_num_grids + 1, dtype = np.float)
    if grids[major_dir_key][-1] != major_dir_max_val:
        grids[major_dir_key][-1] = major_dir_max_val

    grids[minor_dir_key] = np.linspace(minor_dir_min_val, minor_dir_min_val + minor_dir_val*(minor_dir_num_grids), minor_dir_num_grids + 1, dtype = np.float)
    if grids[minor_dir_key][-1] != minor_dir_max_val:
        grids[minor_dir_key][-1] = minor_dir_max_val

    # Create cell grid
    total_grids = major_dir_num_grids*minor_dir_num_grids
    grid_ids = np.zeros((total_grids, 2), dtype = np.int32) # Cell id
    grid_bounds = np.zeros((total_grids, 4), dtype = np.float32)
    # Source
    grid_sr_prop = np.zeros((total_grids, 1), dtype = bool)
    # Target
    if tr_ptc is not None: grid_tr_prop = np.zeros((total_grids, 1), dtype = bool)
    
#    linked_list = np.zeros((ptc.shape[0], 2), dtype = np.uint64) #Connection grid id to point id: Check local index

    grid_sr_ptc_ids = dict()
    if tr_ptc is not None: grid_tr_ptc_ids = dict()

    #% Process to create a rectangle grid
    for i in range(0, major_dir_num_grids):  # major direction

		# Extract the points within the grid_x
        if i+1 == major_dir_num_grids:
            sr_mask = (grids[major_dir_key][i] <= major_sr_ptc)&(major_sr_ptc <= grids[major_dir_key][i+1])
        else:
            sr_mask = (grids[major_dir_key][i] <= major_sr_ptc)&(major_sr_ptc < grids[major_dir_key][i+1])

        minor_grid_sr_ptc = minor_sr_ptc[sr_mask]
        minor_grid_sr_ptc_ids = sr_ptc_ids[sr_mask]
        # Target
        if tr_ptc is not None:
            if i+1 == major_dir_num_grids:
                tr_mask = (grids[major_dir_key][i] <= major_tr_ptc)&(major_tr_ptc <= grids[major_dir_key][i+1])
            else:
                tr_mask = (grids[major_dir_key][i] <= major_tr_ptc)&(major_tr_ptc < grids[major_dir_key][i+1])
    
            minor_grid_tr_ptc = minor_tr_ptc[tr_mask]
            minor_grid_tr_ptc_ids = tr_ptc_ids[tr_mask]

        for j in range (0, minor_dir_num_grids):  # a minor direction
            # Source
            if j+1 == minor_dir_num_grids:
                sr_mask = (grids[minor_dir_key][j] <= minor_grid_sr_ptc)&(minor_grid_sr_ptc <= grids[minor_dir_key][j+1])
            else:
                sr_mask = (grids[minor_dir_key][j] <= minor_grid_sr_ptc)&(minor_grid_sr_ptc < grids[minor_dir_key][j+1])

            in_sr_ptc_ids = minor_grid_sr_ptc_ids[sr_mask]
            # Target
            if tr_ptc is not None:
                if j+1 == minor_dir_num_grids:
                    tr_mask = (grids[minor_dir_key][j] <= minor_grid_tr_ptc)&(minor_grid_tr_ptc <= grids[minor_dir_key][j+1])
                else:
                    tr_mask = (grids[minor_dir_key][j] <= minor_grid_tr_ptc)&(minor_grid_tr_ptc < grids[minor_dir_key][j+1])
    
                in_tr_ptc_ids = minor_grid_tr_ptc_ids[tr_mask]

    		# Assign cell attributes
            grid_order = i*minor_dir_num_grids + j
            grid_ids[grid_order] = [i,j]
            grid_bounds[grid_order,:] =[grids[major_dir_key][i], grids[minor_dir_key][j], grids[major_dir_key][i+1], grids[minor_dir_key][j+1]]
            # Source
            if in_sr_ptc_ids.shape[0] >= 1:
                grid_sr_prop[grid_order] = True
                grid_sr_ptc_ids[grid_order] = in_sr_ptc_ids
#                linked_list[in_ptc_ids, :] = [i,j]
            else:
                grid_sr_prop[grid_order] = False
                grid_sr_ptc_ids[grid_order] = None
            
            # Target
            if tr_ptc is not None:
                if in_tr_ptc_ids.shape[0] >= 1:
                    grid_tr_prop[grid_order] = True
                    grid_tr_ptc_ids[grid_order] = in_tr_ptc_ids
    #                linked_list[in_ptc_ids, :] = [i,j]
                else:
                    grid_tr_prop[grid_order] = False
                    grid_tr_ptc_ids[grid_order] = None

    # Gathering all information
    cell_grid_2D = dict()
    cell_grid_2D['ids'] = grid_ids
    cell_grid_2D['bounds'] = grid_bounds
    cell_grid_2D['prop'] = grid_sr_prop
    cell_grid_2D['ptc_ids'] = [grid_sr_ptc_ids[ids] for ids in range(len(grid_sr_ptc_ids))] 
    if tr_ptc is not None:
        cell_grid_2D['tr_prop'] = grid_tr_prop
        cell_grid_2D['tr_ptc_ids'] = [grid_tr_ptc_ids[ids] for ids in range(len(grid_tr_ptc_ids))] 

    #% Return results
    return cell_grid_2D, grids #, linked_list


# In[radial 2D grid for 2 data sets]
def radial_grid2D_2(ptc, major_dir = 'rho', minor_dir = 'theta', cell_size = 0.2):


    '''This function is to subdive the space into 2D adjoin gird


    ptc = roof_candidate_ptc.copy()
    major_dir = 'rho'
    minor_dir = 'theta'
    cell_size = 0.2
    

    '''
    #% Check input source data
    if isinstance(ptc, pd.DataFrame):
        ptc_ids = ptc.index
        ptc_columns = ptc.columns.astype(str)
        ptc = ptc.values
    else:
        ptc = np.asarray(ptc)
        ptc_ids = np.arange(ptc.shape[0])
        ptc_columns = np.array(['x', 'y', 'z', 'rho', 'theta'])

    # Check input array if the direction is rho or theta
    num_columns = ptc.shape[1]
    if (((major_dir.casefold() == "rho".casefold())|(minor_dir.casefold() == "rho".casefold())|
        (major_dir.casefold() == "theta".casefold())|(minor_dir.casefold() == "theta".casefold()))&(num_columns <= 3)):

        raise ValueError("Input data must be 5 columns: x, y, z, rho, theta")
    del num_columns

    # Check grid direction: major_direction
    mask = major_dir.casefold() == ptc_columns
    major_dir_id = np.asscalar(np.where(mask == True)[0])
    major_ptc = ptc[:, major_dir_id]
    # Check grid direction: minor_direction
    mask = minor_dir.casefold() == ptc_columns
    minor_dir_id = np.asscalar(np.where(mask == True)[0])
    minor_ptc = ptc[:, minor_dir_id]
    
    #% Calculate grid number

    if major_dir.casefold() == 'theta'.casefold():
        major_dir_min_val, major_dir_max_val = 0., 2.0*np.pi
    else:
        major_dir_min_val, major_dir_max_val = np.min(major_ptc), np.max(major_ptc)

    if minor_dir.casefold() == 'theta'.casefold():
        minor_dir_min_val, minor_dir_max_val = 0., 2.0*np.pi
    else:
        minor_dir_min_val, minor_dir_max_val = np.min(minor_ptc), np.max(minor_ptc)

    # Calculate the number of the grids
    major_dir_num_grids = np.ceil((major_dir_max_val - major_dir_min_val)/cell_size).astype(int)
#    minor_dir_num_grids = np.ceil((minor_dir_max_val - minor_dir_min_val)/minor_dir_val).astype(int)

    # Generate mesh grids
#    grids = dict()

    # Major
    major_grids = np.linspace(major_dir_min_val, major_dir_min_val + cell_size*(major_dir_num_grids), major_dir_num_grids + 1, dtype = np.float32)
    if major_grids[-1] != major_dir_max_val:
        major_grids[-1] = major_dir_max_val
        
    #%% Create cell grid

    grid_ids = []
    grid_bounds = []
    grid_prop = []
    
    
    grid_ptc_ids = dict()
   
    grid_order = 0
    #% Process to create a rectangle grid
    for i in range(0, major_dir_num_grids):  # major direction

		# Extract the points within the grid_x
        if i+1 == major_dir_num_grids:
            mask = (major_grids[i] <= major_ptc)&(major_ptc <= major_grids[i+1])
        else:
            mask = (major_grids[i] <= major_ptc)&(major_ptc < major_grids[i+1])

        minor_grid_ptc_ids = ptc_ids[mask]
        minor_grid_ptc = minor_ptc[mask]

        # Calculate the number of the grids in a minor dir
        if minor_dir.casefold() == 'theta'.casefold():
            major_grid_radius = (major_grids[i] + major_grids[i+1])/2.0
            minor_dir_num_grids = np.ceil(2.0*np.pi*major_grid_radius/cell_size).astype(np.int32)
            if minor_dir_num_grids < 4:
                minor_dir_num_grids = 4
            minor_grids = np.linspace(minor_dir_min_val, minor_dir_max_val, minor_dir_num_grids + 1, dtype = np.float32)
            if minor_grids[-1] != minor_dir_max_val:
                minor_grids[-1] = minor_dir_max_val

        for j in range (0, minor_dir_num_grids):  # a minor direction
            # Source
            if j+1 == minor_dir_num_grids:
                mask = (minor_grids[j] <= minor_grid_ptc)&(minor_grid_ptc <= minor_grids[j+1])
            else:
                mask = (minor_grids[j] <= minor_grid_ptc)&(minor_grid_ptc < minor_grids[j+1])

            in_ptc_ids = minor_grid_ptc_ids[mask]
            
    		# Assign cell attributes
            
            grid_ids = np.append(grid_ids, [i,j])
            grid_bounds = np.append(grid_bounds,[major_grids[i], minor_grids[j], major_grids[i+1], minor_grids[j+1]])
            # Source
            if in_ptc_ids.shape[0] >= 10:
                grid_prop = np.append(grid_prop, True)
                grid_ptc_ids[grid_order] = in_ptc_ids
#                linked_list[in_ptc_ids, :] = [i,j]
            else:
                grid_prop = np.append(grid_prop, False)
                grid_ptc_ids[grid_order] = None

            # Update the number of cells
            grid_order = grid_order + 1
            
    #%% Gathering all information
    grid_prop_boolean = np.full(len(grid_prop),False)
    grid_prop_boolean[grid_prop==1] = True
    
    cell_grid_2D = dict()
    cell_grid_2D['ids'] = np.asarray(grid_ids).reshape(-1,2)
    cell_grid_2D['bounds'] = np.asarray(grid_bounds).reshape(-1,4)
    cell_grid_2D['prop'] = np.asarray(grid_prop_boolean).reshape(-1,1)
    cell_grid_2D['ptc_ids'] = [grid_ptc_ids[ids] for ids in range(len(grid_ptc_ids))] 
    #%%% Return results
    return cell_grid_2D
# In[downsampling for plotting]
def rectangular_grid_downsampling(coord, val, xrange, yrange, xstep, ystep, reduction_type = 'mean'):
    """
    This function is to downsample the data in rectangular grid
    \ [input] coord                 Nx2 array represent coordinate of data
    \ [input] val                   N the corresponding value at each point
    \ [input] xrange                The range of the x-grid (actually only xrange[0] is used)
    \ [input] yrange                The range of the y-grid (actually only yrange[0] is used)
    \ [input] xstep                 Size of the cell in x-direction
    \ [input] ystep                 Size of the cell in y-direction
    \ [input] reduction_type        'count'     counting the number of element in the cell
                                    'max'       maximum value in the cell
                                    'min'       minimum value in the cell
                                    'mean'      average value in the cell
                                    'sum'       sum of the value in the cell
                                    'peak'      the peak value in the cell
    \ return
    """
    
    # indicate the cell each point belong to
    coord = copy.deepcopy(coord)
    point_to_cell = copy.deepcopy(coord)
    point_to_cell[:] = 0.0
    point_to_cell[:,0] = np.floor((coord[:,0] - xrange[0])/xstep)
    point_to_cell[:,1] = np.floor((coord[:,1] - yrange[0])/ystep)
    del coord
    
#    numCell_x = max(point_to_cell[:,0])
    numCell_y = np.max(point_to_cell[:,1])
    point_to_CellIdx = point_to_cell[:,0]*numCell_y + point_to_cell[:,1]
    point_to_CellIdx = point_to_CellIdx.astype(int)
    
    # sort all data according to point_to_CellIdx
    sorted_indices = np.argsort(point_to_CellIdx)
    if val is not None:
        sorted_val = val[sorted_indices]
    sorted_point_to_CellIdx = point_to_CellIdx[sorted_indices]
    unique, counts = np.unique(sorted_point_to_CellIdx, return_counts=True)
    stopIdx = np.cumsum(counts)
    startIdx = np.append([0],stopIdx[:-1])
    
    # compute subsampleCell_coord
    subsampleCell_X_Y = np.zeros((len(counts),2),dtype=int)
    subsampleCell_coord = np.zeros((len(counts),2),dtype=float)
    subsampleCell_X_Y[:,0] = np.floor(unique/numCell_y)
    subsampleCell_X_Y[:,1] = unique - subsampleCell_X_Y[:,0]*numCell_y
    subsampleCell_coord[:,0] = xrange[0] + subsampleCell_X_Y[:,0]*xstep + xstep/2.0
    subsampleCell_coord[:,1] = yrange[0] + subsampleCell_X_Y[:,1]*ystep + ystep/2.0
    del subsampleCell_X_Y, unique, sorted_point_to_CellIdx, sorted_indices
    
    # compute subsampleCell_val
    if val is not None:
        subsampleCell_val = np.zeros(len(counts),dtype=float)
        if reduction_type.casefold() == 'mean'.casefold():
            #subsampleCell_val = [np.sum(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            subsampleCell_val = np.add.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            subsampleCell_val = [subsampleCell_val[j]/counts[j] for j in range(0,len(counts))]
            #subsampleCell_val = np.divide(subsampleCell_val,counts)
        elif reduction_type.casefold() == 'sum'.casefold():
            #subsampleCell_val = [np.sum(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            subsampleCell_val = np.add.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
        elif reduction_type.casefold() == 'count'.casefold():
            subsampleCell_val[:] = counts[:]
            del sorted_val, stopIdx, startIdx
        elif reduction_type.casefold() == 'min'.casefold():
            subsampleCell_val = np.minimum.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            #subsampleCell_val = [np.min(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
        elif reduction_type.casefold() == 'max'.casefold():
            subsampleCell_val = np.maximum.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            #subsampleCell_val = [np.max(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
        elif reduction_type.casefold() == 'peak'.casefold():
            subsampleCell_val = [sorted_val[np.argmax(np.abs(sorted_val[beginIdx:stopIdx])+beginIdx)] for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            del sorted_val, stopIdx, startIdx
        else:
            raise ValueError('Error: value for reduction_type')
#            sys.exit()
    else:
        subsampleCell_val = 0.0
    
    return subsampleCell_coord, subsampleCell_val


        
def octree_grid_downsampling(coord, val, xrange, yrange, zrange, xstep, ystep, zstep, reduction_type = 'mean'):
    """
    This function is to downsample the data in rectangular grid
    \ [input] coord                 Nx2 array represent coordinate of data
    \ [input] val                   N the corresponding value at each point
    \ [input] xrange                The range of the x-grid (actually only xrange[0] is used)
    \ [input] yrange                The range of the y-grid (actually only yrange[0] is used)
    \ [input] zrange                The range of the z-grid (actually only zrange[0] is used)
    \ [input] xstep                 Size of the cell in x-direction
    \ [input] ystep                 Size of the cell in y-direction
    \ [input] zstep                 Size of the cell in z-direction
    \ [input] reduction_type        'count'     counting the number of element in the cell
                                    'max'       maximum value in the cell
                                    'min'       minimum value in the cell
                                    'mean'      average value in the cell
                                    'sum'       sum of the value in the cell
                                    'peak'      the peak value in the cell
    \ return
    """
    import sys
    
    # indicate the cell each point belong to
    coord = copy.deepcopy(coord)
    point_to_cell = np.zeros(coord.shape,dtype=int)
    point_to_cell[:] = 0.0
    point_to_cell[:,0] = np.floor((coord[:,0] - xrange[0])/xstep)
    point_to_cell[:,1] = np.floor((coord[:,1] - yrange[0])/ystep)
    point_to_cell[:,2] = np.floor((coord[:,2] - zrange[0])/zstep)
    np.savetxt('point_to_cell.txt', point_to_cell[:,:3], fmt='%d')
    del coord
    numCell_x = max(point_to_cell[:,0]) + 1.0
    numCell_y = max(point_to_cell[:,1]) + 1.0
    numCell_z = max(point_to_cell[:,2]) + 1.0
    point_to_CellIdx = point_to_cell[:,2]*numCell_x*numCell_y + point_to_cell[:,1]*numCell_x + point_to_cell[:,0]
    
    # sort all data according to point_to_CellIdx
    sorted_indices = np.argsort(point_to_CellIdx)
    if val is not None:
        sorted_val = val[sorted_indices]
    sorted_point_to_CellIdx = point_to_CellIdx[sorted_indices]
    unique, counts = np.unique(sorted_point_to_CellIdx, return_counts=True)
    stopIdx = np.cumsum(counts)
    startIdx = np.append([0],stopIdx[:-1])
    
    # compute subsampleCell_coord
    subsampleCell_X_Y_Z = np.zeros((len(counts),3),dtype=int)
    subsampleCell_coord = np.zeros((len(counts),3),dtype=float)
    subsampleCell_X_Y_Z[:,2] = np.floor(unique/(numCell_x*numCell_y))
    subsampleCell_X_Y_Z[:,1] = np.floor((unique - subsampleCell_X_Y_Z[:,2]*numCell_x*numCell_y)/numCell_x)
    subsampleCell_X_Y_Z[:,0] = unique - subsampleCell_X_Y_Z[:,2]*numCell_x*numCell_y - subsampleCell_X_Y_Z[:,1]*numCell_x
    subsampleCell_coord[:,2] = zrange[0] + subsampleCell_X_Y_Z[:,2]*zstep + zstep/2.0
    subsampleCell_coord[:,1] = yrange[0] + subsampleCell_X_Y_Z[:,1]*ystep + ystep/2.0
    subsampleCell_coord[:,0] = xrange[0] + subsampleCell_X_Y_Z[:,0]*xstep + xstep/2.0
    del subsampleCell_X_Y_Z, unique, sorted_point_to_CellIdx, sorted_indices
    
    # compute subsampleCell_val
    if val is not None:
        subsampleCell_val = np.zeros(len(counts),dtype=float)
        if reduction_type.casefold() == 'mean'.casefold():
            #subsampleCell_val = [np.sum(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            subsampleCell_val = np.add.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            subsampleCell_val = [subsampleCell_val[j]/counts[j] for j in range(0,len(counts))]
            #subsampleCell_val = np.divide(subsampleCell_val,counts)
        elif reduction_type.casefold() == 'sum'.casefold():
            #subsampleCell_val = [np.sum(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            subsampleCell_val = np.add.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
        elif reduction_type.casefold() == 'count'.casefold():
            subsampleCell_val[:] = counts[:]
            del sorted_val, stopIdx, startIdx
        elif reduction_type.casefold() == 'min'.casefold():
            subsampleCell_val = np.minimum.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            #subsampleCell_val = [np.min(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
        elif reduction_type.casefold() == 'max'.casefold():
            subsampleCell_val = np.maximum.reduceat(sorted_val,startIdx)
            del sorted_val, stopIdx, startIdx
            #subsampleCell_val = [np.max(sorted_val[beginIdx:stopIdx]) for beginIdx,stopIdx in zip(startIdx,stopIdx)]
        elif reduction_type.casefold() == 'peak'.casefold():
            subsampleCell_val = [sorted_val[np.argmax(np.abs(sorted_val[beginIdx:stopIdx])+beginIdx)] for beginIdx,stopIdx in zip(startIdx,stopIdx)]
            del sorted_val, stopIdx, startIdx
        else:
            print('Error: value for reduction_type')
            sys.exit()
    else:
        subsampleCell_val = 0.0
    
    return subsampleCell_coord, subsampleCell_val

