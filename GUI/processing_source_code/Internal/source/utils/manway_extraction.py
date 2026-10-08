# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Linh Truong-Hong)s
"""
from itertools import chain
import numpy as np
# from .write_results import write_array2txt
# from utils_data.utils_data import get_pts_xyz
from .utils_geometry import leastsq_fit_circle
from .utils_geometry import cart2pol, pol2cart
# from segmentations.segmentation_cells import cell_connectivity
# from segmentations.utils_cluster import get_region_pts_ids
from .radialcellgrid import circular_cellgrid
import pandas as pd
import cv2

def manway_extraction(
    ptc, shellwall_ptc_ids=None, shellwall_fit_circle=None, manway_size=[0.2, 1.0],
    cell_size=None):
    """
    The function is to detect maway and fit a circle
    Parameters
    ----------
    ptc                     : [N x 3 and other] [ x, y, z, other] - data points
    shellwall_ptc_ids       : [M x 1] - pts_ids of the shellwall respect to ptc
    shellwall_fit_circle    : [1 x 4] [xc, yc, zc, rc] - a fitting circle of the shellwall
    manway_size             : [1 x 2] [ min_radius, max_radius] of the manway want to detect. The default is [0.2, 1.0].

    Returns
    -------
    manway                  : Dictionary
        [ptc_ids]           : [N x 1] pts_ids of the boundary points respect with ptc
        [fit_circle]        : [1 x 4] [xc, yc, zc, rc] - a fitting circle of the manway
        DESCRIPTION.
    """
    # Constant
    BUFFER_SIZE = 1.0 #
    # Check input variables and initialisation
    pts_xyz = get_pts_xyz(ptc)
    if shellwall_ptc_ids is None:
        shellwall_pts_xyz = pts_xyz
    else:
        shellwall_pts_xyz = pts_xyz[shellwall_ptc_ids]
    
    # Determine cell size for manway detection
    if manway_size is None:
        manway_size = [0.4, 1.0]
    # Fit circle
    if shellwall_fit_circle is None:
        xc, yc, rc, __, __, __ = leastsq_fit_circle(shellwall_pts_xyz[:, 0], shellwall_pts_xyz[:, 1])
    else:
        (xc, yc, rc) = shellwall_fit_circle
        
    # Generate roll out
    shellwall_pts_rho, shellwall_pts_theta = cart2pol(shellwall_pts_xyz[:, 0] - xc, shellwall_pts_xyz[:, 1] - yc)
    shellwall_pts_xyzrtheta = np.hstack((shellwall_pts_xyz, shellwall_pts_rho.reshape(-1,1), shellwall_pts_theta.reshape(-1,1)))
    del shellwall_pts_xyz, shellwall_pts_rho, shellwall_pts_theta
    
    # Compute the cell size
    if cell_size is None:
        cell_height = np.min(manway_size)/2.5
    else:
        cell_height = cell_size
    angle_interval = cell_height/rc
    # Compute the number of the cells representing the manway
    cell_area = cell_height**2
    manway_num_cells = np.floor([np.pi*(radius**2)/cell_area for radius in manway_size]).astype(int) # The largest square inside the circle rc/sqrt(2)/cell_size     
    
    # Generate 2D grid based on a theta and z    
    cellgrids = circular_cellgrid(shellwall_pts_xyzrtheta[:, [4, 2]], cell_size=[angle_interval, cell_height], min_ptc=2, divide_option="even")
    cellgrids.subdivision()
    
    #% Get empty cells
    mask = cellgrids.prop==False
    emptycell_linear_ids = np.where(mask == True)[0]
    
    # Cluster the emptycell
    emptycell_region = cell_connectivity(
        cellgrids, 
        emptycell_linear_ids,
        bound=[
            int(manway_size[0]/cell_height)+1,
            int(manway_size[1]/cell_height)+1
        ]
    )
    region_linear_ids, emptycell_ids, emptycell_bound_ids = get_region_pts_ids(emptycell_region)
    
    # Filter the manway
    manway = dict()
    manway_id = 1

    all_hole_region_points = np.empty((0,3))
    for count, region_linear_id in enumerate(region_linear_ids):
        # Get cells of the region
        region_emptycell_ids = emptycell_ids[emptycell_bound_ids[count]:emptycell_bound_ids[count + 1]]
        
        # Check the region
        ismanway = (np.min(np.sqrt(manway_num_cells)) <= region_emptycell_ids.shape[0]) & (region_emptycell_ids.shape[0] <= manway_num_cells[1])
        if not ismanway: continue
    
        # Get bounds of the region
        region_bounds = cellgrids.bounds[region_emptycell_ids].reshape(-1, 2)
        region_min_bounds, region_max_bounds = np.min(region_bounds, axis=0), np.max(region_bounds, axis=0)
        
        # Get cells within the buffer
        mask = np.all((region_min_bounds - [BUFFER_SIZE/rc, BUFFER_SIZE] <= cellgrids.bounds[:, :2]) & (cellgrids.bounds[:, 2:] <= region_max_bounds + [BUFFER_SIZE/rc, BUFFER_SIZE]), axis=1)
        buffer_region_allcells_ids = np.where(mask == True)[0]
        buffer_region_fullcells_ids = buffer_region_allcells_ids[cellgrids.prop[buffer_region_allcells_ids]]
        del buffer_region_allcells_ids
        # Search the full cell connect to the empty cells
        # Layer 1
        region_fullcells_ids = []
        for region_emptycell_id in region_emptycell_ids:
            __, region_fullcell_ids, __ = cellgrids.searching_adjoin_cells(region_emptycell_id, buffer_region_fullcells_ids, index_type="linear", searching_opt='xy', cell_prop="full")
            region_fullcells_ids = np.append(region_fullcells_ids, region_fullcell_ids)
        # Convert to array
        region_fullcells_ids = np.asarray(region_fullcells_ids, dtype=np.uint32)
        #Layer 2

        for region_fullcells_id in region_fullcells_ids:
            __, region_fullcell_ids, __ = cellgrids.searching_adjoin_cells(region_fullcells_id, buffer_region_fullcells_ids, index_type="linear", searching_opt='xy', cell_prop="full")
            region_fullcells_ids = np.append(region_fullcells_ids, region_fullcell_ids)
        # Convert to array
        region_fullcells_ids = np.unique(np.asarray(region_fullcells_ids, dtype=np.uint32))
        
        # Get the the points within the buffer
        buffer_region_fullcells_pts_ids = list(chain.from_iterable(cellgrids.ptc_ids[cell_id] for cell_id in buffer_region_fullcells_ids))
        buffer_region_fullcells_pts_ids = np.asarray(buffer_region_fullcells_pts_ids, dtype=np.uint32)
        buffer_region_fullcells_pts_xyzrtheta = shellwall_pts_xyzrtheta[buffer_region_fullcells_pts_ids]

        # Get the points on the boundary of the region
        region_fullcells_pts_ids = list(chain.from_iterable(cellgrids.ptc_ids[cell_id] for cell_id in region_fullcells_ids))
        region_fullcells_pts_ids = np.asarray(region_fullcells_pts_ids, dtype=np.uint32)
        region_fullcells_pts_xyzrtheta = shellwall_pts_xyzrtheta[region_fullcells_pts_ids]
        
        # save
        all_hole_region_points = np.r_[all_hole_region_points, region_fullcells_pts_xyzrtheta[:,:3]]

        # Find the boundary points
        buffer_region_fullcells_pts_lz = np.c_[buffer_region_fullcells_pts_xyzrtheta[:, 4]*rc, buffer_region_fullcells_pts_xyzrtheta[:, 2]]
        region_fullcells_pts_lz = np.c_[region_fullcells_pts_xyzrtheta[:, 4]*rc, region_fullcells_pts_xyzrtheta[:, 2]]
        boundary_pts_local_ids = angle_criterion_boundary_points(buffer_region_fullcells_pts_lz, region_fullcells_pts_lz)
        boundary_pts_ids = region_fullcells_pts_ids[boundary_pts_local_ids]
        boundary_pts_lz = region_fullcells_pts_lz[boundary_pts_local_ids]
        
        # Fit the circle
        region_lc, region_zc, region_rc, __, __, __ = leastsq_fit_circle(boundary_pts_lz[:, 0], boundary_pts_lz[:, 1])
        # Convert to xy
        region_theta_c = region_lc/rc 
        region_local_xc, region_local_yc = pol2cart(rc, region_theta_c)
        region_xc, region_yc = region_local_xc + xc, region_local_yc + yc
        region_fit_circle = np.array([region_xc, region_yc, region_zc, region_rc])
        
        # Update manway     
        manway[manway_id] = {"ptc_ids": shellwall_ptc_ids[boundary_pts_ids],
                             "fit_circle": region_fit_circle}
        manway_id += 1
    # Return
    return manway, all_hole_region_points
    
# In[]
def angle_criterion_boundary_points(src_pts_xy, trg_pts_xy):
    """
    The function is to extract the boundary points based on the angle criterion 
    Parameters
    ----------
    src_pts_xy          : [N x 2] [x, y] of the points defined neighbourhood. Note they are co-planar
    trg_pts_xy          : [M x 2] [x, y] of the points to extract the boundary points. Note they are co-planar
        
    Returns
    -------
    boundary_pts_local_ids : [N x 1] pts_ids of the boundary points respect to trg_pts_xy

    """

    # src_pts_xyz = buffer_region_fullcells_pts_xyzrtheta[:, :3]
    # trg_pts_xyz = region_fullcells_pts_xyzrtheta[:, :3]       
    
    # Call libs
    from sklearn.neighbors import KDTree
    # Constant
    num_kNN = 21
    ANGLE_THRESHOLD = np.pi/2.0
    # Create tree
    tree = KDTree(src_pts_xy, leaf_size=1000)
    __, tree_neighbour_pts_ids = tree.query(trg_pts_xy, k=num_kNN)
    
    # Find the boundary points
    flag_boundary_pts = np.full(trg_pts_xy.shape[0], False, dtype=bool)
    for count, neighbour_pts_ids in enumerate(tree_neighbour_pts_ids):
        # Get the points
        pt_p = trg_pts_xy[count]
        pt_q = src_pts_xy[neighbour_pts_ids[1:]]
        # Convert to polar coordinate
        __, pt_theta = cart2pol(pt_q[:, 0] - pt_p[0], pt_q[:, 1] - pt_p[1])
        # Find the boundary points
        pt_theta = np.sort(pt_theta)
        pt_theta = np.insert(pt_theta, pt_theta.shape[0], pt_theta[0] + 2.0*np.pi)
        if np.max(np.diff(pt_theta)) >= ANGLE_THRESHOLD:
            flag_boundary_pts[count] = True
    # Get the boundary points
    boundary_pts_local_ids = np.where(flag_boundary_pts == True)[0]
    # Return
    return boundary_pts_local_ids
    
        
def get_region_pts_ids(result_region):
    """This function is to create lower and upper bound indices of the region
    Parameters:
    ------------------------
        result_region                   : [N x 2] [pts_id, region_id]
        
    Return:
    ------------------------
        region_linear_ids               : region_ids
        pts_ids                         : new order of pts_ids
        pts_bound_ids                   : lower and upper bound pts_ids of the re respect pts_ids
    Demo:
    ------------------------
    pts_ids, region_ids = result_region[:, 0], result_region[:, 1]
    """
    # extract pts and region ids
    pts_ids, region_ids = result_region[:, 0], result_region[:, 1]
    # create a local indices
    sort_ids = np.argsort(region_ids)
    region_ids = region_ids[sort_ids]
    pts_ids = pts_ids[sort_ids]
    region_linear_ids, count_pts_in_region = np.unique(region_ids, return_counts=True)
    cum_sum = np.cumsum(count_pts_in_region)
    pts_bound_ids = np.append([0], cum_sum) #0 a linear indicies of points within the region
    
    # return
    return region_linear_ids, pts_ids, pts_bound_ids

def get_pts_xyz(pts):
    """
    This function is to get x, y, z coordinates of points
    """

    # Initialization
    if isinstance(pts, pd.DataFrame):
        if any(item in ['x','y','z'] for item in pts.columns):
            pts_xyz = pts[['x','y','z']].values
        else:
            raise ValueError('The input data does not contain x, y, z coordinates')
    elif isinstance(pts, np.ndarray):
        pts_xyz = pts[:, :3]
    else:
        raise ValueError('The input variable ptc does not support in this function')
    # return
    return pts_xyz

# In[]
def cell_connectivity(cellgrids, cell_linear_ids, bound=None):
    """This function is to group the cell connected 
    Parameters
    ----------
        cellgrids                   : [N x 3] 
        cell_linear_ids             : dictionary of a 2D cell grid

    Returns
    -------
        cell_region                 : [cell_ids, region_ids]
    
    Demo:
        
        cell_linear_ids = emptycell_linear_ids
    """
    # Check input data
    if cell_linear_ids is None:
        cell_linear_ids = np.arange(cellgrids.ids.shape[0], dtype=np.uint32)
    
    # Convert to images
    cell_grid_ids = cellgrids.ids[cell_linear_ids]
    cellgrid_img = np.zeros((np.max(cell_grid_ids[:, 0]) + 1, np.max(cell_grid_ids[:, 1]) + 1), dtype=np.uint8) 
    cellgrid_img[cell_grid_ids[:, 0], cell_grid_ids[:, 1]] = 1#cellgrids.prop[cell_linear_ids]
    
    # Connected component labelling
    num_labels, labels = cv2.connectedComponents(cellgrid_img)
    # filter components
    if bound is not None:
        for i in range(1, num_labels + 1):
            cnn = labels == i
            # horizontal/vertical extent needs to be within bound
            horizontal_extent = np.sum(np.any(cnn, axis=0))
            if horizontal_extent < bound[0] or horizontal_extent > bound[1]:
                cellgrid_img[cnn] = 0
            vertical_extent = np.sum(np.any(cnn, axis=1))
            if vertical_extent < bound[0] or vertical_extent > bound[1]:
                cellgrid_img[cnn] = 0
                
    # Connected component labelling
    num_labels, labels = cv2.connectedComponents(cellgrid_img)
    # Convert to 1d
    cell_labels = labels[cell_grid_ids[:, 0], cell_grid_ids[:, 1]]
    # Check if the features to be updated or not
    cell_region = np.vstack((cell_linear_ids, cell_labels)).T
    
    # Update the region
    cell_region = update_regions_ids(cell_region)
    
    # Return
    return cell_region

def update_regions_ids(result_region):
    """This function is to update the new region ids where the region ids were sorted
    Input:
        result_region        : [N x 2] [data_id, region id]
    Output:
        result_region        : [N x 2] [data_id, new region id]
    Demo:
        
    """
    # Get the number of members
    region_stat = regions_statistic(result_region)
    # Get the new region
    new_region_ids = np.zeros_like(result_region[:, 0], dtype=np.uint32)
    for count, region_id in enumerate(region_stat[:, 0]):
        mask = np.in1d(result_region[:, 1], region_id)
        new_region_ids[mask] = count + 1
    # Update new region
    result_region[:, 1] = new_region_ids
    return result_region

def regions_statistic(region_info):
    """This function is to count the member of the region
    Input:
        region_info         : [N x 2] [data_id, region id]
    Output:
        region_stat         : [N x 2] [region id, number of members of the region]
    Demo:
        region_info = result_region
    """
    # Sort region
    region_count = np.bincount(region_info[:, 1])
    region_id = np.nonzero(region_count)[0]
    region_stat = np.vstack((region_id, region_count[region_id])).T #[region_id, num_points_in_region]
    # Sort the region
    mask = np.argsort(region_stat[:, 1])[::-1]
    region_stat = region_stat[mask]
    return region_stat    