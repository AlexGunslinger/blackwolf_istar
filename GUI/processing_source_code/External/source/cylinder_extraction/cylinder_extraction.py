# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:04:51 2021

@author: PrecisionAdmin
"""
import numpy as np
from itertools import chain
from .estimate_cylinder import shell_wall_extraction
from ..utils.voxel_grid import voxel_grid, voxelgridmodify

def xxx(cellgrid, threshold, constant):    
    # Classify the cell
    tank_cells_linear_ids, ground_cells_linear_ids = cells_classification(cellgrid, threshold)
    
    # Clusters the cells of the cyllinder
    clusters = cells_connectivity(cellgrid, tank_cells_linear_ids, threshold)
    
    #%% Estimate the clynder
    # Get the point clouds of the cluster
    cluster_id = 6
    clsuter_cell_linear_ids = clusters[cluster_id]
    clusters_pts_ids = list(chain.from_iterable(cellgrid.ptc_ids[voxel_id] for voxel_id in clsuter_cell_linear_ids))
    clusters_pts_ids = np.asarray(clusters_pts_ids, dtype=np.uint32)
    clusters_pts_xyz = cellgrid.points[clusters_pts_ids]
    
    # Create voxelgrid
    voxelgrid = voxel_grid(clusters_pts_xyz, threshold["cylinder_estimation"]['voxel_size'], threshold["cylinder_estimation"]['voxel_min_ptc'])
    voxelgrid.subdivision()
    voxelgrid = voxelgridmodify(None, None).voxelgrid_extraction(voxelgrid, voxel_prop = "full")
    
    # Extract points of the cylinders and cylider parameters
    
    
    
    shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras = shell_wall_extraction(voxelgrid, threshold["cylinder_estimation"], constant)


# In[Constant anf threshold]
# Constant


# Threshold





#% In[Generate the voxels]






# In[]
def cells_connectivity(cellgrid, cells_linear_ids, threshold):
    """This function is to cluster the cells based on its connectivity
    
    cells_linear_ids = tank_cells_linear_ids
    """
    # Call functions/libs
    import cv2
    # Get grid ids
    cell_grid_ids = cellgrid.ids[cells_linear_ids]
    # Get row and column
    rows = [np.min(cell_grid_ids[:, 0]), np.max(cell_grid_ids[:, 0])]
    cols = [np.min(cell_grid_ids[:, 1]), np.max(cell_grid_ids[:, 1])]
    num_row = rows[1] - rows[0] + 1
    num_col = cols[1] - cols[0] + 1
    
    # Create a grid local ids
    cell_grid_local_ids = cell_grid_ids - [rows[0], cols[0]]
    # Create initial array
    cellgrid_img = np.full((num_row, num_col), 0, dtype=np.int8)
    for cell_grid_local_id in cell_grid_local_ids:
        cellgrid_img[cell_grid_local_id[0], cell_grid_local_id[1]] = 1
    
    # Connected component labelling
    num_labels, labels = cv2.connectedComponents(cellgrid_img)

    cellgrid_img_labels = labels.copy()
    # Convert to 1d
    labels = labels.reshape(-1, 1)
    labels = labels[labels > 0]
    
    # Get cell grid of the clusters
    clusters = dict()
    label_ids = np.unique(labels)
    for count, label_id in enumerate(label_ids):
        # Get the points within the labels
        mask = np.in1d(labels, label_id)
        cluster_cell_linear_ids = cells_linear_ids[mask]
        
        # Update cluster
        if cluster_cell_linear_ids.shape[0] >= np.ceil(threshold["min_length"]/threshold["cell_size"]):
            clusters[label_id] = cluster_cell_linear_ids
    return clusters, cellgrid_img_labels

# In[]
def cells_classification(cellgrid, threshold):
    """This function is to classify the cell based on its height and gap
    """
    #% In[Calculate cell height
    cells_height = cellgrid.cal_cell_height(None, gap_threshold=threshold["point_gap"])

    # Cell classification based on height
    mask = (cells_height[:, 0] >= threshold["min_tank_height"]) & (cells_height[:, 1] >= 0.5*threshold["min_tank_height"])
    tank_cells_linear_ids = np.where(mask == True)[0]
    ground_cells_linear_ids = np.where(mask == False)[0]
    
    # return
    return tank_cells_linear_ids, ground_cells_linear_ids