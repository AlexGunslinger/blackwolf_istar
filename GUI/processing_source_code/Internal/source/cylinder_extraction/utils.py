import os
import gc
import numpy as np
from pathlib import Path

from ..utils.voxel_grid import voxel_grid, voxelgridmodify
from ..cylinder_extraction.estimate_cylinder import shell_wall_extraction

from .estimate_cylinder import shell_wall_extraction
from ..utils.voxel_grid import voxel_grid, voxelgridmodify

def foo(x,y):
    return x + y

def detect_objects(station, cylinder=True, plane=False, estimate_ground_z=True):
    """
    Designed as a unit function for running parallel. This functions extract interested objects from each station scan
    """
    print(f'parallel processing {station.name}')
    # load station points if not yet done
    if station.ptc is None:
        station.read_points()
    # main processing part
    if estimate_ground_z:
        station.estimate_ground_z()
    try:
        # for storing processing/extracting results
        out_dir = os.path.join(Path(station.file_path).parent, f'{station.name}')
        if cylinder:
            station.extract_cylinder(refine=True, output_dir=out_dir, write_file=True, write_parallel=False)
        if plane:
            station.extract_plane(write_file=False, output_dir=out_dir, num_point_min=300000)
            station.create_artificial_cylinders()

        del station.ptc
        del station.clusters

        gc.collect()

    except Exception as e:
        print(f'error in {station.name}')
        print(e)
    return station
    
def fit_cylinder(clusters_pts_xyz, threshold):
    # Create voxelgrid
    voxelgrid = voxel_grid(
        clusters_pts_xyz, 
        threshold["cylinder_estimation"]['voxel_size'], 
        threshold["cylinder_estimation"]['voxel_min_ptc']
    )
    voxelgrid.subdivision()
    voxelgrid = voxelgridmodify(None, None).voxelgrid_extraction(voxelgrid, voxel_prop = "full")
    # Extract points of the cylinders and cylider parameters
    constant = {
        "ox": np.array([1., 0., 0.]),
        "oy": np.array([0., 1., 0.]),
        "oz": np.array([0., 0., 1.])
    }
    shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras = shell_wall_extraction(voxelgrid, threshold["cylinder_estimation"], constant)
    # print(f'{len(shell_wall_pts_ids)} inlier points')
    if len(shell_wall_pts_ids) < threshold['cylinder_estimation']['min_num_ptc']:
        raise ValueError(f"{len(shell_wall_pts_ids)} inlier points do not meet the minimum {threshold['cylinder_estimation']['min_num_ptc']}")
    return shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras

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