import os
import gc
import numpy as np
from pathlib import Path
import logging
import sys

from ..utils.voxel_grid import voxel_grid, voxelgridmodify
from .estimate_cylinder import shell_wall_extraction

def foo(x,y):
    return x + y

def detect_objects(station, cylinder=True, plane=False, estimate_ground_z=True):
    """
    Designed as a unit function for running parallel. This functions extract interested objects from each station scan
    """
    # print(f'parallel processing {station.name}')
    # print(logging.getLogger().handlers)
    # load station points if not yet done
    if station.ptc is None:
        station.read_points()
    # main processing part
    if estimate_ground_z:
        station.estimate_ground_z()
    try:
        # for storing processing/extracting results
        file_path = Path(station.file_path)
        out_dir = file_path.parent / station.name
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        log_file = out_dir / 'log.txt'
        logger = logging.getLogger(station.name)
        fh = logging.FileHandler(str(log_file), mode='w')
        logger.addHandler(fh)
        # stdout_handler = logging.StreamHandler(sys.stdout)
        # logger.addHandler(stdout_handler)
        # return
        # logging.basicConfig(
        #     filename=str(log_file),
        #     filemode='w',
        #     level=logging.DEBUG
        # )
        # logging.info('extracting cylinders')
        if cylinder:
            logger.info('extracting cylinders')
            station.extract_cylinder(refine=True, output_dir=out_dir, write_file=True, write_parallel=False)
        if plane:
            station.extract_plane(write_file=False, output_dir=out_dir, num_point_min=300000)
            station.create_artificial_cylinders()

        del station.ptc
        del station.clusters

        gc.collect()

    except Exception as e:
        logger.error(f'error in {station.name}')
        logger.error(e)
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
    print(f'{len(shell_wall_pts_ids)} inlier points')
    if len(shell_wall_pts_ids) < threshold['cylinder_estimation']['min_num_ptc']:
        raise ValueError(f"{len(shell_wall_pts_ids)} inlier points do not meet the minimum {threshold['cylinder_estimation']['min_num_ptc']}")
    return shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras
