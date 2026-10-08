import os
import pickle
import gc
import numpy as np
from pathlib import Path

from utils.voxel_grid import voxel_grid, voxelgridmodify
from cylinder_extraction.estimate_cylinder import shell_wall_extraction

def foo(x,y):
    return x + y

def detect_objects(station, cylinder=True, plane=False):
# def extract_cylinders(station):
    print(f'parallel processing {station.name}')
    if station.ptc is None:
        station.read_points()
    # if isinstance(station, dict):
    #     # initialise a Station object from data file
    #     file_path = station['src_file']
    #     station_name = station['name']

    #     fn = Path(file_path).stem
    #     print(f'STATION {fn}')
    #     station = Station(
    #         name=station_name, file_path=file_path
    #     )
    try:
        out_dir = os.path.join(Path(station.file_path).parent, f'{station.name}')
        if cylinder:
            station.extract_cylinder(refine=True, output_dir=out_dir, write_file=True, write_parallel=False)
            # station.extract_cylinder_parallel(refine=True)
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

    # out_dir = os.path.join(data_dir,station.name + '-output')
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # # save station without points
    # station.ptc = None
    # out_fp = os.path.join(out_dir, f'{station.name}.pkl')
    # with open(out_fp, 'wb') as f:
    #     pickle.dump(station, f)


 
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
