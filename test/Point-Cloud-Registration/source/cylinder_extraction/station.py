from pathlib import Path
import os
from time import time
import numpy as np
from itertools import chain
import pickle

from utils.read_pts import read_input_data
from utils.cell_grid import cell_grid, cell_grid_modify
from utils.write_output import write_array2txt
from utils.voxel_grid import voxel_grid, voxelgridmodify
from utils.utils import angle
from cylinder_extraction.config import threshold
from cylinder_extraction.estimate_cylinder import shell_wall_extraction
from cylinder_extraction.cylinder_extraction import cells_classification, cells_connectivity
from .cylinder import Cylinder

class Station():
    def __init__(self, name='station', ptc=None, file_path=None, 
    ptc_col_names=['x', 'y', 'z'], read_e57_raw=False,
    transformation=[], cylinders=[]):
        if file_path:
            self.file_path = file_path
            p = Path(file_path)
            # read point cloud
            print(p.parent, p.name)
            _ = read_input_data(p.parent, p.name, read_e57_raw=read_e57_raw)
            src_pts_xyz = _[ptc_col_names].values
            self.ptc = src_pts_xyz
        if ptc:
            self.ptc = ptc
        self.name = name
        self.transformation = transformation
        self.cylinders = cylinders
    
    def __repr__(self):
        return str({
            'name': self.name
        })
    
    def get_cylinder(self, cyl_id, refined=False):
        if refined:
            all_cyls = self.refined_cylinders
        else:
            all_cyls = self.cylinders
        for c in all_cyls:
            if c.id == cyl_id:
                return c
        return None
    
    def extract_cylinder(self, fit=True, output_dir=False, refine=False):
        # Voxel tree
        time_begin = time()
        cellgrid = cell_grid(self.ptc, threshold["cell_size"] , threshold["min_ptc"])
        cellgrid.subdivision()
        cellgrid = cell_grid_modify(None, None, None).cellgrid_extraction(cellgrid, cell_prop="Full")
        print('Time to generate a cell grid {:.03f} seconds'.format(time() - time_begin))
        # Remove ground points
        tank_cells_linear_ids, ground_cells_linear_ids = cells_classification(cellgrid, threshold)
        pts_cell_ids = []
        # for cell_id in range(cellgrid.ids.shape[0]):
        for cell_id in tank_cells_linear_ids:
            # Get points of cell
            cell_pts_ids = cellgrid.ptc_ids[cell_id]
            # Add cell_id
            cell_pts_ids = np.hstack(
                (cell_pts_ids.reshape(-1, 1), 
                np.full((cell_pts_ids.shape[0], 1), cell_id, dtype=np.uint32)))
            # Update cell_pts
            pts_cell_ids = np.append(pts_cell_ids, cell_pts_ids)
        # Convert to array
        pts_cell_ids = np.asarray(pts_cell_ids, dtype=np.uint32).reshape(-1, 2)
        # Objects clustering
        clusters, labels = cells_connectivity(cellgrid, tank_cells_linear_ids, threshold)
        self.clusters = []
        # extract points and fit cylinder for each cluster
        self.cylinders = []
        self.refined_cylinders = []
        if not output_dir:
            output_dir = os.path.join(Path(self.file_path).parent, f'{self.name}-output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for cluster_id in clusters:
            print(cluster_id)
            clsuter_cell_linear_ids = clusters[cluster_id]

            clusters_pts_ids = list(chain.from_iterable(
                cellgrid.ptc_ids[voxel_id] for voxel_id in clsuter_cell_linear_ids))
            clusters_pts_ids = np.asarray(clusters_pts_ids, dtype=np.uint32)
            clusters_pts_xyz = cellgrid.points[clusters_pts_ids]
            # write cluster points to file
            file_name = f"{self.name} cluster_{cluster_id}.txt"
            file_path = os.path.join(output_dir, file_name)
            # write_array2txt(file_path, cellgrid.points[clusters_pts_ids])
            # keep cluster information in station
            self.clusters.append({
                'id': cluster_id,
                'filepath': file_path,
                'points': cellgrid.points[clusters_pts_ids]
            })
            # fit cylinder from raw data
            if fit:
                try:
                    shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras = fit_cylinder(clusters_pts_xyz, threshold)
                    self.cylinders.append(Cylinder(
                        radius=cyl_paras[3],
                        position=cyl_paras[:3],
                        vector=cyl_paras[-3:],
                        metadata={
                            'station': self,
                            'id': cluster_id,
                            'ptc_filepath': file_path
                        }
                    ))
                    if refine:
                        _, _, _, refined_cyl_paras = fit_cylinder(shell_wall_pts_xyz, threshold)
                        self.refined_cylinders.append(Cylinder(
                            radius=refined_cyl_paras[3],
                            position=refined_cyl_paras[:3],
                            vector=refined_cyl_paras[-3:],
                            metadata={
                                'station': self,
                                'id': cluster_id,
                                'ptc_filepath': file_path
                            }
                        ))
                except Exception as e:
                    print(f'Error while fitting cylinder to cluster id={cluster_id}')
                    print(e)
            else:
                self.cylinders.append(Cylinder(
                    metadata={
                        'station': self,
                        'id': cluster_id,
                        'ptc_filepath': file_path
                    }
                ))
    def update_filepath(self, new_dir, num_keep_level=1):
        tails = []
        p = self.file_path
        for i in range(num_keep_level):
            parts = os.path.split(p)
            tails.insert(0, parts[-1])
            p = parts[0]
        self.file_path = os.path.join(new_dir, *tails)

    def write_points(self, filename=None):
        if not filename:
            filename = f'{self.name}.txt'
        print(f'write to {filename}')
        write_array2txt(filename, self.ptc)
    def read_points(self, read_e57_raw=False):
        p = Path(self.file_path)
        # read point cloud
        print(p.parent, p.name)
        _ = read_input_data(p.parent, p.name, read_e57_raw=read_e57_raw)
        ptc_col_names=['x', 'y', 'z']
        self.ptc = _[ptc_col_names].values

    def to_numpy(self, get_refined=False, top_surface=False):
        """This export cylinder data into a numpy array. Each row includes id, radius, xy intersect of the axis of the corresponding cylinder.
        """
        cyls = []
        if get_refined:
            atrb = 'refined_cylinders'
        else:
            atrb = 'cylinders'
        cylinders = getattr(self, atrb)
        if cylinders:
            for c in cylinders:
                if top_surface:
                    if not hasattr(c, 'top_surface'):
                        c.estimate_top_surface()
                    p = list(c.top_surface['center'])
                else:
                    p = c.axis.intersect_xy()
                cyls.append([
                    c.id,
                    c.radius,
                    *p
                ])
            return np.array(cyls)
            # except Exception:
            #     raise ValueError('Not enough cylinder information')

def fit_cylinder(clusters_pts_xyz, threshold):
    # Create voxelgrid
    voxelgrid = voxel_grid(clusters_pts_xyz, threshold["cylinder_estimation"]['voxel_size'], threshold["cylinder_estimation"]['voxel_min_ptc'])
    voxelgrid.subdivision()
    voxelgrid = voxelgridmodify(None, None).voxelgrid_extraction(voxelgrid, voxel_prop = "full")
    # Extract points of the cylinders and cylider parameters
    constant = {
        "ox": np.array([1., 0., 0.]),
        "oy": np.array([0., 1., 0.]),
        "oz": np.array([0., 0., 1.])
    }
    shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras = shell_wall_extraction(voxelgrid, threshold["cylinder_estimation"], constant)

    return shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras