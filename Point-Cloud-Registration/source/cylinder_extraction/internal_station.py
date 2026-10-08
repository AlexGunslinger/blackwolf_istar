import pickle
from cv2 import threshold
import numpy as np
import os
from pathlib import Path
from time import time
from itertools import chain
import open3d as o3d
import logging
from pathlib import Path

from scipy.signal import find_peaks
from scipy.stats import binned_statistic

from ..cylinder_extraction.station import Station
from ..cylinder_extraction.cylinder import Cylinder
from ..cylinder_extraction.config import threshold
from ..cylinder_extraction.utils import cells_classification, cells_connectivity

from ..utils.cell_grid import cell_grid, cell_grid_modify
from ..utils.write_output import write_array2txt, write_array2pcd

from ..utils.utilities import cart2pol
from ..utils.manway_extraction import manway_extraction

class InternalStation(Station):
    def __init__(self, name='station', ptc=None, file_path=None, read_points=True, 
        ptc_col_names=['x', 'y', 'z'], read_e57_raw=False, 
        transformation=[], cylinders=[], station_id=0, floating_roof=False):
        if file_path:
            if isinstance(file_path, str):
                file_path = Path(file_path)
        super().__init__(name, ptc, file_path, read_points, ptc_col_names, read_e57_raw, transformation, cylinders, station_id)
        self.align = {
            'rotation': np.identity(3),
            'translation': np.zeros((3,1))
        }
        self.floating_roof = floating_roof
    
    def save_object(self, write_dir=None):
        self.ptc = None
        if hasattr(self, 'shell'):
            self.shell.points = None
            self.shell.kdtree = None
        if not write_dir:
            write_file = self.file_path.parent / self.file_path.stem / f'{self.name}.pkl'
        else:
            write_file = Path(write_dir) / f'{self.name}.pkl'
        with open(write_file, 'wb') as f:
            pickle.dump(self, f)

    def estimate_ground_z(self, method='voxelize', cell_params={'cell_size': 0.5, 'min_ptc': 10}):
        """
        determinte the z coordinate of ground
        """
        if method == 'voxelize':
            return self._estimate_ground_z_by_voxelize(cell_params)
        else:
            # two step
            # coarse estimate: every 0.5 meter
            step_rough = 0.5
            z_max = self.ptc[:,2].max()
            z_min = self.ptc[:,2].min()
            hist, bin_edges = np.histogram(self.ptc, bins=np.arange(z_min, z_max, step_rough))
            if self.floating_roof:
                # both the bottom and the top surfices have the most number of points
                # find z coordinates of these 2 surfaces
                # first, add 0s to the begin and the end of hist in case there are peaks at these bounds
                hist = np.r_[0, hist, 0]
                peak_ind, _ = find_peaks(hist, distance=2)
                # the corresponding peaks
                peaks = hist[peak_ind]
                # only keep the two highest peaks
                order = np.argsort(peaks)
                floor_rough_ind = peak_ind[order[-2:].min()]-1
            else:
                # the ground is determined with respect to the bin with the most number of points, i.e. z
                floor_rough_ind = np.argmax(hist)
            # fine estimate
            step_fine = 0.02
            z_min = bin_edges[floor_rough_ind]
            z_max = bin_edges[floor_rough_ind + 1]
            hist, bin_edges = np.histogram(self.ptc, bins=np.arange(z_min, z_max, step_fine))
            # the ground is determined with respect to the bin with the most number of points, i.e. z
            z_ground = bin_edges[np.argmax(hist)]

            self.z_ground = z_ground
            return z_ground
    
    def _estimate_ground_z_by_voxelize(self, cell_params):
        time_begin = time()
        threshold.update(cell_params)
        cellgrid = cell_grid(self.ptc, threshold["cell_size"] , threshold["min_ptc"])
        cellgrid.subdivision()
        cellgrid = cell_grid_modify(None, None, None).cellgrid_extraction(cellgrid, cell_prop="Full")
        print('Time for generating a cell grid {:.03f} seconds'.format(time() - time_begin))
        # wall/floor points
        _, floor_cells_linear_ids = cells_classification(cellgrid, threshold)
        # in each cell, take the minimum height as a potential floor height
        floor = np.array([np.min(self.ptc[cellgrid.ptc_ids[ci],2]) for ci in floor_cells_linear_ids])
        # 
        z_max = floor.max()
        z_min = floor.min()
        # to avoid cases that a cell has roof but no floor points
        # first find the range with the most floor points
        if z_max - z_min > 0.2:
            step = 0.2
            hist, bin_edges = np.histogram(floor, bins=np.arange(z_min, z_max, step))
            z_min = bin_edges[np.argmax(hist)]
            z_max = bin_edges[np.argmax(hist) + 1]
        step = 0.02
        hist, bin_edges = np.histogram(self.ptc, bins=np.arange(z_min, z_max, step))
        # the ground is determined with respect to the bin with the most number of points, i.e. z
        z_ground = bin_edges[np.argmax(hist)]

        self.z_ground = z_ground
        return z_ground

    def estimate_scanner_position(self, cell_params={'cell_size': 0.5, 'min_ptc': 10}):
        time_begin = time()
        threshold.update(cell_params)
        self.read_points()
        cellgrid = cell_grid(self.ptc, threshold["cell_size"] , threshold["min_ptc"])
        cellgrid.subdivision()
        cellgrid = cell_grid_modify(None, None, None).cellgrid_extraction(cellgrid, cell_prop="Full")
        print('Time for generating a cell grid {:.03f} seconds'.format(time() - time_begin))
        # wall/floor points
        _, floor_cells_linear_ids = cells_classification(cellgrid, threshold)
        # calculate the number of points in each cell
        point_counts = np.array([cellgrid.ptc_ids[ci].shape[0] for ci in floor_cells_linear_ids])
        # cell with the largest number of points
        ind_max = np.argmax(point_counts)
        # use the center of all points in the most dense cell as the position of scanner
        return self.ptc[cellgrid.ptc_ids[floor_cells_linear_ids[ind_max]]].mean(axis=0)

    def clustering(self, output_dir=None, write_file=True, cluster_params={}):
        # clusters an internal scan into separated groups of points
        # Voxel tree
        time_begin = time()
        threshold.update(cluster_params)
        cellgrid = cell_grid(self.ptc, threshold["cell_size"] , threshold["min_ptc"])
        cellgrid.subdivision()
        cellgrid = cell_grid_modify(None, None, None).cellgrid_extraction(cellgrid, cell_prop="Full")
        print('Time for generating a cell grid {:.03f} seconds'.format(time() - time_begin))
        # wall/floor points
        tank_cells_linear_ids, _ = cells_classification(cellgrid, threshold)
        # wall
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
        clusters, _ = cells_connectivity(cellgrid, tank_cells_linear_ids, threshold)
        self.clusters = []
        # extract points and fit cylinder for each cluster
        self.cylinders = []
        self.refined_cylinders = []
        if not output_dir:
            output_dir = os.path.join(Path(self.file_path).parent, f'{self.name}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # self.poi = np.empty([1,3])
        self.num_clusters = len(clusters)

        for cluster_id in clusters:
            clsuter_cell_linear_ids = clusters[cluster_id]

            clusters_pts_ids = list(chain.from_iterable(
                cellgrid.ptc_ids[voxel_id] for voxel_id in clsuter_cell_linear_ids))
            clusters_pts_ids = np.asarray(clusters_pts_ids, dtype=np.uint32)
            clusters_pts_xyz = cellgrid.points[clusters_pts_ids]
            # write cluster points to file
            file_name = f"{self.name} cluster_{cluster_id}.pcd"
            file_path = os.path.join(output_dir, file_name)
            if write_file:
                write_array2pcd(file_path, clusters_pts_xyz)

            # self.poi = np.r_[self.poi, clusters_pts_xyz]
            # fit cylinder from raw data
            cluster_type = 'unknown'
            # extend points (to include surrounding ground)
            ext_points = None
            
            clst = {
                'id': cluster_id,
                'filepath': file_path,
                'type': cluster_type,
                'point_indices': clusters_pts_ids,
                'num_points': clusters_pts_ids.size,
                'num_cell': clsuter_cell_linear_ids.size
            }
            self.clusters.append(clst)

    def identify_shell_cluster(self, indicator='num_cell', fit=True, quick_fit=False):
        # number of points in all clusters
        counts = [clst[indicator] for clst in self.clusters]
        ind = np.argmax(np.array(counts))
        self.shell_cluster = self.clusters[ind]
        print(f"shell wall {self.name} cluster {self.shell_cluster['id']}")
        # fit cylinder
        if fit:
            self.shell = Cylinder(
                id=self.shell_cluster['id'],
                station=self,
                ptc_filepath=self.shell_cluster['filepath'],
                cluster_id=self.shell_cluster['id']
            )
            self.shell.read_points()

            if quick_fit:
                self.shell.quick_fit(z_range=[
                    self.shell.points[:,2].min(),
                    self.shell.points[:,2].max()
                ], z_cuts=[0.4, 0.6])
            else:
                fit_info = self.shell.fit(refined=True, update=True, ransac_params={'max_dist': 0.2}, ignore_number_of_points=True)
                # shell_pts_ind = fit_info['wall_pts_ids'][fit_info['refined']['wall_pts_ids']]
                self.shell.indices = self.shell_cluster['point_indices'][fit_info['wall_pts_ids']]
                # open3d to build KDTree
                # shell_pcd = o3d.geometry.PointCloud()
                # shell_pcd.points = o3d.utility.Vector3dVector(self.shell.points)
                # # Build KDTree for this shell
                # self.shell.kdtree = o3d.geometry.KDTreeFlann(shell_pcd)
            self.shell.kdtree = None
    
    def estimate_top_z(self):
        self.z_top = self.ptc[self.shell.indices,2].max()
        return self.z_top

    def estimate_height(self):
        self.height = self.z_top - self.z_ground
        return self.height
            
    def remove_wall_points(self, kdtree, min_distance=0.1):
        close_point_ind = []
        for i in np.arange(self.ptc.shape[0]):
            n, _, _ = kdtree.search_radius_vector_3d(self.ptc[i,:], min_distance)
            if n > 0:
                close_point_ind.append(i)

        self.ptc = np.delete(self.ptc, np.array(close_point_ind), axis=0)

    def make_vertical(self, update_ground=True):
        if self.shell.points is None:
            self.shell.read_points()
        self.shell.make_vertical(update_cylinder=True, update_station=True, update_ground=update_ground)
        # translate the station point cloud
        self.transform(
            self.align['rotation'],
            self.align['translation']
        )
        # recalculate z_ground
        # self.z_ground = self.estimate_ground_z()

    def translate_origin(self):
        self.shell.z_min = self.z_ground
        self.shell.translate_origin(recalc_zmin=False, update_cylinder=True, update_station=True)
        # adjust z_ground
        self.z_ground = 0

    def find_columns(self, z_cuts=[1, 2, 3], thickness=0.05, num_pts_min=30, distance_to_wall_min=0.2, write_dir=None):
        # shell_points = self.ptc[self.shell.indices, :]
        # 3 slices
        # self.cross_sessions = []
        if write_dir is None:
            write_dir = self.file_path.parent / self.name / 'slice'
        # write_dir = self.file_path.parent / self.file_path.stem / 'slice'
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        points = []
        for z in z_cuts:
            logging.info(f'cut {z}m')
            # get points in this slice
            ind = (self.ptc[:,2] >= z) & (self.ptc[:,2] <= z + thickness)
            if ind.sum() == 0:
                continue
            cross_session = InternalStation(
                name=f'z_cut_{z}m',
                ptc=self.ptc[ind,:],
                read_points=False
            )

            # remove points on/close to the wall
            logging.info('remove wall points')
            # distance to center axis
            d = self.shell.axis.distance_to_point(cross_session.ptc)
            # deviation from shell wall model
            dev = self.shell.radius - d
            cross_session.ptc = cross_session.ptc[
                dev > distance_to_wall_min
            ]

            cross_session.ptc[:,2] = 0
            points.append(cross_session.ptc)
        
        logging.info('combine slices')
        try:
            cross_session = InternalStation(
                name='slices',
                ptc=np.concatenate(points),
                read_points=False
            )
            
            # cluster points
            logging.info('clustering')
            cross_session.clustering(
                cluster_params={
                    "cell_size": 0.1,
                    "min_ptc": 1,
                    'min_length': 0.1,
                    'min_tank_height': 0 #thickness / 2
                },
                output_dir=write_dir,
                write_file=False,
            )
            valid_clusters = []
            self.refined_cylinders = []
            logging.info('check clusters')
            for clst in cross_session.clusters:
                if clst['num_points'] < num_pts_min:
                    continue
                points = cross_session.ptc[clst['point_indices']]
                valid_clusters.append(clst)

                # write cluster points to file
                file_name = f"{self.name} col_{clst['id']}.pcd"
                file_path = os.path.join(write_dir, file_name)
                write_array2pcd(file_path, points)

                cyl = Cylinder(
                    radius=0.1,
                    position=np.mean(points, axis=0),
                    vector=np.array([0,0,1]).reshape((1,3)),
                    id=clst['id'],
                    ptc_filepath=clst['filepath'],
                    station=self
                )
                self.refined_cylinders.append(cyl)
            cross_session.clusters = valid_clusters
            self.cross_sessions = cross_session
        except Exception:
            self.cross_sessions = None

    def check_invalid_station(self, cross_distance_max=0.004, thickness=0.02):
        # Error 1: two overlapping layers of points
        z_cuts = [
            self.z_ground + 5 * self.height / 8,
            self.z_ground + 6 * self.height / 8,
            self.z_ground + 7 * self.height / 8
        ]
        for h in z_cuts:
            points = self.ptc[
                (self.ptc[:,2] >= h) & (self.ptc[:,2] <= h + thickness)
            ]
            # only considers points close to the shell wall
            d = self.shell.axis.distance_to_point(points)
            dev = abs(self.shell.radius - d)
            # dev = self.shell.radius - d
            points = points[
                dev < 0.2
            ]

            # distance & angle to center
            rho, theta = cart2pol(points[:,0], points[:,1])
            # divide the circle into small segments
            # the length of each segment should be small, ideally smaller than distance between points
            segment_length = 0.005
            num_segment = (2 * np.pi * self.shell.radius) / segment_length
            stat, bin_edges, binnumber = binned_statistic(theta, rho, lambda rs: rs.max() - rs.min(), bins=num_segment)
            # where the deviation is large
            invalid_loc = np.where(stat > cross_distance_max)[0]
            if invalid_loc.shape[0] == 0:
                self.valid = True
                self.suspicious_points = None
                return True, None
            # ranges of consecutive indices
            gaps = np.diff(invalid_loc)
            cont = gaps < 5
            def contiguous_regions(condition):
                """Finds contiguous True regions of the boolean array "condition". Returns
                a 2D array where the first column is the start index of the region and the
                second column is the end index.
                https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
                """

                # Find the indicies of changes in "condition"
                d = np.diff(condition)
                idx, = d.nonzero() 

                # We need to start things after the change in "condition". Therefore, 
                # we'll shift the index by 1 to the right.
                idx += 1

                if condition[0]:
                    # If the start of condition is True prepend a 0
                    idx = np.r_[0, idx]

                if condition[-1]:
                    # If the end of condition is True, append the length of the array
                    idx = np.r_[idx, condition.size] # Edit

                # Reshape the result into two columns
                idx.shape = (-1,2)
                return idx
            range_ind = contiguous_regions(cont)
            range_ind = range_ind[
                np.diff(range_ind).squeeze() > int(0.05 / segment_length) # > 5 cm range
            ].reshape((-1,2))
            
            if range_ind.shape[0] > 0:
                thick_ind = np.zeros(binnumber.shape) == 1
                for r in range_ind:
                    thick_ind |= np.isin(binnumber, invalid_loc[r[0]:r[1]]+1)
                self.valid = False
                points = np.c_[points, np.zeros((points.shape[0],1))]
                points[thick_ind,-1] = 1
                self.suspicious_points = points
                return False, points
            
        self.valid = True
        self.suspicious_points = None
        return True, None

    def extract_manway(self, cell_size=None, write_to_file=True, max_height=3, min_height=0.2):
        # shell wall above floor
        if not hasattr(self, 'z_ground'):
            self.estimate_ground_z()
        ind = self.shell.indices[
            (self.ptc[self.shell.indices, 2] >= (self.z_ground + min_height)) & (self.ptc[self.shell.indices, 2] <= (self.z_ground + max_height))
        ]
        self.manway, self.hole_surrouding_points = manway_extraction(
            self.ptc,
            shellwall_ptc_ids=ind,
            cell_size=cell_size
        )
        if write_to_file:
            path = Path(self.file_path)
            write_dir = path.parent / path.stem
            manway_points = []
            for way_id, way_data in self.manway.items():
                p = self.ptc[way_data['ptc_ids']]
                # a quick check if this potentially does not represent a manway
                # from a test run shows a mis-detection which associates with the floor data
                # to detect this, project the manway's points onto Oxy plane and check the bouding box
                x_min = p[:,0].min()
                x_max = p[:,0].max()
                if x_max - x_min > 2:
                    continue
                y_min = p[:,1].min()
                y_max = p[:,1].max()
                if y_max - y_min > 2:
                    continue
                
                manway_points.append(
                    np.c_[
                        p,
                        way_id * np.ones((p.shape[0], 1))
                    ]
                )
            try:
                write_array2txt(
                    write_dir / f'{self.name}_manway.txt',
                    np.concatenate(manway_points),
                    parallel=False
                )
                write_array2txt(
                    write_dir / f'{self.name}_manway_extent.txt',
                    self.hole_surrouding_points,
                    parallel=False
                )

            except Exception:
                pass
            


