from pathlib import Path
import os
from time import time
import numpy as np
from itertools import chain
import pyransac3d as pyrsc
import open3d as o3d

from multiprocessing import Pool

from ..utils.read_pts import read_input_data
from ..utils.cell_grid import cell_grid, cell_grid_modify
from ..utils.write_output import write_array2txt, write_array2pcd
from ..utils.math_methods import project_point_onto_plane
from ..cylinder_extraction.utils import fit_cylinder
from ..cylinder_extraction.config import threshold
from ..cylinder_extraction.utils import cells_classification, cells_connectivity
from .cylinder import Cylinder

class Station():
    def __init__(self, name='station', ptc=None, file_path=None, read_points=True,
    ptc_col_names=['x', 'y', 'z'], read_e57_raw=False,
    transformation=[], cylinders=[], station_id=0):
        if file_path:
            self.file_path = file_path
            if read_points:
                p = Path(file_path)
                print(p.parent, p.name)
                # read point cloud
                if p.suffix == '.e57':
                    _ = read_input_data(p.parent, p.name, read_e57_raw=read_e57_raw)
                    src_pts_xyz = _[ptc_col_names].values
                    self.ptc = src_pts_xyz
                elif p.suffix == '.pcd':
                    pcd =  o3d.io.read_point_cloud(file_path)
                    self.ptc = np.asarray(pcd.points)
            else:
                self.ptc = ptc
        else:
            self.ptc = ptc
        self.name = name
        self.transformation = transformation
        self.cylinders = cylinders
        self.id = station_id
    
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
    
    def estimate_ground_z(self):
        """
        determinte the z coordinate of ground
        """
        # two step
        # coarse estimate: every 0.5 meter
        step_rough = 0.5
        z_max = self.ptc[:,2].max()
        z_min = self.ptc[:,2].min()
        hist, bin_edges = np.histogram(self.ptc, bins=np.arange(z_min, z_max, step_rough))
        # the ground is determined with respect to the bin with the most number of points, i.e. z
        max_ind = np.argmax(hist)
        # fine estimate
        step_fine = 0.02
        z_min = bin_edges[max_ind]
        z_max = bin_edges[max_ind + 1]
        hist, bin_edges = np.histogram(self.ptc, bins=np.arange(z_min, z_max, step_fine))
        # the ground is determined with respect to the bin with the most number of points, i.e. z
        z_ground = bin_edges[np.argmax(hist)]

        self.z_ground = z_ground
        return z_ground

    
    def extract_cylinder(self, fit=True, output_dir=False, refine=True, extent_ground=False, write_file=False, write_parallel=True, data_type='pcd'):
        # Voxel tree
        time_begin = time()
        cellgrid = cell_grid(self.ptc, threshold["cell_size"] , threshold["min_ptc"])
        cellgrid.subdivision()
        cellgrid = cell_grid_modify(None, None, None).cellgrid_extraction(cellgrid, cell_prop="Full")
        print('Time to generate a cell grid {:.03f} seconds'.format(time() - time_begin))
        # Remove ground points
        tank_cells_linear_ids, _ = cells_classification(cellgrid, threshold)
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
            output_dir = os.path.join(Path(self.file_path).parent, f'{self.name}-output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # self.poi = np.empty([1,3])
        self.num_clusters = len(clusters)
        for cluster_id in clusters:
            print(cluster_id)
            clsuter_cell_linear_ids = clusters[cluster_id]

            clusters_pts_ids = list(chain.from_iterable(
                cellgrid.ptc_ids[voxel_id] for voxel_id in clsuter_cell_linear_ids))
            clusters_pts_ids = np.asarray(clusters_pts_ids, dtype=np.uint32)
            clusters_pts_xyz = cellgrid.points[clusters_pts_ids]
            # write cluster points to file
            file_name = f"{self.name} cluster_{cluster_id}.{data_type}"
            file_path = os.path.join(output_dir, file_name)
            if write_file:
                if data_type == 'txt':
                    write_array2txt(file_path, clusters_pts_xyz, parallel=write_parallel)
                elif data_type == 'pcd':
                    write_array2pcd(file_path, clusters_pts_xyz)

            # self.poi = np.r_[self.poi, clusters_pts_xyz]
            # fit cylinder from raw data
            cluster_type = 'unknown'
            # extend points (to include surrounding ground)
            ext_points = None
            if fit:
                try:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(clusters_pts_xyz)
                    pcd = pcd.voxel_down_sample(voxel_size=0.02)
                    clusters_pts_xyz = np.asarray(pcd.points)
                    _, shell_wall_pts_xyz, _, cyl_paras = fit_cylinder(clusters_pts_xyz, threshold)
                    cyl = Cylinder(
                        radius=cyl_paras[3],
                        position=cyl_paras[:3],
                        vector=cyl_paras[-3:],
                        metadata={
                            'station': self,
                            'id': cluster_id,
                            'ptc_filepath': file_path
                        }
                    )
                    self.cylinders.append(cyl)
                    
                    cluster_type = 'cylinder'
                    if refine:
                        print('refining')
                        _, _, _, refined_cyl_paras = fit_cylinder(shell_wall_pts_xyz, threshold)
                        cyl = Cylinder(
                            radius=refined_cyl_paras[3],
                            position=refined_cyl_paras[:3],
                            vector=refined_cyl_paras[-3:],
                            metadata={
                                'station': self,
                                'id': cluster_id,
                                'ptc_filepath': file_path
                            }
                        )
                        self.refined_cylinders.append(cyl)
                    if extent_ground:
                        ext_points = cyl.extend_ground(in_place=False)
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
            clst = {
                'id': cluster_id,
                'filepath': file_path,
                'type': cluster_type
            }
            if cluster_type != 'cylinder':
                clst['points'] = clusters_pts_xyz
            self.clusters.append(clst)
            if extent_ground:
                self.clusters[-1]['extended_points'] = ext_points
    
    def extract_cylinder_parallel(self, output_dir=False, refine=True, extend_ground=True):
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
        # self.poi = np.empty([1,3])
        self.num_clusters = len(clusters)

        # fit_cylinder_to_cluster(clusters_pts_xyz, cluster_id, station_name, output_dir='', write_file=False, fit=True, refine=True, extend_ground=False):
        pts_list = []
        for cluster_id in clusters:
            print(cluster_id)
            clsuter_cell_linear_ids = clusters[cluster_id]

            clusters_pts_ids = list(chain.from_iterable(
                cellgrid.ptc_ids[voxel_id] for voxel_id in clsuter_cell_linear_ids))
            clusters_pts_ids = np.asarray(clusters_pts_ids, dtype=np.uint32)
            clusters_pts_xyz = cellgrid.points[clusters_pts_ids]
            # add to list
            pts_list.append(clusters_pts_xyz)
        print('starting parallel fitting cylinders')
        with Pool(10) as p:
            parsed_data = p.starmap(
                fit_cylinder_to_cluster,
                zip(
                    pts_list,
                    clusters,
                    [self.name] * self.num_clusters,
                    [output_dir] * self.num_clusters
                )
            )
        print('finish fitting cylinders')
        
        for cls, cyl, ref_cyl in parsed_data:
            # ext_points = None

            if cyl is not None:
                cyl.station = self
                self.cylinders.append(cyl)
                # if extend_ground and (not refine):
                #     ext_points = cyl.extend_ground(in_place=False)
            if ref_cyl is not None:
                ref_cyl.station = self
                self.refined_cylinders.append(ref_cyl)
                # if extend_ground:
                #     ext_points = ref_cyl.extend_ground(in_place=False)
            
            # cls['extended_points'] = ext_points
            self.clusters.append(cls)

    def extract_plane(self, max_cyl_radius=100, num_point_min=50000, write_file=True):
        self.planes = []
        plane = pyrsc.Plane()
        plane_clusters = []
        cluster_id = self.num_clusters + 1
        for clustr in self.clusters:
            # valid cylinders have radius smaller or equal to the max_cyl_radius
            if clustr['type'] == 'cylinder':
                cyl = self.get_cylinder(clustr['id'])
                if cyl.radius < max_cyl_radius:
                    continue
            # not a valid cylinder, try to fit plane
            print(f"fitting plane to cluster {clustr['id']}")
            points = clustr['points']
            while points.shape[0] > num_point_min:
                plane.fit(points, thresh=0.05, minPoints=num_point_min, maxIteration=1000)
                if plane.inliers.shape[0] > num_point_min:
                    # write cluster points to file
                    file_name = f"{self.name} cluster_{cluster_id}.txt"
                    p = Path(clustr['filepath'])
                    file_path = os.path.join(p.parent, file_name)
                    if write_file:
                        write_array2txt(file_path, points[plane.inliers,:])
                    self.planes.append({
                        'id': cluster_id,
                        'original_cluster': clustr['id'],
                        'filepath': file_path,
                        'points': points[plane.inliers,:],
                        'equations': plane.equation
                    })
                    plane_clusters.append({
                        'id': cluster_id,
                        'filepath': file_path,
                        'points': points[plane.inliers,:],
                        'type': 'plane'
                    })
                    # get remaining points
                    points = np.delete(points, plane.inliers, 0)
                    # increase the id
                    cluster_id += 1
                else:
                    break
        self.extended_clusters = self.clusters + plane_clusters
    def create_artificial_cylinders(self, radius=0.001, cylinder_radius_max=100):
        """ This function creates an artificial cylinder for each of projecting points of cylinders' center onto available planes
        """
        art_cylinders = []
        cyl_id = max([c.id for c in self.refined_cylinders]) + 1
        for pl in self.planes:
            for cyl in self.refined_cylinders:
                if cyl.radius > cylinder_radius_max:
                    continue
                # center of cyl
                center = cyl.axis.intersect_xy()
                # project cyl's center onto the plane pl
                proj, _ = project_point_onto_plane(
                    center,
                    pl['equations']
                )
                # create an artificial cylinders
                # perfectly vertical
                art_cylinders.append(Cylinder(
                    radius=radius,
                    position=proj,
                    vector=np.array([0,0,1]),
                    metadata={
                        'id': cyl_id,
                        'station': self,
                        'cylinder_id': cyl.id,
                        'plane_id': pl['id'],
                        'name': f"{cyl.id} on {pl['id']}",
                        'ptc_filepath': None
                    }
                ))
                cyl_id += 1
        self.refined_cylinders = self.refined_cylinders + art_cylinders


    def update_filepath(self, new_dir, num_keep_level=1):
        tails = []
        p = self.file_path
        for i in range(num_keep_level):
            parts = os.path.split(p)
            tails.insert(0, parts[-1])
            p = parts[0]
        self.file_path = os.path.join(new_dir, *tails)

    def write_points(self, filename=None, data_type='txt'):
        if not filename:
            filename = f'{self.name}.{data_type}'
        print(f'write to {filename}')
        if data_type == 'txt':
            write_array2txt(filename, self.ptc[:5000000,:])
        elif data_type == 'pcd':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.ptc)
            # save
            o3d.io.write_point_cloud(
                filename,
                pcd, print_progress=True
            )


    def read_points(self, read_e57_raw=False):
        p = Path(self.file_path)
        # read point cloud
        print(p.parent, p.name)
        _ = read_input_data(p.parent, p.name, read_e57_raw=read_e57_raw)
        ptc_col_names=['x', 'y', 'z']
        self.ptc = _[ptc_col_names].values

    def read_cylinders(self, refined=True, max_radius=100, add_id=False):
        if refined:
            atrb = 'refined_cylinders'
        else:
            atrb = 'cylinders'
        if add_id:
            ptc = np.empty([1,4])
        else:
            ptc = np.empty([1,3])
        for c in getattr(self, atrb):
            # if c.points is None:
            if hasattr(c, 'radius'):
                if c.radius > max_radius:
                    continue
            c.read_points()
            points = c.points
            if add_id:
                points = np.c_[points, c.id * np.ones([points.shape[0], 1])]
            ptc = np.r_[ptc, points]
        self.ptc = ptc
    
    def update_transform(self, rotation, translation):
        rot = rotation @ self.align['rotation']
        tra = rotation @ self.align['translation'] + translation
        self.align.update({
            'rotation': rot,
            'translation': tra
        })
    
    def transform(self, rotation, translation):
        ptc = rotation @ self.ptc.transpose() + translation
        self.ptc = ptc.transpose()

    def to_numpy(self, get_refined=False, top_surface=False):
        """This exports cylinder data into a numpy array. 
        Each row includes id, radius, xy intersect of the axis of the corresponding cylinder.
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
                    p = c.get_center(loc='ground').ravel()
                    # p = c.axis.intersect_xy()
                if c.radius > 100:
                    continue
                if not c.ptc_filepath:
                    p = list(c.axis.position)
                cyls.append([
                    c.id,
                    c.radius,
                    *p
                ])
            return np.array(cyls)
            # except Exception:
            #     raise ValueError('Not enough cylinder information')
    def horizontal_slice(self, lower_bound, thickness):
        # once slice
        # horizontal slice
        i = (lower_bound <= self.ptc[:,2]) & (self.ptc[:,2] <= lower_bound + thickness)
        return self.ptc[i,:]

def fit_cylinder_to_cluster(clusters_pts_xyz, cluster_id, station_name, output_dir='', write_file=True, fit=True, refine=True, extend_ground=False, data_type='pcd'):
    # write cluster points to file
    file_name = f"{station_name} cluster_{cluster_id}.{data_type}"
    file_path = os.path.join(output_dir, file_name)
    print(file_path)
    if write_file:
        if data_type == 'txt':
            write_array2txt(file_path, clusters_pts_xyz)
        elif data_type == 'pcd':
            write_array2pcd(file_path, clusters_pts_xyz)
    # fit cylinder from raw data
    cluster_type = 'unknown'
    # extend points (to include surrounding ground)
    # ext_points = None
    cyl = None
    refined_cyl = None
    if fit:
        try:
            _, shell_wall_pts_xyz, _, cyl_paras = fit_cylinder(clusters_pts_xyz, threshold)
            cyl = Cylinder(
                radius=cyl_paras[3],
                position=cyl_paras[:3],
                vector=cyl_paras[-3:],
                metadata={
                    'id': cluster_id,
                    'ptc_filepath': file_path
                }
            )
            
            cluster_type = 'cylinder'
            if refine:
                print('refining')
                _, _, _, refined_cyl_paras = fit_cylinder(shell_wall_pts_xyz, threshold)
                refined_cyl = Cylinder(
                    radius=refined_cyl_paras[3],
                    position=refined_cyl_paras[:3],
                    vector=refined_cyl_paras[-3:],
                    metadata={
                        'id': cluster_id,
                        'ptc_filepath': file_path
                    }
                )
            
        except Exception as e:
            print(f'Error while fitting cylinder to cluster id={cluster_id}')
            print(e)
    # else:
    #     self.cylinders.append(Cylinder(
    #         metadata={
    #             'station': self,
    #             'id': cluster_id,
    #             'ptc_filepath': file_path
    #         }
    #     ))
    cluster = {
        'id': cluster_id,
        'filepath': file_path,
        'type': cluster_type
    }

    # if cluster_type == 'cylinder':
    #     # write extend ground
    #     write_array2txt(
    #         os.path.join(output_dir, 'ext_ground_' + file_name),
    #         ext_points
    #     )

    return cluster, cyl, refined_cyl

# def extend_cylinder_ground()