from pathlib import Path
from networkx.drawing.layout import rescale_layout_dict
import numpy as np
import os
from numpy.core.fromnumeric import transpose
import scipy
import open3d as o3d
import gc

from skimage.measure import fit
from utils.write_output import write_array2txt
from utils.utilities import cart2pol, pol2cart
from utils.data_structures import grid1d
from utils.outlier_removal import removal_rail
from utils.read_pts import read_input_data
from utils.fitting import leastsq_fit_circle
from utils.math_methods import rotate_vectors, rotate_around_z
from cylinder_extraction.utils import fit_cylinder
from cylinder_extraction.config import threshold

class Line():
    def __init__(self, position=None, vector=None):
        self.position = position
        self.vector = vector
    def intersect_xy(self):
        # p = position + m x vector
        m = - self.position[-1] / self.vector[-1]
        p = self.position + m * self.vector
        return p
    def __repr__(self):
        return str({
            'position': self.position,
            'vector': self.vector
        })
    def transform(self, rotation, translation):
        self.position = self.position.reshape([3,1])
        self.position = (rotation@self.position) + translation
        self.vector = self.vector.reshape([3,1])
        self.vector = rotation@self.vector
    
    def distance_to_point(self, p):
        # 2 points that define this line
        a = self.position.squeeze() # A
        b = a + self.vector.squeeze() # B
        # distance = vector_AP x vector AB / |vector_AB|
        d = np.divide(b - a, np.linalg.norm(b - a))
        c = np.cross(p - a, d)
        h = np.linalg.norm(c, axis=1)
        return h

class Cylinder():
    def __init__(self, points=None, radius=None, position=None, vector=None, match=[], **kwargs):
        self.points = points
        self.radius = radius
        self.axis = Line(position, vector)
        self.match = []
        # metadata: station, tank id
        for key, val in kwargs.items():
            setattr(self, key, val)
    def __repr__(self):
        return str({
            'station': getattr(self, 'station', 'no station'),
            'radius': self.radius,
            'axis': self.axis.__repr__(),
            'ptc_filepath': getattr(self, 'ptc_filepath', 'no ptc_filepath')
        })
    def read_points(self, ptc_col_names=['x', 'y', 'z']):
        p = Path(self.ptc_filepath)
        if p.suffix == '.txt':
            _ = read_input_data(p.parent, p.name)
            src_pts_xyz = _[ptc_col_names].values
            self.points = src_pts_xyz
        elif p.suffix == '.pcd':
            pcd =  o3d.io.read_point_cloud(self.ptc_filepath)
            self.points = np.asarray(pcd.points)
    def write_points(self, sub_dir=None, filename=None, center=False, write_dir=None, data_type='txt'):
        p = Path(self.ptc_filepath)
        if write_dir is None:
            if sub_dir:
                write_dir = str(p.parent / sub_dir)
                if not os.path.exists(write_dir):
                    os.mkdir(write_dir)
            else:
                write_dir = str(p.parent)
        if filename:
            filepath = os.path.join(write_dir, filename)
        else:
            filepath = os.path.join(write_dir, f'{p.stem}.{data_type}')
        # add a center point
        if center:
            points = np.r_[self.point, self.axis.position]
        else:
            points = self.points
        print(f'write to {filepath}: {points.shape} points')
        if data_type == 'txt':
            write_array2txt(filepath, points)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # save
            o3d.io.write_point_cloud(
                filepath,
                pcd, print_progress=True
            )

    def get_and_write_points(self, write_dir, load_data=False, extend_ground=True, transform=False):
        if load_data:
            self.read_points()
        if extend_ground:
            self.station.read_points()
            self.calc_z_min()
            self.extend_ground()
        if transform:
            self.transform(self.align['rotation'], self.align['translation'], update_axis=False, update_z_min=False)
        self.write_points(center=False, write_dir=write_dir, data_type='pcd')
        # clean up
        del self.station.ptc
        self.station.ptc = None
        del self.points
        self.points = None
        gc.collect()

    def extend_ground(self, extra_radius=1.5, in_place=True):
        # bottom center
        # center = self.axis.intersect_xy()
        # self.read_points()
        # z_min = self.points[:,2].min() + 1
        # center = np.c_[center, z_min]
        # 4m + radius
        # read points from related station
        s = self.station
        if s.ptc is None:
            s.read_points()
        
        # distances to the cylinder axis
        d = self.axis.distance_to_point(s.ptc)
        mask = d <= self.radius + extra_radius

        if in_place:
            self.points = s.ptc[mask,:]
            return self.points
        else:
            return s.ptc[mask,:]

    def update_filepath(self, new_dir=None, num_keep_level=1, file_name=None):
        tails = []
        p = self.ptc_filepath
        for i in range(num_keep_level):
            parts = os.path.split(p)
            tails.insert(0, parts[-1])
            p = parts[0]
        if file_name is not None:
            tails[-1] = file_name
        if new_dir is None:
            new_dir = p
        self.ptc_filepath = os.path.join(new_dir, *tails)
    
    def transform(self, rotation, translation, update_cylinder=True, update_axis=True, update_z_min=True):
        # transform
        if self.points is not None:
            points = self.points.transpose()
            points = (rotation@points) + translation
        
            if update_cylinder:
                self.points = points.transpose()
        if update_axis:
            self.axis.transform(rotation, translation)
        if update_z_min:
            self.z_min = points[2,:].min()

    def make_vertical(self, update_cylinder=True, update_station=True):
        # rotate the axis vector to vertical directoin
        vector = self.axis.vector.reshape([1,3])
        # to be sure the vector pointing upwward
        if vector[0,2] < 0:
            vector = - vector
        r = rotate_vectors(np.c_[0, 0, 1], vector)
        if update_cylinder:
            # axis vector
            vector_rot = vector @ r
            self.axis.vector = vector_rot.reshape([3,1])
            # points
            self.points = self.points @ r
        # update the alignment matrix for the related station
        if update_station:
            self.station.update_transform(r.transpose(), np.zeros([3,1]))
    
    def find_ground_z(self, z_scan_max=0, step=0.1):
        points = self.points
        # z_max = points[:,2].max()
        z_min = points[:,2].min()
        cnt = []
        if z_min > z_scan_max:
            # incomplete data at the foot of the tank
            # try to get as much data as possible
            z_scan_max = z_min + 1
        for z in np.arange(z_min, z_scan_max, step):
            # if z % 10 == 0:
            #     print(z)
            ind = (z <= points[:,2]) & (points[:,2] < z + step)
            cnt.append(ind.sum())
        cnt = np.array(cnt)
        diff = np.diff(cnt)
        diff = np.insert(diff, 0, cnt[0])
        # a trick to include the lowest level as a peak
        diff = np.insert(diff, 0, 0)
        # find the left most peak among the top 5 peaks
        peak_indices = scipy.signal.find_peaks(diff)
        peak_indices = peak_indices[0]

        if peak_indices.size > 0:
            i = peak_indices[np.argsort(diff[peak_indices])[-5:]].min() - 1 # -1 to compensate the addition of 0 and cnt[0] above
            self.z_ground = z_min + i * step
        else:
            # not reliable
            self.z_ground = z_scan_max
        return self.z_ground

    def calc_z_min(self):
        self.z_min = self.points[:,2].min()

    def translate_base(self, recalc_zmin=True, update_cylinder=True, update_station=True):
        # slide to z_min = 0
        if recalc_zmin:
            self.calc_z_min()
        z_min = self.z_min
        if update_cylinder:
            self.points[:,2] += - z_min
        # update the alignment matrix for the related station
        if update_station:
            self.station.update_transform(np.identity(3), np.r_[0,0,-z_min].reshape([3,1]))
        
    def translate_origin(self, recalc_zmin=True, update_cylinder=True, update_station=True):
        # slide to z_min = 0
        if recalc_zmin:
            # self.calc_z_min()
            self.z_min = self.find_ground_z()
        if not hasattr(self, 'z_min'):
            print(self)
        z_min = self.z_min
        if update_cylinder:
            self.points[:,2] += - z_min
        else:
            points = self.points
            points[:,2] -= z_min
        # update the alignment matrix for the related station
        if update_station:
            self.station.update_transform(np.identity(3), np.r_[0,0,-z_min].reshape([3,1]))
        # xy
        center = self.axis.intersect_xy()
        if update_cylinder:
            self.points = self.points - center.reshape([1,3])
            self.axis.position = self.axis.position - center
        else:
            points = points - center.reshape([1,3])
        # update the alignment matrix for the related station
        if update_station:
            self.station.update_transform(np.identity(3), -center.reshape([3,1]))

        if update_cylinder:
            return None
        else:
            return points
    
    def rotate_around_z(self, theta, update_station=True):
        self.points = rotate_around_z(self.points, theta)

    def fit(self, refined=False):
        print(threshold['cylinder_estimation']['min_num_ptc'])
        res = fit_cylinder(
            self.points,
            threshold
        )
        # get outliers
        outliers_ind = np.delete(
            np.arange(0, self.points.shape[0]),
            res[0],
            axis=0
        )
        outliers_xyz = self.points[outliers_ind,:]
        
        fit_info = dict(zip(
            ['wall_pts_ids', 'wall_pts_xyz', 'voxels', 'cyl_params'],
            res
        ))
        fit_info.update({
            'outliers_ind': outliers_ind,
            'outliers_xyz': outliers_xyz
        })
        if refined:
            res = fit_cylinder(fit_info['wall_pts_xyz'], threshold)
            # get outliers
            outliers_ind = np.delete(
                np.arange(0, fit_info['wall_pts_xyz'].shape[0]),
                res[0],
                axis=0
            )
            outliers_xyz = fit_info['wall_pts_xyz'][outliers_ind,:]
            fit_info['refined'] = dict(zip(
                ['wall_pts_ids', 'wall_pts_xyz', 'voxels', 'cyl_params'],
                res
            ))
            fit_info['refined'].update({
                'outliers_ind': outliers_ind,
                'outliers_xyz': outliers_xyz
            })
        return fit_info
        
    def estimate_top_surface(self, debug=False):
        if self.points is None:
            self.read_points()
        z_min = min(self.points[:,2])
        z_max = max(self.points[:,2])

        self.shell_grids = grid1d(
            self.points, 
            z_min, z_max, 
            0.2, 
            'z'
        )

        # shell grid info
        # x, y, z
        # only use x,y
        # for now, use the xy_intersect center
        # x, y - index 1, 2
        shell_grid_info = np.r_[0, self.axis.intersect_xy()]

        # remove rails
        self.top_surface, self.all_top_ptc_ids = removal_rail(
            self.points, 
            self.shell_grids, shell_grid_info, 
            interval_thickness = 0.2,
            interval_angle = 2*np.pi/180, 
            rail_gap = 0.1#0.025
        )
        # fit circle
        try:
            center_x, center_y, __, __, __, __ = leastsq_fit_circle(
                self.top_surface['ptc'][:,0],
                self.top_surface['ptc'][:,1],
            )
            center_z = np.mean(self.top_surface['ptc'][:,2])
            self.top_surface['center'] = np.array([
                center_x,
                center_y,
                center_z
            ])
            if debug:
                p = self.ptc_filepath
                p = os.path.split(p)[0]
                p = os.path.split(p)[0]
                debug_dir = os.path.join(p, 'debug')
                if not os.path.exists(debug_dir):
                    os.mkdir(debug_dir)
                write_array2txt(
                    os.path.join(debug_dir, f'{self.station.name} {self.id}.txt'),
                    np.r_[self.points, self.top_surface['center'].reshape([1,-1])]
                )
        except Exception:
            print('fit circle failed')
        