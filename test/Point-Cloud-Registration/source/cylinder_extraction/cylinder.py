from pathlib import Path
import numpy as np
import os
from utils.write_output import write_array2txt
from utils.utilities import cart2pol, pol2cart
from utils.data_structures import grid1d
from utils.outlier_removal import removal_rail
from utils.read_pts import read_input_data
from utils.fitting import leastsq_fit_circle
from utils.math_methods import rotate_vectors
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

class Cylinder():
    def __init__(self, points=None, radius=None, position=None, vector=None, metadata=None, match=[]):
        self.points = points
        self.radius = radius
        self.axis = Line(position, vector)
        self.match = []
        # metadata: station, tank id
        if metadata:
            for key, val in metadata.items():
                setattr(self, key, val)
    def __repr__(self):
        return str({
            'station': getattr(self, 'station'),
            'radius': self.radius,
            'axis': self.axis.__repr__(),
            'ptc_filepath': getattr(self, 'ptc_filepath')
        })
    def read_points(self, ptc_col_names=['x', 'y', 'z']):
        p = Path(self.ptc_filepath)
        _ = read_input_data(p.parent, p.name)
        src_pts_xyz = _[ptc_col_names].values
        self.points = src_pts_xyz
    def write_points(self, sub_dir=None, filename=None, center=False):
        p = Path(self.ptc_filepath)
        if sub_dir:
            write_dir = str(p.parent / sub_dir)
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
        else:
            write_dir = str(p.parent)
        if filename:
            filepath = os.path.join(write_dir, filename)
        else:
            filepath = os.path.join(write_dir, p.name)
        # check center
        if center:
            points = np.r_[self.point]
        print(f'write to {filepath}')
        write_array2txt(filepath, points)
    def extend_ground(self, extra_radius=1.5):
        # bottom center
        center = self.axis.intersect_xy()
        # self.read_points()
        # z_min = self.points[:,2].min() + 1
        # center = np.c_[center, z_min]
        # 4m + radius
        # read points from related station
        s = self.station
        if s.ptc is None:
            s.read_points()
        print(s.ptc.shape)
        # extend points
        rho, theta = cart2pol(
            s.ptc[:,0] - center[0], 
            s.ptc[:,1] - center[1]
        )

        mask = rho <= extra_radius + self.radius
        self.points = s.ptc[mask,:]
        print(self.points.shape)

    def update_filepath(self, new_dir, num_keep_level=1):
        tails = []
        p = self.ptc_filepath
        for i in range(num_keep_level):
            parts = os.path.split(p)
            tails.insert(0, parts[-1])
            p = parts[0]
        self.ptc_filepath = os.path.join(new_dir, *tails)
    
    def make_vertical(self):
        # rotate the axis vector to vertical directoin
        r = rotate_vectors(np.c_[0, 0, 1], self.axis.vector)
        # axis vector
        self.axis.vector = self.axis.vector @ r
        # points
        self.points = self.points @ r
    
    def translate_base(self):
        # slide to z_min = 0
        z_min = self.points[:,2].min()
        self.points[:,2] = self.points[:,2] - z_min
        
    def translate_origin(self):
        # slide to z_min = 0
        z_min = self.points[:,2].min()
        self.points[:,2] = self.points[:,2] - z_min
        # xy
        center = self.axis.intersect_xy()
        self.points = self.points - center
        
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
        