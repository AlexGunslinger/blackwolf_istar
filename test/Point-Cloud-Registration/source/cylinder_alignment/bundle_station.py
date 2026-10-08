import networkx as nx
import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from pathlib import Path
from utils.read_pts import read_input_data
from utils.write_output import write_array2txt

class BundleStation():
    def __init__(self, stations=[]):
        self.stations = stations
    def __repr__(self):
        return str([
            s for s in self.stations
        ])
    def get_station(self, name):
        for s in self.stations:
            if s.name == name:
                return s
        return None
    def guess_target_cylinder(self, num_station=100):
        n = 0
        best_guess = None
        self.target_cylinder = []
        for s in self.stations:
            for c in s.cylinders:
                if len(c.match) > n:
                    best_guess = c
                elif len(c.match) == num_station:
                    self.target_cylinder.append(c)
        self.target_cylinder.insert(0, best_guess)
                    
    def get_transform_matrices(self):
        trans = {}
        for s in self.stations:
            # ref: s
            trans[(s,s)] = {
                'rotation': np.identity(3),
                'translation': np.zeros([3,1])
            }
            for t in s.transformation:
                trans[s, t['station']] = t
        return trans
    def rough_alignment(self, ref_station=None):
        # first build a graph connection between aligned stations
        g = nx.Graph()
        for s in self.stations:
            for t in s.transformation:
                g.add_edge(s, t['station'], weight=t['RMS'])
        # translate to the reference station
        if not ref_station:
            # select the first station as the ref
            ref_station = self.stations[0]
        # all tranformations between stations
        trans = self.get_transform_matrices()
        # return value
        info = {
            'ref_station': ref_station,
            'transformation': {}
        }
        # now go through each station
        for s in self.stations:
            if s is ref_station:
                s.align = {
                'ref_station': ref_station,
                'rotation': np.identity(3),
                'translation': np.zeros([3,1])
            }
                continue
            # shortest path
            p = nx.shortest_path(g, s, ref_station)
            # calculate transform matrix
            r = np.identity(3)
            t = np.zeros([3,1])
            for i,j in zip(p[:-1],p[1:]):
                # j: ref
                rot = trans[(j,i)]['rotation']
                tra = trans[(j,i)]['translation']
                r = rot@r
                t = rot@t + tra
            info['transformation'][s] = {
                'station': s,
                'rotation': r,
                'translation': t
            }
            s.align = {
                'ref_station': ref_station,
                'rotation': r,
                'translation': t
            }
        self.rough_align = info
    
    def get_all_xy_intersect(self, transform=True, ignore_radius=100, refined=False):
        p = []
        for s in self.stations:
            r = s.align['rotation']
            t = s.align['translation']
            if refined:
                cylinders = getattr(s, 'refined_cylinders')
            else:
                cylinders = getattr(s, 'cylinders')
            for c in cylinders:
                if c.radius >= ignore_radius:
                    continue
                center = c.axis.intersect_xy().reshape((3,1))
                # transform
                if transform:
                    center = r@center + t
                p.append([
                    s.name,
                    c.id,
                    c.radius,
                    center[0][0],
                    center[1][0],
                    center[:2].squeeze()
                ])
        return p
    
    def cluster_cylinder(self):
        # get xy intersect of all 
        centers = self.get_all_xy_intersect(transform=True, refined=True)
        # construct a dataframe
        center_df = pd.DataFrame(
            centers, 
            columns=['station', 'cylinder id', 'radius', 'x', 'y', 'xy']
        )
        # use DBSCAN to cluster points
        db_model = DBSCAN(eps=5, min_samples=3)
        db_model.fit(center_df[['x','y']].to_numpy())
        center_df['label'] = [str(l) for l in db_model.labels_]
        # add cluster label to cylinders
        tanks = {}
        for i, cyl in center_df.iterrows():
            lbl = cyl.label
            if int(lbl) < 0:
                continue
            # get station
            s = self.get_station(cyl['station'])
            # find cylinder object
            c = s.get_cylinder(cyl['cylinder id'], refined=True)
            c.tank_id = int(lbl)
            if lbl not in tanks:
                tanks[lbl] = []
            tanks[lbl].append(c)
        self.tanks_info = {
            'centers_xy': center_df,
            'tanks': tanks
        }
    def write_cylinder_segments(self, tank_label, write_dir='.', tank_name=None, surrounding=False, max_radius=50):
        if not tank_label in self.tanks_info['tanks']:
            print('tank id not found')
            return
        if not tank_name:
            tank_name = tank_label
        for c in self.tanks_info['tanks'][tank_label]:
            # # read original points
            # c.extend_ground()
            src_pts_xyz = c.points
            p = Path(c.ptc_filepath)
            # ptc_col_names =  ["x","y","z"]
            # # read point cloud
            # _ = read_input_data(p.parent, p.name)
            # src_pts_xyz = _[ptc_col_names].values
            # if the corresponding station has less than 3 cylinders, we write all the available segments
            # otherwise, we only write the target segment
            # get station
            if surrounding:
                s = c.station

                # nummber of cylinders
                valid_cyls = [cyl for cyl in s.refined_cylinders if cyl.radius < max_radius]
                num_cylinder = len(valid_cyls)
                if num_cylinder < 30:
                    for vc in valid_cyls:
                        if vc.id == c.id:
                            continue
                        vc.read_points()
                        # add these points to a total one
                        src_pts_xyz = np.r_[src_pts_xyz, vc.points]
            # transform
            rotation = c.station.align['rotation']
            translation = c.station.align['translation']
            src_pts_xyz = src_pts_xyz.transpose()
            translated_xyz = (rotation@src_pts_xyz) + translation
            # write
            # file_name = f'tank_{tank_name}_{p.stem}.txt'
            file_name = f'{c.station.name}.txt'
            file_path = os.path.join(write_dir, file_name)
            write_array2txt(
                file_path, 
                translated_xyz.transpose()
            )
    def write_all_tanks(self, write_dir='.', tank_name=None, surrounding=False, max_radius=50):
        for tank_label in self.tanks_info['tanks']:
            print(f'Writing tank {tank_label}')
            tank_dir = os.path.join(write_dir, f'tank_{tank_label}')
            if not os.path.exists(tank_dir):
                os.mkdir(tank_dir)
            self.write_cylinder_segments(
                tank_label,
                tank_dir,
                surrounding=surrounding,
                max_radius = max_radius
            )