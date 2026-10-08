from copy import deepcopy
from genericpath import exists
import networkx as nx
import numpy as np
import pandas as pd
import os
import gc
import open3d as o3d
from glob import glob
import pickle
import json
from multiprocessing import Pool

from sklearn.cluster import DBSCAN
from pathlib import Path
from ..cylinder_extraction.cylinder import Cylinder

from .station_alignment import StationAlignment
from .utils import draw_thin_slice

from ..registration.utils import multiway_load_point_clouds as load_point_clouds
from ..registration.utils import multiway_full_registration_with_pairs as full_registration_with_pairs
from ..registration.utils import multiway_filter_point_clouds

from ..utils.write_output import write_array2txt
from ..utils.math_methods import calc_overlap_curve, calc_circle_span
import traceback

class BundleStation():
    def __init__(self, stations=[], data_dir=None):
        self.stations = stations
        self.data_dir = data_dir
        if data_dir is not None:
            if isinstance(data_dir, str):
                self.data_dir = Path(data_dir)

    def __repr__(self):
        return str([
            s for s in self.stations
        ])

    def get_station(self, name):
        for s in self.stations:
            if s.name == name:
                return s
        return None
    
    def read_extracted_cylinder_data(self, file_types=['e57'], print_info=True, duplicate_original_axis=True):
        source_files = []
        for ext in file_types:
            source_files.extend(glob(str(self.data_dir / f'*.{ext}')))
        stations = []
        for fp in source_files:
            if 'roof' in fp.lower():
                continue
            p = Path(fp)
            station_name = p.stem
            station_pkl_file = p.parents[0] / station_name / f'{station_name}.pkl'
            if print_info: print(station_name)
            if os.path.exists(str(station_pkl_file)):
                with open(station_pkl_file, 'rb') as f:
                    stations.append(pickle.load(f))
        # check data
        valid_stations = []
        for s in stations:
            # ignore stations without cylinders
            if hasattr(s, 'refined_cylinders'):
                valid_stations.append(s)
            else:
                continue
            valid_cyls = []
            # removing invalid cylinders
            for c in s.refined_cylinders:
                if not np.isnan(c.radius):
                    valid_cyls.append(c)
                else:
                    if print_info: print('remove {s.name}, {c.id}')
            s.refined_cylinders = valid_cyls
        stations = valid_stations
        # summarise information
        summary = []
        for s in stations:
            # print('======')
            # print(s)
            s.update_filepath(self.data_dir, num_keep_level=1)
            for c in s.refined_cylinders:
                # update path
                if c.ptc_filepath:
                    c.update_filepath(self.data_dir, num_keep_level=2)
                # print(c)
                if duplicate_original_axis:
                    c.ori_axis = deepcopy(c.axis)
                summary.append({
                    'station': s.name,
                    'cylinder': c.id,
                    'radius': c.radius
                })
        summary_df = pd.DataFrame(summary)
        
        self.stations = stations
        return summary_df
    
    def duplicate_original_cylinder_info(self):
        for s in self.stations:
            for c in s.refined_cylinders:
                c.ori_axis = deepcopy(c.axis)
    
    def align_pairwise_station(self):
        num_station = len(self.stations)
        alignment = []
        for i in range(num_station):
            st0 = self.stations[i]
            for j in range(i+1, num_station):
                st1 = self.stations[j]
                align = StationAlignment(st0, st1)
                align.align(max_delta_radius=1, top_surface=False, scan_radius=17.8, RMS_min=1)
                # try:
                #     align.align(max_delta_radius=1, top_surface=False, scan_radius=17.8, RMS_min=1)
                # except Exception as exc:
                    # pass
                alignment.append(align)
        self.pairwise_alignment = alignment

    def remove_points(self):
        for s in self.stations:
            if hasattr(s, 'ptc'):
                del s.ptc
                s.ptc = None
            for c in s.refined_cylinders:
                if hasattr(c, 'points'):
                    del c.points
                    c.points = None
        gc.collect()

    def write_target_tank(self, write_dir='.', num_process=8, load_data=False, extend_ground=False, transform=False):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        num_cylinders = len(self.target_tank)
        # Cylinder.get_and_write_points(self.target_tank[0], write_dir, load_data, extend_ground, transform)
        with Pool(num_process) as p:
            p.starmap(
                Cylinder.get_and_write_points,
                zip(
                    self.target_tank,
                    [write_dir]*num_cylinders,
                    [load_data]*num_cylinders,
                    [extend_ground]*num_cylinders,
                    [transform]*num_cylinders)
            )
    
    def cut_horizontal(self, mode='target_tank', z_cut=3, thickness=0.01, write_dir=Path('.'), print_info=True):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        points = np.empty([1,4])
        for ind, cyl in enumerate(self.target_tank):
            if print_info: print(cyl.station.name, 'cluster', cyl.id)
            
            if mode == 'target_tank':
                cyl.read_points()
                cyl.transform(cyl.align['rotation'], cyl.align['translation'], update_axis=False, update_z_min=False)
                p = cyl.points
                file_path = write_dir / Path(cyl.ptc_filepath).name
            elif mode == 'station_cylinders':
                s = cyl.station
                s.read_cylinders()
                # transform
                s.transform(cyl.align['rotation'], cyl.align['translation'])
                p = s.ptc
                file_path = write_dir / f'{s.name}.pcd'
            # once slice
            i = (z_cut <= p[:,2]) & (p[:,2] <= z_cut + thickness)
            p = p[i,:]
            # accumulate
            points = np.r_[points, np.c_[p, ind*np.ones([p.shape[0],1])]]
            # write
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            # save
            if print_info: print(f'write to {file_path}')
            o3d.io.write_point_cloud(
                str(file_path),
                pcd, print_progress=True
            )
            del cyl.station.ptc
            cyl.station.ptc = None
            del cyl.points
            cyl.points = None
            gc.collect()
        write_array2txt(
            os.path.join(write_dir, f'{mode}_phase_1.txt'),
            points[1:,:]
        )

    def set_target_tank(self, tank_id):
        self.target_tank_id = tank_id
        self.target_tank = self.tanks_info['tanks'][str(self.target_tank_id)]

    def identify_target_tank(self, z_cut=3, thickness=0.01, print_info=True):
        # get a horizontal cut through all cylinders from all clusters
        cs_df = self.get_clusters_crosssection_points(z_cut, thickness, step=1)
        # center points
        center_df = self.tanks_info['centers_xy']
        # evaluate each cluster
        circle_spans = []
        labels = cs_df.tank_label.unique()
        for lbl in labels:
            coords = cs_df[cs_df.tank_label==lbl][['x','y']].values
            # center
            center = center_df[center_df.tank_label==lbl][['x', 'y']].values
            center = np.mean(center, axis=0)
            circle_spans.append(calc_circle_span(coords-center))
        if print_info: print('span', circle_spans)
        i = np.argmax(np.array(circle_spans))
        self.set_target_tank(tank_id=int(labels[i]))
        return self.target_tank_id
    
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
    def align_all_stations_by_axis(self, ref_station=None, stations=None, print_info=True):
        if stations is None:
            stations = self.stations
        # first build a graph connection between aligned stations
        g = nx.Graph()
        for s in self.stations:
            if s not in stations:
                continue
            for t in s.transformation:
                if t['station'] not in stations:
                    continue
                if not t['uncertain_flag']:
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
                s.is_aligned = True
                s.align = {
                    'ref_station': ref_station,
                    'rotation': np.identity(3),
                    'translation': np.zeros([3,1])
                }
                continue
            # shortest path
            if g.has_node(s):
                p = nx.shortest_path(g, s, ref_station, 'weight')
            else:
                print(f'station {s.name} is not connected')
                s.is_aligned = False
                continue
            if print_info: print(p)
            s.is_aligned = True
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
                # center = c.axis.intersect_xy().reshape((3,1))
                center = c.get_center(loc='ground').reshape((3,1))
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
    
    def cluster_cylinder(self, min_samples=2):
        # get xy intersect of all 
        centers = self.get_all_xy_intersect(transform=True, refined=True)
        # construct a dataframe
        center_df = pd.DataFrame(
            centers, 
            columns=['station', 'cylinder id', 'radius', 'x', 'y', 'xy']
        )
        # use DBSCAN to cluster points
        db_model = DBSCAN(eps=5, min_samples=min_samples)
        db_model.fit(center_df[['x','y']].to_numpy())
        center_df['tank_label'] = [str(l) for l in db_model.labels_]
        # add cluster label to cylinders
        tanks = {}
        for i, cyl in center_df.iterrows():
            lbl = cyl.tank_label
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
    
    def find_uncertain_stations(self):
        uncertain_stations = []
        for s in self.stations:
            certain = False
            for trans in s.transformation:
                certain = certain or (not trans['uncertain_flag'])
            if not certain:
                uncertain_stations.append(s)
        return uncertain_stations

    def handle_uncertain(self, neighbor_info):
        for s in self.find_uncertain_stations():
            # if s is not roughly aligned (due to having only 2 cylinders)
            print(s)
            # set neighbor stations
            # for now, do it manually
            s.neighbor_station = self.get_station(
                neighbor_info[s.name]
            )
            # read data
            # s.read_cylinders(add_id=False)
            s.read_points()
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(s.ptc[:,:3])
            # s.write_points(data_type='pcd')
            # z0 = s.find_ground_z()
            z0 = 0
            s1 = s.neighbor_station
            # s1.read_cylinders(add_id=False)
            s1.read_points()

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(s1.ptc[:,:3])
            # test matching
            lower_bound = 3
            thickness = 1
            # thin slice
            # s.ptc = s.horizontal_slice(z0 + lower_bound, thickness)
            # write_array2txt(
            #     os.path.join(data_dir, f'thin_slice_{s.name}.txt'),
            #     slice_ptc_0
            # )
            for trans_info in s.transformation:
                if trans_info['station'] == s.neighbor_station:
                    # transform from neighbor station (s1) to the non-connected station (s0)
                    trans0_rotations = trans_info['rotation']
                    trans0_translations = trans_info['translation']
                    # neighbor_points = deepcopy(s.neighbor_station.ptc)
                    # overlaps = []

                    # s1 = s.neighbor_station
                    z1 = 0
                    # s1.ptc = s1.horizontal_slice(z1 + lower_bound, thickness)

                    for trans_info_1 in s1.transformation:
                        if trans_info_1['station'] == s:
                            trans1_rotations = trans_info_1['rotation']
                            trans1_translations = trans_info_1['translation']
                            overlaps = []
                            evals = []
                            for i in range(2):
                                trans_info['rotation'] = trans0_rotations[i]
                                trans_info['translation'] = trans0_translations[i]
                                trans_info['uncertain_flag'] = False

                                trans_info_1['rotation'] = trans1_rotations[i]
                                trans_info_1['translation'] = trans1_translations[i]
                                trans_info_1['uncertain_flag'] = False

                                # s1.ptc = deepcopy(neighbor_points)

                                # pcd1 = o3d.geometry.PointCloud()
                                # pcd1.points = o3d.utility.Vector3dVector(s1.ptc[:,:3])

                                transformation = np.identity(4)
                                transformation[:3,:3] = trans_info['rotation']
                                transformation[:3,-1] = trans_info['translation'].squeeze()

                                # eval = o3d.pipelines.registration.evaluate_registration(
                                #     pcd1, pcd0, 0.5, 
                                #     transformation
                                # )

                                # evals.append(eval)

                                pcd11 = deepcopy(pcd1)
                                pcd11.transform(transformation)

                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(np.r_[np.asarray(pcd0.points), np.asarray(pcd11.points)])

                                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)

                                # len(voxel_grid.get_voxels())

                                evals.append(len(voxel_grid.get_voxels()))

                                del pcd11
                                del pcd
                                gc.collect()


                                # s1.transform( trans_info['rotation'], trans_info['translation'])

                                continue
                            # continue

                                # # create a temporal bundle that consists of only these 2 stations
                                # local_bundle = BundleStation(stations = [s, s1])
                                # # rough alignment
                                # local_bundle.rough_alignment(stations=[s,s1])
                                # # clustering
                                # local_bundle.cluster_cylinder()
                                # overlap = 0
                                # for cluster in local_bundle.tanks_info['tanks'].values():
                                #     for cyl in cluster:
                                #         cyl.points = cyl.station.ptc[cyl.station.ptc[:,-1] == cyl.id, :3]
                                #         # keep original axis
                                #         ori_axis = deepcopy(cyl.axis)
                                #         cyl.transform(
                                #             cyl.station.align['rotation'],
                                #             cyl.station.align['translation'],
                                #             update_cylinder=True, update_axis=True, update_z_min=False
                                #         )
                                #         cyl.z_min = cyl.points[:,2].min()
                                #         cyl.points = cyl.translate_origin(recalc_zmin=False, update_cylinder=False, update_station=False)
                                #         # recover original axis
                                #         cyl.axis = ori_axis
                                #     overlap += calc_overlap_curve(cluster[0].points[:,:2], cluster[1].points[:,:2])

                                # pass
                                # overlaps.append(overlap)
                            
                            # i_best = np.argmax(np.array(overlaps))
                            i_best = np.argmin(np.array(evals))
                            print(evals, i_best)
                            # r_best = rotations[i_best]
                            # t_best = translations[i_best]

                            # # update transformation information
                            trans_info['rotation'] = trans0_rotations[i_best]
                            trans_info['translation'] = trans0_translations[i_best]

                            trans_info_1['rotation'] = trans1_rotations[i_best]
                            trans_info_1['translation'] = trans1_translations[i_best]
                            # trans_info['translation'] = t_best
                            # trans_info['uncertain_flag'] = False

                                # # find transformation in s1 that point to s0
                                # for trans_info_1 in s.neighbor_station.transformation:
                                #     if trans_info_1['station'] == s:
                        del s.ptc
                        s.ptc = None
                        del s1.ptc
                        s1.ptc = None
                        gc.collect()

                    # for r, t in zip(rotations, translations):
                    #     s.neighbor_station.ptc = deepcopy(neighbor_points)
                    #     # transform the neighbor station
                    #     s.neighbor_station.transform(r, t)
                    #     # z1 = s.neighbor_station.find_ground_z()
                    #     z1 = 0
                    #     # thin slice
                    #     slice_ptc_1 = s.neighbor_station.horizontal_slice(z1 + lower_bound, thickness)
                    #     # create a temporal bundle that consists of only these 2 stations
                    #     local_bundle = BundleStation(stations = [s, s.neightbor_station])
                    #     # run clustering
                    #     local_bundle.
                    #     # overlap
                    #     overlap.append(
                    #         calc_overlap_curve(slice_ptc_0[:,:2], slice_ptc_1[:,:2])
                    #     )
                    # i_best = np.argmax(np.array(overlap))
                    # r_best = rotations[i_best]
                    # t_best = translations[i_best]

                    # # update transformation information
                    # trans_info['rotation'] = r_best
                    # trans_info['translation'] = t_best
                    # trans_info['uncertain_flag'] = False


                    # for _trans in s.neighbor_station.transformation:
                    #     if _trans['station'] == s:
                    #         _trans['rotation'] = _trans['rotation'][i_best]
                    #         _trans['translation'] = _trans['translation'][i_best]
                    #         _trans['uncertain_flag'] = False

                    # print(r_best)
                    # print(t_best)
    
    def get_clusters_crosssection_points(self, z_cut=3, thickness=0.01, step=0.1):
        cross_sections = []
        for tank_label, cyls in self.tanks_info['tanks'].items():
            tank_id = int(tank_label)
            # print('tank', tank_id)
            for c in cyls:
                # print(c.ptc_filepath)
                c.read_points()
                # print(c.points.shape)
                c.transform(c.station.align['rotation'], c.station.align['translation'], update_axis=False, update_z_min=False)
                # if not hasattr(c.station, 'z_ground'):
                #     # z_ground = c.find_ground_z(step=step)
                #     # c.station.z_ground = z_ground
                #     if c.station.ptc is None:
                #         c.station.read_points()
                #     c.station.estimate_ground_z()
                z_ground = c.station.z_ground
                z = z_cut + z_ground
                ind = (z <= c.points[:,2]) & (c.points[:,2] <= z + thickness)
                c.points = c.points[ind,:]
                p = np.c_[c.points, tank_id * np.ones([c.points.shape[0],1])]
                cross_sections.append(p)
                del c.points
                c.points = None
                gc.collect()
        # combine all points from all stations
        cs_points = np.concatenate(cross_sections, axis=0)
        cs_df = pd.DataFrame(cs_points, columns=['x', 'y', 'z', 'tank_id'])
        cs_df['tank_label'] = cs_df.tank_id.apply(lambda i: str(int(i)))
        return cs_df

    def draw_cross_section(self, z_cut=3, thickness=0.01):
        cs_df = self.get_clusters_crosssection_points(z_cut, thickness)

        # center points
        center_df = self.tanks_info['centers_xy']
        # center_df.sort_values('label')
        # center_df.rename(columns={'label': 'tank_label'}, inplace=True)

        # combine
        df = pd.concat([cs_df, center_df])
        df = df.drop(['station', 'cylinder id', 'radius', 'xy'], axis=1)

        fig = draw_thin_slice(df)
        fig.show()
        return fig
    
    def draw_alignment_phase_1b(self, target_tank_only=True):
        df_stations = []
        for ind, c in enumerate(self.target_tank):
            if target_tank_only:
                file_name = None
            else:
                file_name = f'{c.station.name}.pcd'

            c.update_filepath(
                new_dir=self.data_dir / 'target_tank' / 'global' / 'thin_slice', 
                num_keep_level=1, file_name = file_name
            )
            c.read_points()

            _df = pd.DataFrame(c.points, columns=['x','y','z'])
            _df['station_name'] = c.station.name
            _df['station_id'] = c.station.id
            df_stations.append(_df)
        df = pd.concat(df_stations)
        fig = draw_thin_slice(df, color='station_name')
        fig.show()
        return fig

    def calc_align_matrix(self, tank_label, make_vertical=True, align_bottom=True, print_info=True):
        if print_info: print(f'calculating alignment matrix for tank {tank_label}')
        if not tank_label in self.tanks_info['tanks']:
            if print_info: print(f'tank {tank_label} not found')
            return
        for c in self.tanks_info['tanks'][tank_label]:
            if print_info: print(c.station.name, 'cluster', c.id)
            if (c.points is None) or (c.points.shape[0] == 0):
                c.read_points()
            # _c = Cylinder(
            #     points = c.points, radius=c.radius,
            #     position=deepcopy(c.axis.position),
            #     vector=deepcopy(c.axis.vector),
            #     metadata={
            #         'station': Station(read_points=False)
            #     }
            # )
            # _c.station.z_ground = c.station.z_ground
            # _c.station.align = {
            #     'rotation': c.station.align['rotation'],
            #     'translation': c.station.align['translation']
            # }
            # _c.transform(
            #     _c.station.align['rotation'],
            #     _c.station.align['translation'],
            #     update_z_min=False
            # )
            c.transform(
                c.station.align['rotation'],
                c.station.align['translation'],
                update_z_min=False
            )
            if make_vertical:
                c.make_vertical()
            if align_bottom:
                c.z_min = c.station.z_ground
                c.translate_origin(recalc_zmin=True)
            c.align = {
                'rotation': c.station.align['rotation'],
                'translation': c.station.align['translation']
            }
            

    def write_cylinder_segments(self, tank_label, write_dir='.', tank_name=None, surrounding=False, max_radius=50, make_vertical=True, align_bottom=True, recalc_z_min=True, data_type='txt'):
        print(f'writing cylinder {tank_label}')
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
            # src_pts_xyz = src_pts_xyz.transpose()
            # translated_xyz = (rotation@src_pts_xyz) + translation
            # tmp_cyl = Cylinder(
            #     points=c.points,
            #     radius=c.radius,
            #     position=c.axis.position,
            #     vector=c.axis.vector
            # )
            # if hasattr(c, 'z_min'):
            #     tmp_cyl.
            # tmp_cyl = c
            c.transform(rotation, translation)

            if make_vertical:
                c.make_vertical()
            if align_bottom:
                c.translate_origin(recalc_zmin=recalc_z_min)
                # tmp_cyl.translate_base()
            # write
            # file_name = f'tank_{tank_name}_{p.stem}.txt'
            file_name = f'{c.station.name}.{data_type}'
            file_path = os.path.join(write_dir, file_name)
            tmp_c = Cylinder(points=c.points, metadata={'ptc_filepath': file_path})
            tmp_c.write_points(data_type = 'pcd')
            del tmp_c
            # z_min = min(tmp_cyl.points[:,-1])
            # print(f'z_min = {z_min}')
            # if data_type == 'txt':
            #     write_array2txt(
            #         file_path, 
            #         c.points#translated_xyz.transpose()
            #     )

            # del tmp_cyl
            del c.points
            c.points = None
            gc.collect()
    def write_tank(self, tank_label, write_dir='.', read_points=True, data_type='txt'):
        print(f'writing cylinder {tank_label}')
        if not tank_label in self.tanks_info['tanks']:
            print('tank id not found')
            return
        for c in self.tanks_info['tanks'][tank_label]:
            if read_points:
                c.read_points()
            # transform
            c.transform(
                c.align['rotation'],
                c.align['translation']
            )
            # write
            # file_name = f'tank_{tank_name}_{p.stem}.txt'
            file_name = f'{c.station.name}.{data_type}'
            c.write_points(write_dir=write_dir, filename=file_name, data_type = 'pcd')
            del c.points
            c.points = None
            gc.collect()

    def write_all_tanks(self, write_dir='.', tank_name=None, surrounding=False, max_radius=50):
        for tank_label in self.tanks_info['tanks']:
            print(f'Writing tank {tank_label}')
            tank_dir = os.path.join(write_dir, f'tank_{tank_label}')
            if not os.path.exists(tank_dir):
                os.mkdir(tank_dir)
            try:
                self.write_cylinder_segments(
                    tank_label,
                    tank_dir,
                    surrounding=surrounding,
                    max_radius = max_radius
                )
            except Exception as e:
                print(f'error writing tank {tank_label}')
                traceback.print_exc(e)
    
    def determine_pairing_sequence(self, visual=False, ignore_station_ids=[]):
        # read data of slices from all tank segments
        work_dir = self.data_dir / 'target_tank' / 'global' / 'thin_slice'
        df = pd.read_csv(
            work_dir / 'target_tank_phase_1.txt', 
            delimiter=" ", header=0, names=['x', 'y', 'z', 'station']
        )
        # create a graph representing the overlappings between the segments
        G = nx.Graph()
        station_ids = df.station.unique()
        num_stations = len(station_ids)
        for i in range(num_stations):
            if i in ignore_station_ids:
                continue
            xy_i = df[df.station == station_ids[i]][['x', 'y']].values
            for j in range(i+1, num_stations):
                if j in ignore_station_ids:
                    continue
                xy_j = df[df.station == station_ids[j]][['x', 'y']].values
                d = calc_overlap_curve(xy_i, xy_j)
                if d > 0:
                    G.add_edge(i,j, weight=d)
        # the best way to pair is determined by the maximum spanning tree
        mst = nx.algorithms.tree.maximum_spanning_edges(G, algorithm="kruskal", data=False)
        edgelist = list(mst)

        with open(os.path.join(work_dir, 'station_pairs.txt'), 'w') as f:
            json.dump({'pair': edgelist}, f)

        if visual:
            G2 = nx.Graph()
            G2.add_edges_from(edgelist)
            nx.draw(G2, with_labels=True)
        
        return edgelist
    
    def register_complete(
        self,
        max_correspondence_distance_coarse = 0.2, #voxel_size * 15
        max_correspondence_distance_fine1  = 0.01, #voxel_size * 1.5
        max_correspondence_distance_fine2  = 0.005 #voxel_size * 1.5
        ):
        # data_dir = self.data_dir / 'target_tank' / 'global' / 'ground'
        # with open(os.path.join(self.data_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
        #     bundle = pickle.load(f)
        all_cylinders = self.target_tank

        # use radius to classify cylinders into inliners or outliers
        in_cyls, out_cyls = multiway_filter_point_clouds(all_cylinders, radius_offset_max=0.5)

        # read pairing
        with open(self.data_dir / 'target_tank' / 'global' / 'thin_slice' / 'station_pairs.txt', 'r') as f:
            pair_info = json.load(f)
        pair_info = pair_info['pair']

        pcds_down, pcds_full = load_point_clouds(
            in_cyls, 
            file_dir=self.data_dir / 'target_tank' / 'global' / 'ground',
            voxel_size=0.01
        )

        # Pose graph
        print("Full registration ...")
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            pose_graph = full_registration_with_pairs(
                pcds_down, pair_info,
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine1,
                max_correspondence_distance_fine2
            )
        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine2,
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        
        # save
        icp_dir = self.data_dir / 'target_tank' / 'global' / 'ground' / 'icp'
        if not os.path.exists(icp_dir):
            os.mkdir(icp_dir)

        for cyl, pcd, node in zip(in_cyls, pcds_full, pose_graph.nodes):
            # transform
            pcd.transform(node.pose)
            filepath = Path(cyl.ptc_filepath)
            filename = filepath.stem
            o3d.io.write_point_cloud(
                str(icp_dir / f'{filename}.pcd'),
                pcd, print_progress=False
            )
            # transform cylinder axis
            cyl.axis.transform(node.pose[:3, :3], node.pose[:3, -1].reshape([3,1]))
            # write cylinder data
            # [xc, yc, zc, rc, tx, ty, tz]
            cyl_params = np.array(cyl.axis.position.ravel().tolist() + [cyl.radius] + cyl.axis.vector.ravel().tolist())
            with open(icp_dir / f'fit_cylinder_{filename}.pkl', 'wb') as f:
                pickle.dump([[], [], cyl_params], f)

        
        self.registration = {
            'pcds_down': pcds_down,
            'pcds_full': pcds_full,
            'pose_graph': pose_graph
        }
        return pose_graph