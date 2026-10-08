import logging
import networkx as nx
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import gc
import pandas as pd
import plotly.express as px
from glob import glob
import os
import open3d as o3d
import pickle
import json
import time

from ..cylinder_extraction.internal_station import InternalStation, Station
from ..cylinder_alignment.bundle_station import BundleStation
from ..cylinder_alignment.station_alignment import StationAlignment
from ..utils.write_output import write_array2txt

from ..registration.utils import parallel_load_pcd_point_clouds
from ..registration.utils import multiway_full_registration_internal
from ..registration.utils import combine_point_cloud, load_transform_and_write

from ..internal_scans.without_floating_roof.processing import detect_objects
from ..utils.utils import extract_station_ids
from ..utils.write_output import pcd_to_e57

from ..config import PCD2E57_SCRIPT

class InternalBundleStation(BundleStation):
    def __init__(self, stations=[], data_dir=None, rough_align_subdir='rough_alignment', fine_align_subdir='fine_alignment', data_type='e57', with_floating_roof=False, ref_scan=0):
        super().__init__(stations, data_dir)
        self.rough_align_dir = self.data_dir / rough_align_subdir
        self.fine_align_dir = self.data_dir / fine_align_subdir
        self.with_floating_roof = with_floating_roof
        self.ref_scan = ref_scan
        # check if stations are provided
        if stations == []:
            # scan .e57 files and initiate corresponding stations
            source_files = sorted(glob(str(self.data_dir / f'*.{data_type}')))
            # process file names for scan numbers
            names, ids, source_files = extract_station_ids(source_files)
            # initiate one InternalStation object for each scan
            for name, id, fp in zip(names, ids, source_files):
                # file path
                src_file_path = Path(fp)
                # station name
                station_name = name
                # station id
                station_id = id

                print(f'STATION {station_name}')
                
                station = InternalStation(
                    name=station_name,
                    file_path=src_file_path,
                    read_points=False,
                    station_id=station_id
                )
                self.stations.append(station)

    def save_object(self, write_dir=None):
        """write current information of the object to file

        Args:
            write_dir (str, optional): location to write file. 
            Defaults to None, meaning at the same location as data.
        """
        # remove points to save space
        for s in self.stations:
            try:
                s.ptc = None
                for c in s.refined_cylinders:
                    c.points = None
                s.shell.points = None
            except Exception:
                pass
        # remove intermediate results of the rough alignment phase
        for s in self.stations:
            try:
                s.transformation = None
            except Exception:
                pass
            
        try:
            self.registration = {
                'poses': [n.pose for n in self.registration['pose_graph'].nodes]
            }
        except Exception:
            pass
        if not write_dir:
            write_file = self.data_dir / 'bundle.pkl'
        else:
            if isinstance(write_dir, str):
                write_dir = Path(write_dir)
            write_file = write_dir / 'bundle.pkl'
        with open(write_file, 'wb') as f:
            pickle.dump(self, f)
        
    def process(self, func, chunk_size, params=[]):
        """apply parallel processing to all stations

        Args:
            func (func)             : the function that is applied to each station
            chunk_size (numeric)    : number of parallel processes
            params (list, optional) : parameters passing to the handling function. Defaults to [].
        """
        num_stations = len(self.stations)
        for i in range(0, num_stations, chunk_size):
            stations = self.stations[i:i+chunk_size]
            with Pool(chunk_size) as p:
                p.starmap(
                    func,
                    [(s, *params) for s in stations]
                )


    def extract_objects_of_interest(self, z_slices, min_tank_height, cell_size=1, num_point_min=30, distance_to_wall_min=0.2, num_process=4):
        """detect objects for roughly aligning internal tanks without floating roofs

        Args:
            z_slices (list)                         : elevations at which horizontal slies are extracted
            min_tank_height (float)                 : minimum height to identify cells with wall points
            cell_size (int, optional)               : size of cells to cluster points. Defaults to 1.
            num_point_min (int, optional)           : mimimum number of beloing points to filter cells.
                                                      Defaults to 30.
            distance_to_wall_min (float, optional)  : minimum distance to wall of objects. Points within
                                                      this distance belong to wall. Defaults to 0.2.
            num_process (int, optional)             : number of parallel processes. Defaults to 4.
        """
        num_stations = len(self.stations)
        write_dirs = [self.data_dir / 'cylinder_extraction' / s.name for s in self.stations]

        # detect_objects(self.stations[0], min_tank_height, cell_size, z_slices, num_point_min, distance_to_wall_min)
        for i in range(0, num_stations, num_process):
            j = min(i + num_process, num_stations)
            station_pack = self.stations[i:j]
            write_dir_pack = write_dirs[i:j]
            print(f'parallel processing {station_pack}')
            count = len(station_pack)
            with Pool(count) as p:
                p.starmap(
                    detect_objects,
                    zip(
                        station_pack,
                        [min_tank_height] * count,
                        [cell_size] * count,
                        [z_slices] * count,
                        [num_point_min] * count,
                        [distance_to_wall_min] * count,
                        write_dir_pack
                    )
                )

    def align_pairwise_station(self, object='column'):
        """align stations, two at a time

        Args:
            object (str, optional): objects used for rough alignment - column or manway.
                                    Defaults to 'column'.
        """
        # reset transformation info and initialise number of columns
        for s in self.stations:
            s.transformation = []
            s.num_columns = 0

        num_station = len(self.stations)
        alignment = []
        for i in range(num_station):
            st0 = self.stations[i]
            for j in range(i+1, num_station):
                st1 = self.stations[j]
                align = StationAlignment(st0, st1)
                try:
                    align.align_internal(object=object)
                except Exception:
                    logging.warning(f'could not align {st0.name} and {st1.name}')
                alignment.append(align)
        self.pairwise_alignment = alignment

    def align_by_columns(self, ref_station=None, stations=None, indicator='RMS', dynamic=True):
        """roughly align all stations by making used information from the pairwise alignments

        Args:
            ref_station (InternalStation, optional) : a reference station. 
                                                      Defaults to None - station with the most columns.
            stations ([InternalStation,], optional) : list of stations for alignment. Defaults to None.
            indicator (str, optional)               : the metric to justify the alignment of two stations -
                                                      RMS or the number of matching columns. Defaults to 'RMS'.
            dynamic (bool, optional)                : whether to check all possible pairs of stations or only 
                                                      stations in a sequencing order. Defaults to True.
        """
        # indicator: RMS/num_matching_cylinders
        if stations is None:
            stations = self.stations
        # first build a graph connection between aligned stations
        g = nx.Graph()
        if dynamic:
            for s in self.stations:
                if s not in stations:
                    continue
                for t in s.transformation:
                    if t['station'] not in stations:
                        continue
                    if not t['uncertain_flag']:
                        g.add_edge(s, t['station'], weight=t[indicator])
            if indicator == 'num_matching_cylinders':
                mst = nx.algorithms.tree.maximum_spanning_edges(g, algorithm="kruskal", data=False)
                g = nx.Graph(mst)
        else:
            for i in range(len(self.stations) - 1):
                s = self.stations[i]
                s_next = self.stations[i + 1]
                if s not in stations:
                    continue
                for t in s.transformation:
                    if (t['station'] not in stations) or (t['station'].name != s_next.name):
                        continue
                    if not t['uncertain_flag']:
                        g.add_edge(s, t['station'], weight=t[indicator])
        # translate to the reference station
        if not ref_station:
            ref_station = max(stations, key=lambda s:s.num_columns)
        # all tranformations between stations
        trans = self.get_transform_matrices()
        # return value
        info = {
            'ref_station': ref_station,
            'transformation': {}
        }
        # now go through each station
        for s in stations:
            if s is ref_station:
                s.is_aligned = True
                continue
            # shortest path
            if g.has_node(s):
                p = nx.shortest_path(g, s, ref_station, 'weight')
            else:
                print(f'station {s.name} is not connected')
                s.is_aligned = False
                continue
            print(p)
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
            s.update_transform(r, t)
        self.rough_align = info

    def write_aligned(self, write_dir='./', chunk_size=8, slice=True, z_cuts=[3,5], thick_ness=0.02):
        """write (either roughly or fine) aligned stations to disk

        Args:
            write_dir (str, optional): destination directory. Defaults to './'.
            chunk_size (int, optional): number of parallel processes. Defaults to 8.
            slice (bool, optional): if horizontal slices are extracted and written. Defaults to True.
            z_cuts (list, optional): elevations at which horizontal slides are extracted. Defaults to [3,5].
            thick_ness (float, optional): the thickness of slices. Defaults to 0.02.
        """
        if isinstance(write_dir, str):
            write_dir = Path(write_dir)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        # redundant data (which can cause parallel-processing failed)
        for s in self.stations:
            s.transformation = None

        params = [write_dir, slice, z_cuts, thick_ness]
        with Pool(chunk_size) as p:
            slices_all_stations = p.starmap(
                transform_and_write,
                [(s, *params) for s in self.stations]
            )

        if slice:
            # add station index to each slice
            for i, station_slices in enumerate(slices_all_stations):
                for j, slice in enumerate(station_slices):
                    slices_all_stations[i][j] = np.c_[
                        slice,
                        np.ones((slice.shape[0], 1)) * i
                    ]
            figs = []
            for i in range(len(z_cuts)):
                z_slice_points = np.concatenate([s[i] for s in slices_all_stations])
                # write to file
                write_array2txt(
                    write_dir / f'slice_{z_cuts[i]}m.txt',
                    z_slice_points
                )
                # image
                df = pd.DataFrame(z_slice_points, columns=['x', 'y', 'z', 'station_id'])
                df['station name'] = df.station_id.apply(lambda i: self.stations[int(i)].name)
                fig = px.scatter(df, 'x', 'y', color='station name')
                fig.update_layout(
                    width = 800,
                    height = 800,
                    title = f'slices at {z_cuts[i]}m'
                )
                fig.update_yaxes(
                    scaleanchor = "x",
                    scaleratio = 1,
                )
                figs.append(fig)

            with open(write_dir / f'slices.html', 'w') as f:
                for fig in figs:
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn')) # only DIV containing the figure

    def register_complete(self, max_correspondence_distances=[0.2, 0.01], voxel_size=0.01, z_cut=5, pairs=None, debug=False, write_dir=None):
        """fine align all (roughly aligned) stations

        Args:
            max_correspondence_distances (list, optional)   : searching radius for closet points when running icp. 
                                                              Defaults to [0.2, 0.01].
            voxel_size (float, optional)                    : to downsample point clouds before running icp. 
                                                              Defaults to 0.01.
            z_cut (int, optional)                           : horizontal slices for post evaluation. 
                                                              Defaults to 5.
            pairs (list, optional)                          : Pairs for alignment with icp. 
                                                              Defaults to None - use file order.
            debug (bool, optional)                          : whether to save slices for quick evalutions of results. 
                                                              Defaults to False.

        Returns:
            fine_alignment_info (dict)  : station names and corresponding tranformation matrices
            pose_graph                  : open3d object represents all stations and transformation among them
            order                       : the order of all stations in the pose_graph's nodes
        """
        if write_dir is None:
            write_dir = self.rough_align_dir
        self.exc_time = {}
        # read point clouds
        source_files = glob(str(self.rough_align_dir / '*.pcd'))
        _, _, source_files = extract_station_ids(source_files)
        # source_files.pop(1)
        source_files = source_files

        # pcds_down, _ = load_pcd_point_clouds(source_files, voxel_size=voxel_size)
        time_begin = time.time()
        pcds_down, _ = parallel_load_pcd_point_clouds(source_files, keep_original=False, downsample=True, voxel_size=voxel_size, num_proces=3)
        time_end = time.time()
        self.exc_time['parallel_load_pcd_point_clouds'] = time_end - time_begin

        print("Full registration ...")
        max_correspondence_distance_coarse = max_correspondence_distances[0]
        max_correspondence_distance_fine = max_correspondence_distances[1]
        # max_correspondence_distance_fine2  = 0.005#voxel_size * 1.5
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            time_begin = time.time()
            pose_graph, order = multiway_full_registration_internal(
                pcds_down,
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine,
                pairs
            )
            time_end = time.time()
            self.exc_time['multiway_full_registration_internal'] = time_end - time_begin
            # reorder pcds
            pcds_down[:] = [pcds_down[i] for i in order]

        time_begin = time.time()
        # print("Optimizing PoseGraph ...")
        # option = o3d.pipelines.registration.GlobalOptimizationOption(
        #     max_correspondence_distance=max_correspondence_distance_fine,
        #     edge_prune_threshold=0.25,
        #     reference_node=0)
        
        # with o3d.utility.VerbosityContextManager(
        #         o3d.utility.VerbosityLevel.Debug) as cm:
        #     o3d.pipelines.registration.global_optimization(
        #         pose_graph,
        #         o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        #         o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        #         option)
        # time_end = time.time()
        # self.exc_time['optimizing PoseGraph'] = time_end - time_begin
        # save poses
        print(pose_graph)
        
        print('===================')
        print(source_files)
        source_files = np.array(source_files)
        print(source_files)

        poses = []
        for n in pose_graph.nodes:
            print(n)
            print(n.pose)
            poses.append(n.pose)

        # poses = [n.pose for n in pose_graph.nodes]
        print(poses)
        fine_alignment_info = {}
        for i, pose in zip(order, poses):
            print(pose)
            fine_alignment_info[source_files[i]] = pose.tolist()
        print('&&&&&&&&&&&&&&&&&&&&&&')
        with open(write_dir / 'fine_alignment_info.txt', 'w') as f:
            json.dump(fine_alignment_info, f)

        print('===================')
        # for debuging, write a slice
        if debug:
            time_begin = time.time()
            inliner_pcd_combined_down = combine_point_cloud(pcds_down, pose_graph)
            # write a slice
            thickness = 0.01
            ind = (z_cut <= inliner_pcd_combined_down[:,2]) & (inliner_pcd_combined_down[:,2] <= z_cut + thickness)
                
            write_array2txt(
                os.path.join(write_dir, f'icp_downsample_slice_{z_cut}m.txt'),
                inliner_pcd_combined_down[ind,:]
            )
            time_end = time.time()
            self.exc_time['write a downsampling slice'] = time_end - time_begin
        print('=1111111=======')
        self.fine_alignment = fine_alignment_info
        return fine_alignment_info, pose_graph, order

    def transform_and_write_final_icp_to_original_data(self, write_dir=None, ref_scan=0, pcd_2_e57=True):
        """write fine aligned (point cloud) data to disk

        Args:
            write_dir (str, optional)       : the location for storing fine aligned scans
        """

        rough_transform_info_file = os.path.join(self.rough_align_dir,'bundle.pkl')
        fine_transform_info_file  = os.path.join(Path(self.rough_align_dir).parent,'fine_alignment','fine_alignment_info.txt')
        trans_matrices,_,_,info_fine = inverse_transform_matrix(rough_transform_info_file, fine_transform_info_file,ref_scan)
        
        # write dir
        if write_dir is None:
            write_dir = self.rough_align_dir / 'icp'
        elif isinstance(write_dir, str):
            write_dir = Path(write_dir)

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        ind = 0
        for key,value in info_fine.items():
            transform_and_write_data2origin(rough_transform_info_file,trans_matrices,write_dir,key,ind)
            ind = ind + 1

        # args = []
        # ind  = 0
        # for key,value in info_fine.items():
        #     args.append([rough_transform_info_file,trans_matrices,write_dir,key,ind])
        #     ind = ind + 1
        #     # print(ind)
                          
        # with Pool(12) as p:
        #     p.starmap(
        #         transform_and_write_data2origin,
        #         args
        #     )

        if pcd_2_e57:
            pcd_to_e57(write_dir, to_e57_script=PCD2E57_SCRIPT)

    def write_final_icp(self, pose_graph, order, write_dir=None, pcd_2_e57=True):
        """write fine aligned (point cloud) data to disk

        Args:
            pose_graph (open3d PoseGraph)   : pose_graph obtained from running icp
            order (list)                    : maps stations to nodes in pose_graph
            write_dir (str, optional)       : the location for storing fine aligned scans
        """
        source_files = glob(str(self.rough_align_dir / '*.pcd'))
        _, _, source_files = extract_station_ids(source_files)
        # reorder w.r.t the order returned from icp
        source_files = [source_files[i] for i in order]
        # write dir
        if write_dir is None:
            write_dir = self.rough_align_dir / 'icp'
        elif isinstance(write_dir, str):
            write_dir = Path(write_dir)

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        with Pool(12) as p:
            p.starmap(
                load_transform_and_write,
                zip(
                    source_files,
                    [n.pose for n in pose_graph.nodes],
                    [write_dir] * len(source_files)
                )
            )
        if pcd_2_e57:
            pcd_to_e57(write_dir, to_e57_script=PCD2E57_SCRIPT)

    def determine_icp_pairs_by_scanner_position(self, write_dir, close_distance=5, estimate_scanner_position=False):
        """pair stations for running icp. The idea is that close stations are matched up as they
        likely share the most common features/points

        Args:
            write_dir (str)                             : location to save the resulting pairing information
            close_distance (int, optional)              : stations within this distance have high confidence
                                                          in their icp-matching results. Defaults to 5.
            estimate_scanner_position (bool, optional): : whether to automatically estimate scanner positions.
                                                          Defaults to False.
        Returns:
            pairs: list of pairs
        """
        # determine scanner positions after the rough alignment phase
        if not estimate_scanner_position:
            for s in self.stations:
                s.ptc = np.array([[0, 0, 0]])
                    # # # use roughly aligned files
                    # # file_path = str(self.rough_align_dir / f'{s.name}.pcd')
                    # # pcd =  o3d.io.read_point_cloud(file_path)
                    # # s.ptc = np.asarray(pcd.points)
                    # # estimate the position of the scanner
                    # s.ptc = s.estimate_scanner_position().reshape([1,3])
                s.transform(
                    s.align['rotation'],
                    s.align['translation']
                )
                s.scanner_position = s.ptc
        else:
            with Pool(12) as p:
                positions = p.map(
                    _estimate_scanner_position,
                    self.stations
                )
                for s, p in zip(self.stations, positions):
                    s.scanner_position = p
                    
        # create a graph with stations as nodes and their (Euclidean) distances as edges
        g = nx.Graph()
        num_stations = len(self.stations)
        for i in range(num_stations):
            s0 = self.stations[i]
            for j in range(i+1, num_stations):
                s1 = self.stations[j]
                # distance
                dist = np.linalg.norm(s0.scanner_position - s1.scanner_position)
                # add edge
                g.add_edge(s0, s1, weight=dist)
        # find the minimum spanning tree
        mst = nx.algorithms.tree.minimum_spanning_tree(g, algorithm="kruskal")
        # add loop closure edges
        for n in mst.nodes:
            if mst.degree(n) == 1:
                remain_edges = [e for e in g.edges(n, data=True) if not mst.has_edge(e[0], e[1])]
                if len(remain_edges) > 0:
                    closet_edge = min(remain_edges, key=lambda e: e[2]['weight'])
                    mst.add_edge(closet_edge[0], closet_edge[1], **closet_edge[2])
        # get pairs from the MST tree
        ids = [s.id for s in self.stations]
        ext_pairs = []
        for e in nx.edge_bfs(mst, source=self.stations[0]):
            ind_0 = ids.index(e[0].id)
            ind_1 = ids.index(e[1].id)
            if g.edges[e]['weight'] < close_distance:
                is_closure = 1
            else:
                is_closure = 0
            ext_pairs.append([ind_0, ind_1, is_closure])
        # save
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        with open(write_dir / 'pairs.txt', 'w') as f:
            json.dump({'pairs': ext_pairs}, f)
        return ext_pairs

    def create_rough_alingment_folder_and_write_pcds(self):

        # write pcd data to rough alignment folder
        if not os.path.exists(self.rough_align_dir):
            os.mkdir(self.rough_align_dir)

        info = {
            'ref_station': self.stations[0],
            'transformation': {}
        }    

        for s in self.stations:
            s.read_points()
            s.write_points(
                data_type = 'pcd',
                filename = os.path.join(
                    self.rough_align_dir,f'{s.name}.pcd'
                )
            )

            info['transformation'][s] = {
                'station': s,
                'rotation': np.identity(3),
                'translation': np.zeros([3,1])
            }
            
        self.rough_align = info
        self.save_object(write_dir=self.rough_align_dir)    

def transform_and_write(s, write_dir, slice=True, z_cuts=[3,5], thick_ness=0.05):
    s.read_points()
    s.transform(
        s.align['rotation'],
        s.align['translation']
    )
    s.write_points(
        filename=str(write_dir / f'{s.name}.pcd'),
        data_type='pcd'
    )

    if slice:
        slice_points = []
        # ind = np.zeros((s.ptc.shape[0],), dtype=bool)
        # cut one/a afew slice for quick assessment of the quality of the rough-alignment phase
        for z in z_cuts:
            ind = ((s.ptc[:,2] >= z) & (s.ptc[:,2] < z + thick_ness))
            slice_points.append(s.ptc[ind])
    else:
        slice_points = None
    s.ptc=None
    gc.collect()
    return slice_points

def _estimate_scanner_position(s):
    """estimate scanner position

    Args:
        s (InternalStation): an internal station

    Returns:
        (numpy array): coordinates of scanner position
    """
    s.ptc = s.estimate_scanner_position().reshape([1,3])
    s.transform(
        s.align['rotation'],
        s.align['translation']
    )
    return s.ptc

def inverse_transform_matrix(rough_transform_info_file, fine_transform_info_file,sel_ref_scan=0):
    with open(fine_transform_info_file, 'rb') as f:
        info_fine = json.load(f) 
    with open(rough_transform_info_file, 'rb') as f:
        bundle_rough_info = pickle.load(f)
    m_fine = np.array(info_fine[str(Path(rough_transform_info_file).parent / f'{bundle_rough_info.stations[sel_ref_scan].id}.pcd')])
    m_rough = np.identity(4)
    m_rough[:3,:3] = bundle_rough_info.stations[sel_ref_scan].align['rotation']
    m_rough[:3,-1:] = bundle_rough_info.stations[sel_ref_scan].align['translation']
    inverse_matrix = np.linalg.inv(m_fine) @ np.linalg.inv(m_rough)

    trans_matrices = []
    for key,value in info_fine.items(): #range(len(bundle_rough_info.stations)):
        #value = np.array(info_fine[str(Path(rough_transform_info_file).parent / f'{bundle_rough_info.stations[j].id}.pcd')])
        trans_matrices.append(inverse_matrix @ value) 
    
    return trans_matrices, inverse_matrix, bundle_rough_info, info_fine

def transform_and_write_data2origin(rough_transform_info_file,trans_matrices,write_dir,key,ind):

    s = Station(file_path=str(Path(rough_transform_info_file).parent / f'{Path(key).stem}.pcd'))
    s.transform(trans_matrices[ind][:3,:3],trans_matrices[ind][:3,-1:])
    s.write_points(filename=os.path.join(write_dir,f'{Path(key).stem}.pcd'),data_type='pcd')
    ind = ind + 1