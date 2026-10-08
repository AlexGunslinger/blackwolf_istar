import sys
sys.path.append('D:\Dev\Point-Cloud-Registration\source')

import os
import time
import numpy as np
from itertools import chain
import pickle
from glob import glob
from pathlib import Path
from copy import deepcopy
import networkx as nx
import json

from multiprocessing import Pool

from cylinder_extraction.station import Station
from cylinder_extraction.utils import detect_objects

from cylinder_alignment.station_alignment import StationAlignment
from cylinder_alignment.bundle_station import BundleStation
from cylinder_alignment.utils import extract_target_cylinder, draw_thin_slice

from registration.utils import multiway_load_point_clouds as load_point_clouds

from utils.write_output import write_array2txt
from utils.math_methods import project_point_onto_plane, calc_overlap_curve, cart2pol

if __name__=="__main__":

    ###############################################################################
    # CYLINDER EXTRACTION
    ###############################################################################

    field = 'JB-448 26ft EXT Faro'
    pack_num = 8 # number of parallel cores
    execuated_time = {}

    time_begin = time.time()
    try:
        print('===========================')
        print(field)
        data_dir = Path(f'D:\REGISTRATION DATA\External/{field}')
        filename_template = '{}.e57'
        ptc_col_names =  ["x","y","z"]

        source_files = glob(str(data_dir / '*.e57')) + glob(str(data_dir / '*.pcd'))
        station_names = []
        stations = []
        for ind, fp in enumerate(source_files):
            if 'roof' in fp.lower():
                continue
            # station name
            station_name = Path(fp).stem
            # station id
            station_id = ind

            print(f'STATION {station_name}')
            station = Station(
                name=station_name, file_path=fp, read_points=False, station_id=station_id
            )
            stations.append(station)

        # pack_num = 8
        num_stations = len(stations)
        for i in range(0, num_stations, pack_num):
            j = min(i+pack_num, num_stations)
            pack = stations[i:j]
            with Pool(pack_num) as p:
                pack = p.map(
                    detect_objects,
                    pack
                )
            for station in pack:
                station.clusters = None
                # write station metadata
                out_dir = os.path.join(data_dir,station.name)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                # save station without points
                station.ptc = None
                out_fp = os.path.join(out_dir, f'{station.name}.pkl')
                with open(out_fp, 'wb') as f:
                    pickle.dump(station, f)
    except Exception as e:
        print(f'Failed handling scan {field}')
        print(e)
    time_end = time.time()
    lasted = time_end - time_begin
    print(f'processing took {lasted} seconds')
    execuated_time[field] = lasted
        

    ###############################################################################
    # ROUGH ALIGNMENT
    ###############################################################################
    execuated_time = {}  
    time_begin = time.time()
    try:
        print('===========================')
        print(field)
        data_dir = Path(f'D:/REGISTRATION DATA/External/{field}')
        # READ EXTRACTED CYLINDERS DATA
        bundle = BundleStation(data_dir=data_dir)
        summary_df = bundle.read_extracted_cylinder_data(file_types=['e57', 'pcd'])
        # ############################
        # there are some specilities with some tanks that need to be set manually
        if field == 'JB-531 40ft EXT':
            # roof stations 001, 002, 003
            valid_stations = []
            for s in bundle.stations:
                if '001' in s.name or '002' in s.name or '003' in s.name:
                    continue
                valid_stations.append(s)
            bundle.stations = valid_stations

        if field == 'JB-567 27ft EXT Leica/tilted stations':
            # roof stations 001, 002, 003
            valid_stations = []
            for s in bundle.stations:
                if '001' in s.name or '002' in s.name or '003' in s.name:
                    continue
                valid_stations.append(s)
            bundle.stations = valid_stations

        if field == 'JB-566 120ft EXT':
            # not-understandable station data: setup 030
            valid_stations = []
            for s in bundle.stations:
                if '030' in s.name:
                    continue
                valid_stations.append(s)
            bundle.stations = valid_stations
        # px.scatter(
        #     summary_df[summary_df.radius < 50], y='radius', x='station', 
        #     color='station', hover_data=summary_df.columns
        # )
        # PAIRWISE STATION ALIGNMENT
        bundle.align_pairwise_station()
        time_pairwise = time.time()
        # Save
        bundle.remove_points()
        with open(os.path.join(data_dir, 'rough_alignment.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        # load
        with open(os.path.join(data_dir, 'rough_alignment.pkl'), 'rb') as f:
            bundle = pickle.load(f)
        # uncertain stations
        uncertain_stations = bundle.find_uncertain_stations()
        if len(uncertain_stations) > 0:
            print('uncertain stations')
            print(uncertain_stations)
            raise ValueError('Uncertain stations exist')
        # bundle.handle_uncertain(
        #     neighbor_info={
        #         '009': '008',
        #         '016': '015',
        #         '018': '017'
        #     }
        # )
        # bring all stations to a global coordinate
        bundle.align_all_stations_by_axis()
        # CLUSTER TANKS
        bundle.cluster_cylinder()
        time_cluster = time.time()
        # save
        bundle.remove_points()
        with open(os.path.join(data_dir, 'globalization_center_phase_1.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        with open(os.path.join(data_dir, 'globalization_center_phase_1.pkl'), 'rb') as f:
            bundle = pickle.load(f)
        # VISUAL INSPECTION
        # fig = bundle.draw_cross_section()
        # fig.show()
        # TARGET TANK
        bundle.identify_target_tank()
        print('target tank', bundle.target_tank_id)
        # bundle.set_target_tank(tank_id=3)
        # Put all fragments of the target tank in one place. Include points at their foot
        bundle.write_target_tank(
            write_dir=bundle.data_dir / 'target_tank' / 'local' / 'ground',
            load_data=True, extend_ground=True, transform=False)
        # update location of all the segments of the target tank: load original ground-extended data
        for c in bundle.target_tank:
            c.update_filepath(new_dir=bundle.data_dir / 'target_tank' / 'local' / 'ground', num_keep_level=1, file_name=f'{Path(c.ptc_filepath).stem}.pcd')
            print(c.ptc_filepath)

        # alignment transformation matrices
        bundle.calc_align_matrix(f'{bundle.target_tank_id}')
        # save
        bundle.remove_points()
        with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
            bundle = pickle.load(f)

        # write the transformed segments
        bundle.write_target_tank(
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'ground', 
            load_data=True, transform=True, extend_ground=False, num_process=5)

        # THIN SLICE
        bundle.cut_horizontal(
            mode='station_cylinders', z_cut=3, thickness=0.01, 
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'thin_slice')
        bundle.cut_horizontal(
            mode='target_tank', z_cut=3, thickness=0.01,
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'thin_slice')
    except Exception as exc:
        print(f'===== FAILED HANDLING SCAN {field} =====')
        print(exc)
    time_end = time.time()
    lasted = time_end - time_begin
    print(f'processing took {lasted} seconds')
    execuated_time[field] = lasted
    del bundle
    gc.collect()


    ###############################################################################
    # ROUGH ALIGNMENT
    ###############################################################################
    # READ data from rough alignment phase 1 and get a thin slice
    data_dir = Path(f'D:/REGISTRATION DATA/External/{field}')
    with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    edgelist = bundle.determine_pairing_sequence(visual=True, ignore_station_ids=[])

    ###############################################################################
    # ROUGH ALIGNMENT
    ###############################################################################
    root_dir = Path(f'D:/REGISTRATION DATA/External/{field}')
    data_dir = root_dir / 'target_tank' / 'global' / 'ground'
    with open(os.path.join(root_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    # registration
    bundle.register_complete()
    # write test data
    inliner_pcd_combined_down = combine_point_cloud(
        deepcopy(bundle.registration.pcds_down), bundle.registration.pose_graph)
    write_array2txt(
        os.path.join(data_dir, f'inliers_combined_with_ids_down.txt'),
        inliner_pcd_combined_down
    )
    # write a slice
    l = 3.0
    thickness = 0.01
    ind = (l <= inliner_pcd_combined_down[:,2]) & (inliner_pcd_combined_down[:,2] <= l + thickness)
    write_array2txt(
        os.path.join(data_dir, f'slice_3_down_optimize_.txt'),
        inliner_pcd_combined_down[ind,:]
    )