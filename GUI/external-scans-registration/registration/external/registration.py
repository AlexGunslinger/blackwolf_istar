import sys
import traceback

import os
import time
import pickle
from glob import glob
from pathlib import Path
from copy import deepcopy

from multiprocessing import Pool

from External.source.cylinder_extraction.station import Station
from External.source.cylinder_extraction.utils import detect_objects

from External.source.cylinder_alignment.bundle_station import BundleStation

from External.source.registration.utils import combine_point_cloud

from External.source.utils.write_output import write_array2txt

from enum import IntEnum

import streamlit as st

###############################################################################
# CYLINDER EXTRACTION
###############################################################################
def pcr_cylinder_extraction(data_dir, pack_num=6):
    import logging
    time_begin = time.time()
    try:
        # print('Initiating a dedicated object for each station.')
        source_files = glob(str(data_dir / '*.e57')) + glob(str(data_dir / '*.pcd'))
        stations = []
        for ind, fp in enumerate(source_files):
            if 'roof' in fp.lower():
                continue
            if '006' in fp.lower():
                continue
            # station name
            station_name = Path(fp).stem
            # station id
            station_id = ind

            # print(f'Creat an object for station {station_name}')
            station = Station(
                name=station_name, file_path=fp, read_points=False, station_id=station_id
            )
            stations.append(station)
        # return

        # pack_num = 8
        num_stations = len(stations)
        for i in range(0, num_stations, pack_num):
            j = min(i+pack_num, num_stations)
            pack = stations[i:j]
            print(f'==> Parallel processing stations:')
            for s in pack:
                print(f'{s.name}')
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
            yield pack
    except Exception as e:
        print(f'Cylinder extraction encountered an error')
        traceback.print_exc(e)
    time_end = time.time()
    execuated_time = time_end - time_begin
    print('===========================================')
    print(f'==> Processing took {execuated_time} seconds')

###############################################################################
# ROUGH ALIGNMENT
###############################################################################
def pcr_rough_alignment(data_dir):
    execuated_time = {}  
    time_begin = time.time()
    try:
        print('==> Reading extracted cylinder data')
        # READ EXTRACTED CYLINDERS DATA
        bundle = BundleStation(data_dir=data_dir)
        summary_df = bundle.read_extracted_cylinder_data(file_types=['e57', 'pcd'], print_info=False)
        # PAIRWISE STATION ALIGNMENT
        print('==> Aligning all pairs of stations')
        # return
        bundle.align_pairwise_station()
        time_pairwise = time.time()
        # Save
        print('==> Saving pair-wise station alignment results')
        bundle.remove_points()
        with open(os.path.join(data_dir, 'rough_alignment.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        # load
        with open(os.path.join(data_dir, 'rough_alignment.pkl'), 'rb') as f:
            bundle = pickle.load(f)
        # uncertain stations
        print('==> Identifying uncertain stations')
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
        print('==> Aligning all stations')
        bundle.align_all_stations_by_axis(print_info=False)
        # CLUSTER TANKS
        print('==> Grouping cylinders to corresponding stations')
        bundle.cluster_cylinder()
        time_cluster = time.time()
        # save
        print('==> Saving all-station alignment results')
        bundle.remove_points()
        with open(os.path.join(data_dir, 'globalization_center_phase_1.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        with open(os.path.join(data_dir, 'globalization_center_phase_1.pkl'), 'rb') as f:
            bundle = pickle.load(f)
        # VISUAL INSPECTION
        # fig = bundle.draw_cross_section()
        # fig.show()
        # TARGET TANK
        print('==> Identifying the target tank')
        bundle.identify_target_tank(print_info=False)
        # print('target tank', bundle.target_tank_id)
        # bundle.set_target_tank(tank_id=3)
        # Put all fragments of the target tank in one place. Include points at their foot
        print('==> Extending data points at the foots of all cylinders of the target tank')
        bundle.write_target_tank(
            write_dir=bundle.data_dir / 'target_tank' / 'local' / 'ground',
            load_data=True, extend_ground=True, transform=False)
        # update location of all the segments of the target tank: load original ground-extended data
        for c in bundle.target_tank:
            c.update_filepath(new_dir=bundle.data_dir / 'target_tank' / 'local' / 'ground', num_keep_level=1, file_name=f'{Path(c.ptc_filepath).stem}.pcd')
            # print(c.ptc_filepath)

        # alignment transformation matrices
        print('==> Calculating transforming matrices for all stations')
        bundle.calc_align_matrix(f'{bundle.target_tank_id}', print_info=False)
        # save
        print('==> Saving (globally) transforming matrices')
        bundle.remove_points()
        with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
            bundle = pickle.load(f)

        # write the transformed segments
        print('==> Transforming and saving all cylinders of the target tank')
        bundle.write_target_tank(
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'ground', 
            load_data=True, transform=True, extend_ground=False, num_process=5)

        # THIN SLICE
        print('==> Cutting a cross-section from each station for quick inspections')
        bundle.cut_horizontal(
            mode='station_cylinders', z_cut=3, thickness=0.01, 
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'thin_slice', print_info=False)
        bundle.cut_horizontal(
            mode='target_tank', z_cut=3, thickness=0.01,
            write_dir=bundle.data_dir / 'target_tank' / 'global' / 'thin_slice', print_info=False)
        print('==> Rough alignment job finished successfully')
    except Exception as exc:
        print(f'Cylinder extraction encountered an error\n')
        print(exc)
        traceback.print_exc(exc)
        sys.exit()
    time_end = time.time()
    execuated_time = time_end - time_begin
    print(f'==> Processing took {execuated_time} seconds')


###############################################################################
# FINE ALIGNMENT
###############################################################################
def pcr_fine_alignment(data_dir):
    time_begin = time.time()
    # READ data from rough alignment phase 1 and get a thin slice
    with open(os.path.join(data_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    bundle.determine_pairing_sequence(visual=True, ignore_station_ids=[])

    # ICP
    root_dir = Path(data_dir)
    data_dir = root_dir / 'target_tank' / 'global' / 'ground'
    with open(os.path.join(root_dir, 'globalization_axis_phase_1.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    # registration
    bundle.register_complete()
    # write test data
    inliner_pcd_combined_down = combine_point_cloud(
        deepcopy(bundle.registration['pcds_down']), bundle.registration['pose_graph'])
    write_array2txt(
        os.path.join(data_dir, f'inliers_combined_with_ids_down.txt'),
        inliner_pcd_combined_down
    )
    # write a slice
    l = 3.0
    thickness = 0.01
    ind = (l <= inliner_pcd_combined_down[:,2]) & (inliner_pcd_combined_down[:,2] <= l + thickness)
    write_array2txt(
        os.path.join(data_dir, f'icp_slice_down.txt'),
        inliner_pcd_combined_down[ind,:]
    )
    time_end = time.time()
    execuated_time = time_end - time_begin
    print(f'processing took {execuated_time} seconds')