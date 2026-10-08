import sys
sys.path.append('D:\Dev\Point-Cloud-Registration\source')

import os
import time
from pathlib import Path
import traceback
import gc
import pickle
from enum import IntEnum

from ...cylinder_alignment.internal_bundle_station import InternalBundleStation
from .processing import standardize_shell_and_find_columns

from .parameters import default_parameters, parameters, update_params


class STAGE(IntEnum):
    CYLINDER_EXTRACTION = 0
    ROUGH_ALIGNMENT     = 1
    PAIR_DETERMINATION  = 2
    ICP                 = 3

if __name__=="__main__":
    # %%
    tank_name = sys.argv[-3]
    request_parts = sys.argv[-2].split()
    stage = getattr(STAGE, sys.argv[-1])
    print(tank_name, request_parts, stage)
    print('===========================')
    print(tank_name)
    params = default_parameters
    update_params(params, parameters[tank_name])
    data_dir = Path(params['data_dir'])
    
    execuated_time = {}
    for tank_part in request_parts: #['top', 'bottom']:
        program_start = time.time()
        print('====================================================================')
        print(tank_part)
        print('====================================================================')
        part_data_dir = Path(params['data_dir']) / tank_part
        # log
        sys.stdout = open(part_data_dir / 'log.txt', 'w')
        # processing time
        with open(part_data_dir / 'processing_time.txt', 'w') as f:
            pass
        ###################################
        if stage <= STAGE.CYLINDER_EXTRACTION:
            # Extraction
            # TOP
            print('====================================================================')
            print('FEATURE EXTRACTION')
            execuated_time['cylinder extraction'] = {}

            part_data_dir = Path(params['data_dir']) / tank_part
            pack_num = params[tank_part]['num_process']
            min_height = params[tank_part]['min_tank_height']
            z_slices = params[tank_part]['z_slices']
            try:
                print('===========================')
                print(f'Cylinder extraction: {tank_part} part')

                # initialise a bundle comprising all scans
                bundle = InternalBundleStation(
                    stations=[],
                    data_dir=part_data_dir,
                    with_floating_roof=True)
                if len(bundle.stations) == 0:
                    print(f'Error: no station found for {tank_part} part')
                    continue
                time_begin = time.time()
                bundle.process(
                    standardize_shell_and_find_columns, chunk_size=pack_num, 
                    params=[min_height, False, z_slices, 0.2, 10]
                )
                time_end = time.time()
                execuated_time['cylinder extraction']['standardize_shell_and_find_columns'] = time_end - time_begin
            except Exception as e:
                print(f'Failed extracting features in tank {tank_name}')
                traceback.print_exc(exc)
                continue
            lasted = sum(execuated_time['cylinder extraction'].values())
            print(f'processing took {lasted} seconds')
            execuated_time['cylinder extraction']['total'] = lasted
            with open(part_data_dir / 'processing_time.txt', 'a') as f:
                for desc, exc_time in execuated_time['cylinder extraction'].items():
                    f.write(f'[{tank_part}] - cylinder extraction - {desc}: {exc_time} seconds\n')

            
        ###################################
        if stage <= STAGE.ROUGH_ALIGNMENT:
            # Rough alignment
            # TOP
            print('====================================================================')
            print('ROUGH ALIGNMENT')
            execuated_time['rough alignment'] = {}
            
            part_data_dir = Path(params['data_dir']) / tank_part
            min_height = params[tank_part]['min_tank_height']
            z_slices = params[tank_part]['z_slices']
            chunk_size = params[tank_part]['num_process']

            try:
                print('===========================')
                print(f'Rough alignment: {tank_part} part')
                # READ EXTRACTED CYLINDERS DATA
                # initialise a bundle comprising all scans
                bundle = InternalBundleStation(
                    stations=[],
                    data_dir=part_data_dir,
                    with_floating_roof=True)
                time_begin = time.time()
                bundle.read_extracted_cylinder_data()
                time_end = time.time()
                execuated_time['rough alignment']['read_extracted_cylinder_data'] = time_end - time_begin
                # ############################
                # px.scatter(
                #     summary_df[summary_df.radius < 50], y='radius', x='station', 
                #     color='station', hover_data=summary_df.columns
                # )
                # ############################
                # PAIRWISE STATION ALIGNMENT
                print('align pairwise')
                time_begin = time.time()
                bundle.align_pairwise_station()
                time_end = time.time()
                execuated_time['rough alignment']['align_pairwise_station'] = time_end - time_begin
                # ############################
                # align all stations
                print('align all')
                time_begin = time.time()
                bundle.align_by_columns(indicator='num_matching_cylinders', dynamic=False)
                time_end = time.time()
                execuated_time['rough alignment']['align_by_columns'] = time_end - time_begin
                print('saving bundle')
                time_begin = time.time()
                bundle.save_object()
                time_end = time.time()
                execuated_time['rough alignment']['save_object'] = time_end - time_begin
                # with open(part_data_dir / 'bundle.pkl', 'rb') as f:
                #     bundle = pickle.load(f)

                # ############################
                # write roughly aligned stations
                print('write roughly aligned stations')
                time_begin = time.time()
                write_dir = data_dir / f'rough_alignment_{tank_part}'
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                bundle.write_aligned(write_dir, z_cuts=z_slices, chunk_size=chunk_size)
                time_end = time.time()
                execuated_time['rough alignment']['write_aligned'] = time_end - time_begin
                # ############################        
                time_pairwise = time.time()
            except Exception as exc:
                print(f'Failed roughly align tank {tank_name}')
                traceback.print_exc(exc)
                continue
            lasted = sum(execuated_time['rough alignment'].values())
            print(f'processing took {lasted} seconds')
            execuated_time['rough alignment']['total'] = lasted
            with open(part_data_dir / 'processing_time.txt', 'a') as f:
                for desc, exc_time in execuated_time['rough alignment'].items():
                    f.write(f'[{tank_part}] - rough alignment - {desc}: {exc_time} seconds\n')
            del bundle
            gc.collect()

         
        ###################################
        if stage <= STAGE.ICP:
            # Fine alignment ICP
            print('====================================================================')
            print('FINE ALIGNMENT')
            execuated_time['fine alignment'] = {}
            
            icp_params = params[tank_part]['icp']
            rough_align_data_dir = data_dir / f'rough_alignment_{tank_part}'
            
            part_data_dir = Path(params['data_dir']) / tank_part
            # load rough alignment data
            with open(part_data_dir / 'bundle.pkl', 'rb') as f:
                bundle = pickle.load(f)
            # bundle = InternalBundleStation(data_dir=part_data_dir)
            bundle.rough_align_dir = rough_align_data_dir

            # pairing
            if icp_params['auto_pair']:
                bundle.determine_icp_pairs_by_scanner_position(
                    rough_align_data_dir,
                    estimate_scanner_position=icp_params['estimate_scanner_position'])
            else:
                num_stations = len(bundle.stations)
                ext_pairs = []
                for i in range(num_stations-1):
                    ext_pairs.append([i, i+1, 0])
                ext_pairs.append([num_stations-1, 0, 1])
                write_dir = rough_align_data_dir
                if not os.path.exists(write_dir):
                    os.mkdir(write_dir)
                import json
                with open(write_dir / 'pairs.txt', 'w') as f:
                    json.dump({'pairs': ext_pairs}, f)

            # pcds_down, pcds_full = load_pcd_point_clouds(source_files, voxel_size=icp_params['voxel_size'])

            # # for pcd in pcds_down:
            # #     p = np.asarray(pcd.points)
            # #     ind = (p[:,2] > 4) & (p[:,2] < 6)
            # #     pcd.points = o3d.utility.Vector3dVector(p[ind])
            # #     n = np.asanyarray(pcd.normals)
            # #     pcd.normals = o3d.utility.Vector3dVector(n[ind])
            
            # # #### Pose graph
            print("Full registration ...")
            max_correspondence_distance_coarse = icp_params['max_correspondence_distance_coarse']#voxel_size * 15
            max_correspondence_distance_fine  = icp_params['max_correspondence_distance_fine']#voxel_size * 1.5
            # # max_correspondence_distance_fine2  = 0.005#voxel_size * 1.5
            _, pose_graph, order = bundle.register_complete(
                [max_correspondence_distance_coarse, max_correspondence_distance_fine],
                icp_params['voxel_size'],
                params[tank_part]['z_slices'][1],
                use_pairs=True,
                debug=True
            )
            # write final registered pcds
            bundle.write_final_icp(pose_graph, order)
            execuated_time['fine alignment'] = bundle.exc_time
            lasted = sum(execuated_time['fine alignment'].values())
            print(f'processing took {lasted} seconds')
            execuated_time['fine alignment']['total'] = lasted
            with open(part_data_dir / 'processing_time.txt', 'a') as f:
                for desc, exc_time in execuated_time['fine alignment'].items():
                    f.write(f'[{tank_part}] - fine alignment - {desc}: {exc_time} seconds\n')
                program_end = time.time()
                f.write(f'\n[{tank_part}] - Total processing time: {program_end - program_start} seconds\n')
            # clean up
            # del pcds_down, pcds_full, pcd_combined
            # gc.collect()