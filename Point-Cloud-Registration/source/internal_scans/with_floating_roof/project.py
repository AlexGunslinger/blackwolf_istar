import os
import time
from pathlib import Path
import traceback
import gc
import pickle
from enum import IntEnum
import sys

from .parameters import default_parameters, parameters, update_params

from ...cylinder_alignment.internal_bundle_station import InternalBundleStation
from .processing import standardize_shell_and_find_columns

class STAGE(IntEnum):
    CYLINDER_EXTRACTION = 0
    ROUGH_ALIGNMENT     = 1
    ICP                 = 2

class InternalFloatRoofProject():
    def __init__(self, data_dir, custom_params=None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        # parameters
        self.parameters = default_parameters
        if custom_params is not None:
            update_params(self.parameters, custom_params)

    def run(self, parts, phase, mode='all', pre_aligned=False):
        job_status = []
        for tank_part in parts:
            running_phase = phase
            if running_phase == STAGE.CYLINDER_EXTRACTION:
                success = self._run_phase_cylinder_extraction(tank_part)
                if not success:
                    job_status.append(False)
                    continue
                if mode == 'all':
                    running_phase += 1
            
            if running_phase == STAGE.ROUGH_ALIGNMENT:
                success = self._run_phase_rough_alignment(tank_part, pre_aligned=pre_aligned)
                if not success:
                    job_status.append(False)
                    continue
                if mode == 'all':
                    running_phase += 1
            
            if running_phase == STAGE.ICP:
                success = self._run_phase_fine_alignment(tank_part)
                if not success:
                    job_status.append(False)
                    continue
                if mode == 'all':
                    running_phase += 1
            
            job_status.append(True)
        return job_status
            
            

    def _run_phase_cylinder_extraction(self, tank_part):
        # Extraction
        # TOP
        print('====================================================================')
        print('CYLINDER EXTRACTION')
        execuated_time = {'cylinder extraction': {}}
        params = self.parameters

        part_data_dir = self.data_dir / tank_part
        pack_num = params[tank_part]['num_process']
        min_height = params[tank_part]['min_tank_height']
        z_slices = params[tank_part]['z_slices']

        # output data directory
        out_dir = part_data_dir / 'cylinder_extraction'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = part_data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_cylinder_extraction.txt', 'w')

        try:
            print('===========================')
            print(f'Cylinder extraction: {tank_part} part')

            # initialise a bundle comprising all scans
            bundle = InternalBundleStation(
                stations=[],
                data_dir=part_data_dir,
                with_floating_roof=True)
            if len(bundle.stations) == 0:
                print(f'Error: no station found')
                return False
            time_begin = time.time()
            bundle.process(
                standardize_shell_and_find_columns, chunk_size=pack_num, 
                params=[min_height, False, z_slices, 0.2, 10, out_dir]
            )
            time_end = time.time()
            execuated_time['cylinder extraction']['standardize_shell_and_find_columns'] = time_end - time_begin
        except Exception as exc:
            print(f'Failed extracting features')
            traceback.print_exc(exc)
            return False
        lasted = sum(execuated_time['cylinder extraction'].values())
        print(f'processing took {lasted} seconds')
        execuated_time['cylinder extraction']['total'] = lasted
        with open(log_dir / 'processing_time_feature_extraction.txt', 'w') as f:
            for desc, exc_time in execuated_time['cylinder extraction'].items():
                f.write(f'[{tank_part}] - cylinder extraction - {desc}: {exc_time} seconds\n')
        return True

    def _run_phase_rough_alignment(self, tank_part, pre_aligned=False):
        # Rough alignment
        print('====================================================================')
        print('ROUGH ALIGNMENT')
        execuated_time = {'rough alignment': {}}
        params = self.parameters
        
        part_data_dir = self.data_dir / tank_part
        min_height = params[tank_part]['min_tank_height']
        z_slices = params[tank_part]['z_slices']
        chunk_size = params[tank_part]['num_process']
        
        # output data directory
        out_dir = part_data_dir / 'rough_alignment'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = part_data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_rough_alignment.txt', 'w')

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
            bundle.read_extracted_cylinder_data(cyl_data_dir=part_data_dir / 'cylinder_extraction')
            time_end = time.time()
            execuated_time['rough alignment']['read_extracted_cylinder_data'] = time_end - time_begin
            # ############################
            # px.scatter(
            #     summary_df[summary_df.radius < 50], y='radius', x='station', 
            #     color='station', hover_data=summary_df.columns
            # )
            # ############################
            if not pre_aligned:
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
            else:
                import numpy as np
                info = {
                    'ref_station': bundle.stations[0],
                    'transformation': {}
                }
                for s in bundle.stations:
                    info['transformation'][s] = {
                        'station': s,
                        'rotation': np.identity(3),
                        'translation': np.zeros([3,1])
                    }
                bundle.rough_align = info
            print('saving bundle')
            time_begin = time.time()
            bundle.save_object(write_dir=out_dir)
            time_end = time.time()
            execuated_time['rough alignment']['save_object'] = time_end - time_begin
            # with open(part_data_dir / 'bundle.pkl', 'rb') as f:
            #     bundle = pickle.load(f)

            # ############################
            # write roughly aligned stations
            print('write roughly aligned stations')
            time_begin = time.time()
            # write_dir = self.data_dir / f'rough_alignment_{tank_part}'
            # if not os.path.exists(write_dir):
            #     os.makedirs(write_dir)
            bundle.write_aligned(out_dir, z_cuts=z_slices, chunk_size=chunk_size)
            time_end = time.time()
            execuated_time['rough alignment']['write_aligned'] = time_end - time_begin
            # ############################        
            time_pairwise = time.time()
        except Exception as exc:
            print(f'Rough alignment failed')
            traceback.print_exc(exc)
            return False
        lasted = sum(execuated_time['rough alignment'].values())
        print(f'processing took {lasted} seconds')
        execuated_time['rough alignment']['total'] = lasted
        with open(log_dir / 'processing_time_rough_alignment.txt', 'w') as f:
            for desc, exc_time in execuated_time['rough alignment'].items():
                f.write(f'[{tank_part}] - rough alignment - {desc}: {exc_time} seconds\n')
        del bundle
        gc.collect()
        return True

    def _run_phase_fine_alignment(self, tank_part):
        # Fine alignment ICP
        print('====================================================================')
        print('FINE ALIGNMENT')
        execuated_time = {'fine alignment': {}}
        params = self.parameters
        part_data_dir = self.data_dir / tank_part

        out_dir = part_data_dir / 'fine_alignment'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = part_data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_fine_alignment.txt', 'w')
        
        icp_params = params[tank_part]['icp']
        rough_align_data_dir = part_data_dir / 'rough_alignment'
        # rough_align_data_dir = self.data_dir / f'rough_alignment_{tank_part}'
        
        # load rough alignment data
        with open(rough_align_data_dir / 'bundle.pkl', 'rb') as f:
            bundle = pickle.load(f)
        # bundle = InternalBundleStation(data_dir=part_data_dir)
        bundle.rough_align_dir = rough_align_data_dir

        # pairing
        if icp_params['auto_pair']:
            ext_pairs = bundle.determine_icp_pairs_by_scanner_position(
                out_dir,
                estimate_scanner_position=icp_params['estimate_scanner_position'])
        else:
            num_stations = len(bundle.stations)
            ext_pairs = []
            for i in range(num_stations-1):
                ext_pairs.append([i, i+1, 0])
            ext_pairs.append([num_stations-1, 0, 1])
            # write_dir = rough_align_data_dir
            # if not os.path.exists(write_dir):
            #     os.mkdir(write_dir)
            import json
            with open(out_dir / 'pairs.txt', 'w') as f:
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
            pairs=ext_pairs,
            debug=True,
            write_dir=out_dir
        )
        # write final registered pcds
        bundle.write_final_icp(pose_graph, order, out_dir, pcd_2_e57=True)
        # for evaluation
        for s in bundle.stations:
            # write cylinder data
            # [xc, yc, zc, rc, tx, ty, tz]
            cyl_params = [0, 0, 0, s.shell.radius, 0, 0, 1]
            with open(out_dir / f'fit_cylinder_{s.name}.pkl', 'wb') as f:
                pickle.dump([[], [], cyl_params], f)

        execuated_time['fine alignment'] = bundle.exc_time
        lasted = sum(execuated_time['fine alignment'].values())
        print(f'processing took {lasted} seconds')
        execuated_time['fine alignment']['total'] = lasted
        with open(log_dir / 'processing_time_fine_alignment.txt', 'w') as f:
            for desc, exc_time in execuated_time['fine alignment'].items():
                f.write(f'[{tank_part}] - fine alignment - {desc}: {exc_time} seconds\n')
            # program_end = time.time()
            # f.write(f'\n[{tank_part}] - Total processing time: {program_end - program_start} seconds\n')
        # clean up
        # del pcds_down, pcds_full, pcd_combined
        # gc.collect()