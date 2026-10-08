import os
import time
from pathlib import Path
import traceback
import gc
import pickle
from enum import IntEnum
import sys

import time
import traceback
import gc
from pathlib import Path

from ...cylinder_alignment.internal_bundle_station import InternalBundleStation

from .parameters import default_parameters, update_params

class STAGE(IntEnum):
    CYLINDER_EXTRACTION = 0
    ROUGH_ALIGNMENT     = 1
    ICP                 = 2

class InternalNoFloatRoofProject():
    def __init__(self, data_dir, custom_params=None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        # parameters
        self.parameters = default_parameters
        if custom_params is not None:
            update_params(self.parameters, custom_params)

    def run(self, phase, mode='all', pre_aligned=False):
        running_phase = phase
        if running_phase == STAGE.CYLINDER_EXTRACTION:
            self._run_phase_cylinder_extraction()
            if mode == 'all':
                running_phase += 1
        
        if running_phase == STAGE.ROUGH_ALIGNMENT:
            self._run_phase_rough_alignment(pre_aligned=pre_aligned)
            if mode == 'all':
                running_phase += 1
        
        if running_phase == STAGE.ICP:
            self._run_phase_fine_alignment()
            if mode == 'all':
                running_phase += 1
            
            

    def _run_phase_cylinder_extraction(self):
        # Extraction
        # TOP
        print('====================================================================')
        print('CYLINDER EXTRACTION')
        execuated_time = {'cylinder extraction': {}}
        params = self.parameters

        data_dir = self.data_dir
        pack_num = params['num_process']
        min_height = params['min_tank_height']
        z_slices = params['z_slices']

        # output data directory
        out_dir = data_dir / 'cylinder_extraction'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_cylinder_extraction.txt', 'w')

        try:
            print('===========================')
            print(f'Cylinder extraction')
            time_begin = time.time()
            # initialise a bundle comprising all scans
            # initialise a bundle comprising all scans
            bundle = InternalBundleStation(data_dir=self.data_dir)
            # extract columns
            bundle.extract_objects_of_interest(
                z_slices=z_slices,
                min_tank_height=min_height,
                distance_to_wall_min=params['distance_to_wall_min'],
                num_process=pack_num
            )
            
            time_end = time.time()
            execuated_time['cylinder extraction']['extract_objects_of_interest'] = time_end - time_begin
        except Exception as exc:
            print(f'Failed extracting features')
            traceback.print_exc(exc)
            return False
        lasted = sum(execuated_time['cylinder extraction'].values())
        print(f'processing took {lasted} seconds')
        execuated_time['cylinder extraction']['total'] = lasted
        with open(log_dir / 'processing_time_feature_extraction.txt', 'w') as f:
            for desc, exc_time in execuated_time['cylinder extraction'].items():
                f.write(f'cylinder extraction - {desc}: {exc_time} seconds\n')
        return True

    def _run_phase_rough_alignment(self, pre_aligned=False):
        # Rough alignment
        print('====================================================================')
        print('ROUGH ALIGNMENT')
        execuated_time = {'rough alignment': {}}
        params = self.parameters
        
        data_dir = self.data_dir
        z_slices = params['z_slices']

        # output data directory
        out_dir = data_dir / 'rough_alignment'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_rough_alignment.txt', 'w')

        try:
            print('===========================')
            print(f'Rough alignment')
            time_begin = time.time()
            # READ EXTRACTED CYLINDERS DATA
            bundle = InternalBundleStation(data_dir=data_dir)
            summary_df = bundle.read_extracted_cylinder_data(cyl_data_dir=data_dir / 'cylinder_extraction', file_types=['e57'])
            time_end = time.time()
            execuated_time['rough alignment']['read_extracted_cylinder_data'] = time_end - time_begin
            if not pre_aligned:
                # PAIRWISE STATION ALIGNMENT
                time_begin = time.time()
                bundle.align_pairwise_station(object=params['target_object'])
                time_end = time.time()
                execuated_time['rough alignment']['align_pairwise_station'] = time_end - time_begin
                # align all stations
                time_begin = time.time()
                bundle.align_by_columns()
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
            # write roughly aligned stations
            time_begin = time.time()
            bundle.write_aligned(bundle.data_dir / 'rough_alignment', z_cuts=z_slices[:2])
            time_end = time.time()
            execuated_time['rough alignment']['write_aligned'] = time_end - time_begin
        except Exception as exc:
            print(f'Rough alignment failed')
            traceback.print_exc(exc)
            return False
        lasted = sum(execuated_time['rough alignment'].values())
        print(f'processing took {lasted} seconds')
        execuated_time['rough alignment']['total'] = lasted
        with open(log_dir / 'processing_time_rough_alignment.txt', 'w') as f:
            for desc, exc_time in execuated_time['rough alignment'].items():
                f.write(f'rough alignment - {desc}: {exc_time} seconds\n')
        del bundle
        gc.collect()
        return True

    def _run_phase_fine_alignment(self):
        # Fine alignment ICP
        print('====================================================================')
        print('FINE ALIGNMENT')
        execuated_time = {'fine alignment': {}}
        params = self.parameters
        
        data_dir = self.data_dir

        # output data directory
        out_dir = data_dir / 'fine_alignment'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # log
        log_dir = data_dir / 'log'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        sys.stdout = open(log_dir / 'log_fine_alignment.txt', 'w')
        
        icp_params = params['icp']
        rough_align_data_dir = data_dir / 'rough_alignment'
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
            params['z_slices'][1],
            pairs=ext_pairs,
            debug=True,
            write_dir=out_dir
        )
        # write final registered pcds
        bundle.write_final_icp(pose_graph, order, out_dir)
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
                f.write(f'fine alignment - {desc}: {exc_time} seconds\n')
            # program_end = time.time()
            # f.write(f'\n[{tank_part}] - Total processing time: {program_end - program_start} seconds\n')
        # clean up
        # del pcds_down, pcds_full, pcd_combined
        # gc.collect()