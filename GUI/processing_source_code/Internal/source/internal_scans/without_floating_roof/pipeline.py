# %%

import sys
sys.path.append('D:\Dev\Point-Cloud-Registration\source')

import time
import traceback
import gc
from pathlib import Path

from cylinder_alignment.internal_bundle_station import InternalBundleStation

from parameters import parameters as all_job_parameters

if __name__=="__main__":

    # %%
    all_job_parameters.keys()

    # %%
    tank_name = sys.argv[-1]#'JB-582 90ft API Column'
    params = all_job_parameters[tank_name]

    # %%
    data_dir = params['data_dir']
    z_slices = params['z_slices']
    pack_num = params['num_process']

    execuated_time = {}
    execuated_time[tank_name] = {}

    sys.stdout = open(Path(data_dir) / 'log.txt', 'w')

    # %% [markdown]
    # # Extraction

    # %%
    time_begin = time.time()
    try:
        print('===========================')
        print(tank_name)
        # initialise a bundle comprising all scans
        bundle = InternalBundleStation(data_dir=data_dir)
        # extract columns
        bundle.extract_objects_of_interest(
            z_slices=z_slices,
            min_tank_height=params['min_tank_height'],
            distance_to_wall_min=params['distance_to_wall_min'],
            num_process=pack_num
        )
        
    except Exception as exc:
        print(f'Failed handling scan {tank_name}')
        traceback.print_exc(exc)
    time_end = time.time()
    lasted = time_end - time_begin
    print(f'processing took {lasted} seconds')
    execuated_time[tank_name]['extraction'] = lasted
    with open(bundle.data_dir / 'processing_time.txt', 'w') as f:
        f.write(f'extraction: {lasted} seconds\n')

    # %% [markdown]
    # # Rough alignment

    # %%
    time_begin = time.time()
    try:
        print('===========================')
        print(tank_name)
        # READ EXTRACTED CYLINDERS DATA
        bundle = InternalBundleStation(data_dir=data_dir)
        summary_df = bundle.read_extracted_cylinder_data(file_types=['e57'])
        # PAIRWISE STATION ALIGNMENT
        bundle.align_pairwise_station()
        # align all stations
        bundle.align_by_columns()
        # write roughly aligned stations
        bundle.write_aligned(bundle.data_dir / 'rough_alignment', z_cuts=z_slices[:2])
        # ############################        
        time_pairwise = time.time()
    except Exception as exc:
        print(f'===== FAILED HANDLING SCAN {tank_name} =====')
        traceback.print_exc(exc)
    time_end = time.time()
    lasted = time_end - time_begin
    print(f'processing took {lasted} seconds')
    execuated_time[tank_name]['rough_alignment'] = lasted
    with open(bundle.data_dir / 'processing_time.txt', 'a') as f:
        f.write(f'rough_alignment: {lasted} seconds\n')
            

    # %% [markdown]
    # # ICP

    # %%
    time_begin = time.time()
    try:
        print('===========================')
        print(tank_name)
        # init bundle
        bundle = InternalBundleStation(data_dir=data_dir)
        # icp
        bundle.register_complete(
            max_correspondence_distances=[
                params['icp']['max_correspondence_distance_coarse'],
                params['icp']['max_correspondence_distance_fine']            
            ],
            voxel_size=params['icp']['voxel_size']
        )     
        time_pairwise = time.time()
    except Exception as exc:
        print(f'===== FAILED HANDLING SCAN {tank_name} =====')
        traceback.print_exc(exc)
    time_end = time.time()
    lasted = time_end - time_begin
    print(f'processing took {lasted} seconds')
    execuated_time[tank_name]['fine_alignment'] = lasted
    with open(bundle.data_dir / 'processing_time.txt', 'a') as f:
        f.write(f'fine_alignment: {lasted} seconds\n')
            