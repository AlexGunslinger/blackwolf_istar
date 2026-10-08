import sys
sys.path.append('/Users/tinnguyen/BlackBox/PointRegistration/source')

import pickle
import os

from cylinder_extraction.station import Station
from cylinder_extraction.cylinder import Cylinder


from utils.write_output import write_array2txt

if __name__=="__main__":
    station_name = '063'
    cyl_ind = 1
    cluster_id = 26
    with open(f'/Users/tinnguyen/BlackBox/PointRegistration/data/JB-483/{station_name}-output/{station_name}.pkl', 'rb') as f:
        station = pickle.load(f)
    cylinder = station.cylinders[cyl_ind]
    cylinder.ptc_filepath = f'/Users/tinnguyen/BlackBox/PointRegistration/data/JB-483/{station_name}-output/{station_name} cluster_{cluster_id}.txt'
    cylinder.read_points()
    cylinder.estimate_top_surface()
    #
    write_array2txt(
        '/Users/tinnguyen/BlackBox/PointRegistration/data/JB-483/065-output/065_cluster_7_high_points.txt',
        # cylinder.all_top_ptc
        # cylinder.top_surface['ptc']
        # cylinder.points[cylinder.top_surface['grid_ptc_ids'],:]
        cylinder.points[cylinder.all_top_ptc_ids,:]
        # cylinder.points[cylinder.shell_grids['ptc_ids'][73],:]
    )
    pass