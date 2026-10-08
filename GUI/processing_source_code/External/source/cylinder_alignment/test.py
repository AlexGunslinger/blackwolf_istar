import pickle
import os
import sys
sys.path.append('source')

from .bundle_station import BundleStation
from ..cylinder_extraction.station import Station
from cylinder_extraction.station import Station
from cylinder_alignment.station_alignment import StationAlignment

if __name__=='__main__':
    # read stations
    data_dir = '/Users/tinnguyen/BlackBox/PointRegistration/data/External Scans'
    station_names = ['4', '5', '6', '7', '8', '9']
    stations = []
    for sn in station_names:
        pkl_fp = os.path.join(
            data_dir,
            f'00{sn}-output',
            f'00{sn}.pkl'
        )
        with open(pkl_fp, 'rb') as f:
            st = pickle.load(f)
            stations.append(st)
    # pair align
    num_station = len(stations)
    alignment = []
    for i in range(num_station):
        for j in range(i+1, num_station):
            st0 = stations[i]
            st1 = stations[j]
            align = StationAlignment(st0, st1)
            align.align(delta_radius_min=1)
            alignment.append(align)
    # all stations        
    bundle = BundleStation(stations)
    # 
    bundle.rough_alignment()
    # print(bundle.rough_align)
    # target cylinder
    bundle.guess_target_cylinder()
    bundle.get_all_xy_intersect()
    pass