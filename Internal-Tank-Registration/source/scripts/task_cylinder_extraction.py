import sys
sys.path.append('D:\Dev\Point-Cloud-Registration\source')

import os
import time
import numpy as np
from itertools import chain
import pickle
from glob import glob
from pathlib import Path

from cylinder_extraction.station import Station
from cylinder_alignment.station_alignment import StationAlignment
from utils.write_output import write_array2txt

if __name__=="__main__":

    # data_dir = 'D:/REGISTRATION DATA/External/JB-448 26ft EXT'
    data_dir = Path('D:/REGISTRATION DATA/External/JB-448 26ft EXT')
    filename_template = '{}.e57'
    ptc_col_names =  ["x","y","z"]

    source_files = glob(str(data_dir / '*.e57'))

    for fp in source_files:
        fn = Path(fp).stem
        print(f'STATION {fn}')
        # station id
        sn = fn.split('_')[-1]
        ptc_fn = filename_template.format(sn)
        print(ptc_fn)
        station = Station(
            name=sn, file_path=fp
        )
        # station.write_points()
        station.extract_cylinder()
        out_dir = os.path.join(data_dir,station.name + '-output')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # save station without points
        station.ptc = None
        out_fp = os.path.join(out_dir, f'{station.name}.pkl')
        with open(out_fp, 'wb') as f:
            pickle.dump(station, f)