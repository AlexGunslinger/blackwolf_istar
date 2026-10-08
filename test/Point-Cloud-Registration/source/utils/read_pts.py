# -*- coding: utf-8 -*-
"""
The function is to read the point cloud
@author: Linh Truong-Hong
Created by 11 Jan 2021

"""

import os
import numpy as np
import pandas as pd

# In[1- Read data]
def read_input_data(path, file_name, read_e57_raw=True):
    """This function is to read point cloud
    Read file las format and return pandas DataFrame
    Parameters:
    ------------------------
        path                : Path
        file_name           : File name

    Return:
    ------------------------
        ptc                 : Point cloud in Dataframe

    Demo:
        path_file =
    ------------------------
        path = "d:\\Point Cloud Processing\\Acernis\\Test data\\"
        file_name = "test_1.e57"
        file_name = "test_1.pcd"
        file_name = "test_1.las"
        file_name = "test_1.ply"
        file_name = "test_1.xyz"
        file_name = "test_2.xyz"

    """
    # Check the file is available
    __, file_ext = os.path.splitext(file_name)
    path_file = os.path.join(path, file_name)
    if not os.path.exists(path_file):
        raise ValueError("File {:s} is not available.".format(file_name))
        return
    # compare the extension
    supported_format = ['.e57', '.las', '.laz', '.pcd', '.asc', '.pts', '.txt', '.csv', '.xyz', '.ply']
    if not(file_ext.lower() in supported_format):
        raise NameError("Format {:s} does not supported.".format(file_ext))
    elif (file_ext.lower()=='.e57'):
        #Read E57
        pts = read_e57(path_file, read_e57_raw)

    elif (file_ext.lower()=='.las') | (file_ext.lower()=='.laz'):
        # Read las
        pts = read_lasfile(path_file)
    elif (file_ext.lower()=='.pcd'):
        # Read pcd
        pts = read_pcd(path_file)
    elif (file_ext.lower()=='.ply'):
        # Read ply
        pts = read_ply(path_file)
    else:
        # Read ASCII
        pts = read_ascii(path_file)
    return pts
# In[Read bin file]
def read_ascii(path_file):
    """Read file bin format and return pandas DataFrame
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Call library
    import numpy as np
#    import re
    # Read data
    with open(path_file, 'rb') as f:
        for count, ln in enumerate(f):
            ln = f.readline().strip().decode()

            ln = ln.replace(" ", "").replace(".", "").replace("-", "")
            if ln.isdecimal(): break # Header contain character

        # Read data
        if count == 0:
            ptc_data = pd.read_csv(path_file, delimiter=' ', header = None)
        else:
            ptc_data = pd.read_csv(path_file, delimiter=' ', header = count, skiprows = count)

        # Read the header
        columns_names = np.full(len(ptc_data.columns), ' ', np.dtype('U100'))
        if count == 0: # No header available
            ii = 0
            for i, col_dtype in enumerate(ptc_data.dtypes):
                if col_dtype == np.float64:
                    if i == 0:
                        columns_names[i] = 'x'
                    elif i == 1:
                        columns_names[i] = 'y'

                    elif i == 2:
                        columns_names[i] = 'z'
                    elif (3 <= i)&(i <= 6):
                        columns_names[i] = 'intensity'
                    else:
                        columns_names[i] = 'Scale {:d}'.format(i)
                elif col_dtype == np.int64:

                    if (3 <= i)&(i <= 6):
                        if ii == 0:
                            columns_names[i] = 'R'
                            ii += 1
                        elif ii == 1:
                            columns_names[i] = 'G'
                            ii += 1
                        elif ii == 2:
                            columns_names[i] = 'B'
                            ii += 1
                        else:
                            columns_names[i] = 'Scale {:d}'.format(i)
                            ii += 1
                    else:
                        columns_names[i] = 'Scale {:d}'.format(i)
            # Assign the column name
            ptc_data.columns = columns_names

    # Filter fields of the point cloud
    pts_attribute = ['x', 'y', 'z', 'intensity', 'R', 'red', 'G', 'green', 'B', 'blue', 'normal_x', 'normal_y','normal_z']
    flag = np.full(len(ptc_data.columns), False, dtype = bool)
    for count, column in enumerate(ptc_data.columns):
        if column in pts_attribute:
            flag[count] = True
    # Drop non necessary column
    ptc_data = ptc_data.drop(columns = ptc_data.columns[~flag], axis=1)

    # Return
    return ptc_data

# In[Read pcd file]
def read_ply(path_file):
    """Read file las format and return pandas DataFrame
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Call library
    from pyntcloud.io import read_ply
    # Read points
    ptc_data = read_ply(path_file)
    # Extract points only
    ptc_data = ptc_data["points"]

    # Filter fields of the point cloud
    ptc_data = pts_attribute(ptc_data)

    # Return
    return ptc_data
# In[Read pcd file]
def read_lasfile(path_file):
    """Read file las format and return pandas DataFrame
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Call library
    from pyntcloud.io import read_las
    # Read points
    ptc_data = read_las(path_file)#, backend="laspy")
    # Extract points only
    ptc_data = ptc_data["points"]

    # Filter fields of the point cloud
    ptc_data = pts_attribute(ptc_data)

    # Return
    return ptc_data

# In[Read pcd file]
def read_pcd(path_file):
    """Read file pcd format and return pandas DataFrame
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Call library
    from pyntcloud.io import read_pcd
    # Read points
    ptc_data = read_pcd(path_file)
    # Extract points only
    ptc_data = ptc_data["points"]

    # Filter fields of the point cloud
    ptc_data = pts_attribute(ptc_data)

    # Return
    return ptc_data

# In[Read E57 file]
def read_e57(path_file, read_raw=True):
    """Read file E57 and return pandas DataFrame
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Call libs
    import pye57
    # Read the number of the scan

    e57 = pye57.E57(path_file)
    num_scans = e57.scan_count

    # Load data
    ptc_data = pd.DataFrame()
    xyz_header = ["cartesianX", "cartesianY", "cartesianX"]
    for count in range(num_scans):
        if read_raw:
            scan_station = e57.read_scan_raw(count)
            # Check valid header
            if ("cartesianInvalidState" in scan_station.keys()):
                valid = ~scan_station["cartesianInvalidState"].astype("?")
                for field in scan_station:
                    scan_station[field] = scan_station[field][valid]
                del scan_station["cartesianInvalidState"], valid
            # Get header of
            header = e57.get_header(count)
            header_field = header.scan_fields

            # Read raw data of each station with a local coordinate
            station_ptc = pd.DataFrame()
            for field in scan_station.keys():
                station_ptc[field] = scan_station[field]
            del field

            # Convert to the global coordinate
            if ("pose" in header_field):
                ptc_local_xyz = station_ptc[xyz_header].values
                ptc_global_xyz = e57.to_global(ptc_local_xyz, header.rotation, header.translation)
                station_ptc[xyz_header] = ptc_global_xyz
                del ptc_local_xyz, ptc_global_xyz
            del scan_station, header, header_field
        else:
            print(count)
            station_ptc = pd.DataFrame(e57.read_scan(count, ignore_missing_fields=True))
        # Combinate all stations
        ptc_data = pd.concat([ptc_data, station_ptc], axis = 0, ignore_index = False)
        del station_ptc
    del count, num_scans, xyz_header

    # Get point attribute
    ptc_data = pts_attribute(ptc_data)

    # Return
    return ptc_data

# In[Filter point attribution]
def pts_attribute(ptc_data):
    """Filter main attribution of points for computation
    Parameters:
    ------------------------
        path_file           : File including path

    Return:
    ------------------------
        ptc_data            : Dataframe

    Demo:
        path_file =
    ------------------------

    """
    # Filter fields of the point cloud
    pts_attribute = ['x', 'cartesianX', 'y', 'cartesianY', 'z', 'cartesianZ', 'intensity',
                     'red', 'R', 'colorRed', 'green', 'G', 'colorGreen', 'blue', 'B', 'colorBlue',
                     'normal_x', 'normal_y','normal_z', 'pt_src_id']
    flag = np.full(len(ptc_data.columns), False, dtype = bool)
    for count, column in enumerate(ptc_data.columns):
        if column in pts_attribute:
            flag[count] = True
    # Drop non necessary column
    ptc_data = ptc_data.drop(columns = ptc_data.columns[~flag], axis=1)

    # Rename the columns
    columns_names = ptc_data.columns.values.tolist()
    if 'cartesianX' in columns_names: ptc_data = ptc_data.rename(columns={"cartesianX": "x"})
    if 'cartesianY' in columns_names: ptc_data = ptc_data.rename(columns={"cartesianY": "y"})
    if 'cartesianZ' in columns_names: ptc_data = ptc_data.rename(columns={"cartesianZ": "z"})
    ptc_data = ptc_data.rename(columns={col_name: "R" for col_name in columns_names if col_name in ['red', 'R', 'colorRed']})
    ptc_data = ptc_data.rename(columns={col_name: "G" for col_name in columns_names if col_name in ['green', 'G', 'colorGreen']})
    ptc_data = ptc_data.rename(columns={col_name: "B" for col_name in columns_names if col_name in ['blue', 'B', 'colorBlue']})

    return ptc_data
