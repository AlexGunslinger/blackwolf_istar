# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:58:44 2021

@author: LinhTruongHong
"""

import numpy as np
import os
from pathlib import Path
from glob import glob
from multiprocessing import Pool
import open3d as o3d

def write_txt(path_file, data, fmt = "%.3f", delimiter = " "):
    """ Write data to txt file
    Parameters:
    ------------------------
        path_file                : Path
        data                     : File name
    
    Return:
    ------------------------
        txt file                 : Point cloud in Dataframe
    
    Demo:
    """
    np.savetxt(path_file, data, fmt = fmt, delimiter = delimiter)

def to_str(batch, fmt):
    """This function converts a numpy array to string before being written to file. It is used in the function write_array2txt below

    Args:
        batch (numpy array): a numpy array
        fmt ([]): format of each row

    Returns:
        [str]: the formatted string
    """
    # print(batch, fmt)
    if isinstance(fmt, list):
        fmt_row = ' '.join(fmt)
    else:
        fmt_row = fmt
    fmt_all = '\n'.join([fmt_row]*batch.shape[0])
    return fmt_all % tuple(batch.ravel())

def write_array2txt(path_file, data, fmt='%.3f', delimiter=' ', newline='\n', batch_size=100000, parallel=True, num_process=4):
    """ This function is to write an array to txt file
    Parameters:
    ------------------------
            path_file               : Path + filename
            data                    : 1D or 2D array_like
            fmt                     : str or sequence of strs, optional
                                      A single format (%10.5f), a sequence of formats, or a
                                      multi-format string, e.g. 'Iteration %d -- %10.5f', in which
                                      case 'delimiter' is ignored. For complex 'data', the legal options
                                      for `fmt` are:
                                          * a single specifier, `fmt='%.4e'`, resulting in numbers formatted
                                          like `' (%s+%sj)' % (fmt, fmt)`
                                          * a full string specifying every real and imaginary part, e.g.
                                          `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns
                                          * a list of specifiers, one per column - in this case, the real
                                          and imaginary part must have separate specifiers,
                                          e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
          delimiter                 : str, optional. String or character separating columns.
          newline                   : str, optional. String or character separating lines.
    
    Return:
    ------------------------
        txt file                 : Point cloud in Dataframe
    
    Demo:
    """
    # Check input array
    data = np.asarray(data)
    
    # Handle 1-dimensional arrays
    if data.ndim == 0 or data.ndim > 2:
        raise ValueError("Array with {:d}D does not support. Only 1D or 2D array".format(data.ndim))
    elif data.ndim == 1:
        # Common case -- 1d array of numbers
        if data.dtype.names is None:
            data = np.atleast_2d(data).T
            ncol = 1
        # Complex dtype -- each field indicates a separate column
        else:
            ncol = len(data.dtype.descr)
    else:
        ncol = data.shape[1]

    iscomplex_data = np.iscomplexobj(data)
    
    # Get format
    if type(fmt) in (list, tuple):
        if len(fmt) != ncol:
            raise AttributeError('Shape of fmt {:s} is incompatible with an array columns'.format(str(fmt)))
        format = np.compat(delimiter).join(map(np.compat, fmt))
    elif isinstance(fmt, str):
        n_fmt_chars = fmt.count('%')
        error = ValueError('String {:s} shows formats incompatible with an array'.format(fmt))
        if n_fmt_chars == 1:
            if iscomplex_data:
                fmt_imag = fmt.replace(r'%.', '%+.')
                fmt = ['(%s%sj)' % (fmt, fmt_imag),] * ncol
            else:
                fmt = [fmt, ] * ncol
            format = delimiter.join(fmt)
        elif iscomplex_data and n_fmt_chars != (2 * ncol):
            raise error
        elif ((not iscomplex_data) and n_fmt_chars != ncol):
            raise error
        else:
            format = fmt
    else:
        raise ValueError('Invalid format %r' % (fmt,))

    if iscomplex_data:
        # create an extended array with real and imaginary values in separate columns
        real = data.real
        imag = data.imag
        data = np.stack([real, imag]).reshape(2,-1).transpose().reshape(-1,2*imag.shape[1])

    # Write array to txt file
    with open(path_file, 'w') as f:
        if parallel:
            with Pool(num_process) as p:
                parts = np.split(data, range(1,data.shape[0], batch_size))
                for text in p.starmap(to_str, zip(parts, [format]*len(parts))):
                    f.write(text + '\n')
        else:
            f.write(to_str(data, format))

def write_array2pcd(file_path, data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(
        file_path,
        pcd, print_progress=True
    )

def pcd_to_e57(in_dir, out_dir=None, to_e57_script='toE57converter.bat'):
    if isinstance(in_dir, str):
        in_dir = Path(in_dir)
    # scan for all .pcd files
    pcd_files = glob(os.path.join(in_dir / '*.pcd'))
    # put quotations around them
    string = '\" \"'.join(pcd_files)
    # construct a command string
    command = f'{to_e57_script} \"{string}\"'
    # execute
    os.system(command)
    # move all .e57 files to destination folder
    if out_dir is None:
        out_dir = in_dir / 'e57'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # scan all .e57 files
    e57_files = glob(str((in_dir / '*.e57')))
    for e57_file in e57_files:
        e57_file = Path(e57_file)
        new_path = in_dir / 'e57' / e57_file.name
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(
            e57_file,
            new_path
        )
# In[]
def get_path_file(path_dir, file_name):
    """This function is to get file path
    Parameters:
    ------------------------
        path                            : path
        file_name                       : file name with extension

    Return:
    ------------------------
        path_file                       : path_file without extension
    """
    # Call libs
    import os
    # Remove extention
    supported_format = ['.e57', '.las', '.laz', '.pcd', '.asc', '.pts', '.txt', '.csv', '.xyz', '.ply']
    # Find extension
    for count, ext_format in enumerate(supported_format):
        if ext_format in file_name: break
    # check path
    if path_dir is None:
        # get a current directory
        cwd_path = os.getcwd()
        # Create temp direction
        new_dir = "temp"
        path_dir = os.path.join(cwd_path, new_dir) 
        # Check if it is availbale or not
        if not os.path.exists(path_dir):
            os.mkdir(path_dir, mode=0o777)
    # Create path_file
    
    path_file = os.path.join(path_dir, file_name.replace(supported_format[count], ''))
    return path_file
    