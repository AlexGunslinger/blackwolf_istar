import pickle
import gc
import os
import logging
from pathlib import Path
import traceback

from ...utils.write_output import write_array2txt

def standardize_shell(station, min_height, out_dir=None, cleanup=True):
    """" centralise the corresponding cylinder at the Oz axis, 
    and move the bottom (floor) level at Oxy (i.e. z = 0)

    Args:
        station (InternalStation)   : an internal scan
        min_height (float)          : the minimum height for a cell of points to be 
                                      considered as a shell-wall-related cell
        out_dir (str, optional)     : location for storing outputs
        cleanup (bool, optional)    : whether to delete points from station object after processing. 
                                      Defaults to True.

    Returns:
        (bool): if all steps finish successfully
    """
    # for storing station metadata to disk
    if out_dir is None:
        out_dir = station.file_path.parent / station.name
    elif isinstance(out_dir, str):
        out_dir = Path(out_dir)
        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # log messages
    log_file = out_dir / 'log.txt'
    logging.basicConfig(
        filename=str(out_dir / log_file), 
        filemode='w',
        level=logging.DEBUG
    )
    logging.info('==================')
    logging.info(f'parallel processing {station.name} - part 1'.upper())
    # load station points if not yet done
    logging.info('read points')
    if station.ptc is None:
        station.read_points()
    # main processing part
    try:
        # clustering station point cloud
        # default settings
        # "cell_size": 1.0,
        # "min_ptc": 100,
        # "min_tank_height": 5.,
        # "point_gap": 0.2,
        # "min_length": 5.0,
        logging.info('clustering')
        station.clustering(
            output_dir=out_dir,
            cluster_params={
                "min_tank_height": min_height,
                'cell_size': 0.5,
                'min_length': 0.5 # the same as cell_size
            }
        )
        # identify the shell wall
        logging.info('identify the wall')
        station.identify_shell_cluster(quick_fit=False)
        # # estimate the ground z
        # logging.info('estimate ground z')
        station.estimate_ground_z()
        # straight out the cylinder
        logging.info('make vertical')
        station.make_vertical(update_ground=False)
        # estimate the ground z
        logging.info('estimate ground z')
        # move the center axis to coincide with Oz, also make the bottom center the same as the origin of the coordinate system
        logging.info('centralise')
        station.estimate_ground_z()
        station.translate_origin()
        
        # recalculate z_ground
        station.z_ground = station.estimate_ground_z()

        # calculate height
        station.estimate_top_z()
        station.estimate_height()

        # save information about this part
        station.standardize = {
            'align': station.align
        }

        try:
            station.check_invalid_station()
            if not station.valid:
                write_array2txt(
                    out_dir / f'{station.name}_error_checking.txt', 
                    station.suspicious_points, 
                    parallel=False
                )
                logging.warning('check data for potential error')
        except Exception:
            logging.error('checking station validity encountered error')

        if cleanup:
        # save station without points
            logging.info('save object')
            station.save_object(out_dir)

            # del station.ptc
            logging.info('clean up')
            station.ptc = None

            gc.collect()

    except Exception as exc:
        logging.info(f'error in {station.name}')
        msg = traceback.format_exc()
        logging.error(msg)
        return False
    
    logging.info('returning')
    return True

def find_columns(station, z_slices=[1,2,3], thickness=0.05, num_pts_min=30, transform=True, reload_station=True, out_dir=None):
    """ find columns (or alike objects) from each station scan (for roughly align different scans)

    Args:
        station (InternalStation)       : an internal scan
        z_slices (list, optional)       : elevations at which horizontal slices are taken. 
                                          Defaults to [1,2,3].
        thickness (float, optional)     : thicknes of slices to take. Defaults to 0.05.
        num_pts_min (int, optional)     : the minimum number of points that a cluster must have 
                                          to be considered as a slice of a column. Defaults to 30.
        transform (bool, optional)      : whether to do a transformation of scan's points. 
                                          Defaults to True.
        reload_station (bool, optional) : overwrite the input station with one from disk (which is
                                          resulted from applying the standardize_shell function).
                                          Defaults to True.
        out_dir (str, optional)         : location for storing outputs

    Returns:
        (bool): if all steps finish successfully
    """
    # write station metadata
    if out_dir is None:
        out_dir = station.file_path.parent / station.name
    elif isinstance(out_dir, str):
        out_dir = Path(out_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    log_file = out_dir / 'log.txt'
    import logging
    logging.basicConfig(
        filename=str(out_dir / log_file), 
        level=logging.DEBUG
    )

    logging.info('==================')
    logging.info(f'parallel processing {station.name} part 2 - find columns'.upper())
    
    # load result from the first part
    if reload_station:
        logging.info('load results from part 1')
        write_file = Path(out_dir) / f'{station.name}.pkl'
        with open(write_file, 'rb') as f:
            station = pickle.load(f)

    # load station points if not yet done
    if station.ptc is None:
        logging.info('read points')
        station.read_points()
    # main processing part
    try:
        # transform (to standardize)
        if transform:
            logging.info('transform (to standardize)')
            station.transform(
                ** station.standardize['align']
                # station.align['rotation'],
                # station.align['translation']
            )
            station.estimate_ground_z()
        # detect columns' centers
        logging.info('find supporting columns')
        station.find_columns(z_cuts=z_slices, thickness=thickness, num_pts_min=num_pts_min, write_dir=out_dir / 'slice')

        # save station without points
        logging.info('save object')
        station.save_object(out_dir)

        # del station.ptc
        logging.info('clean up')
        station.ptc = None

        gc.collect()

    except Exception as exc:
        logging.info(f'error in {station.name}')
        msg = traceback.format_exc()
        logging.error(msg)
        return False
    
    logging.info('returning')
    return True

def standardize_shell_and_find_columns(station, min_height, floating_roof=False, z_slices=[1,2,3], thickness=0.05, num_pts_min=30, out_dir=None):
    # make cylinder vertical and centralised at Oz, move bottom (floor) onto Oxy
    success = standardize_shell(station, min_height, out_dir / station.name, cleanup=False)
    if not success:
        return False
    # find supporting columns for rough alignment purposes
    success = find_columns(station, z_slices, thickness, num_pts_min, transform=False, reload_station=False, out_dir=out_dir / station.name)
    if not success:
        return False
    return True