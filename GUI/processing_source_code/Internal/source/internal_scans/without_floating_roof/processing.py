import gc
import os
import logging
from pathlib import Path

from ...utils.write_output import write_array2txt

def detect_objects(
    station, min_height, cell_size=0.5, 
    z_slices=[1,2,3], num_pts_min=30, distance_to_wall_min=0.2, out_dir=None):
    """
    Designed as a unit function for running parallel. This functions extract interested objects from each station scan
    """
    # write station metadata
    if out_dir is None:
        out_dir = station.file_path.parent / station.name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    log_file = out_dir / 'log.txt'
    import logging
    logging.basicConfig(
        filename=str(out_dir / log_file),
        filemode='w',
        level=logging.DEBUG
    )

    logging.info(f'parallel processing {station.name}')
    # load station points if not yet done
    logging.info('read points')
    if station.ptc is None:
        station.read_points()
    # main processing part
    try:
        # clustering station point cloud
        logging.info('clustering')
        station.clustering(
            output_dir=out_dir,
            cluster_params={
                "min_tank_height": min_height,
                'cell_size': cell_size,
                'min_length': cell_size # the same as cell_size
            }
        )
        # identify the shell wall
        logging.info('identify the wall')
        station.identify_shell_cluster()
        # estimate the ground z
        logging.info('estimate ground z')
        station.estimate_ground_z()
        # straight out the cylinder
        logging.info('make vertical')
        station.make_vertical()
        # move the center axis to coincide with Oz, also make the bottom center the same as the origin of the coordinate system
        logging.info('centralise')
        station.estimate_ground_z()
        station.translate_origin()
        # recalculate z_ground
        logging.info('recalculate top, bottom and height of the tank')
        station.z_ground = station.estimate_ground_z()
        # calculate height
        station.estimate_top_z()
        station.estimate_height()
        # save information about this part
        station.standardize = {
            'align': station.align
        }
        # check if station data is valid
        logging.info('validate station data')
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
            logging.info('station data seems fine, no error was detected')
        # detect columns centers# detect columns centers
        logging.info('find supporting columns')
        station.find_columns(z_cuts=z_slices, num_pts_min=num_pts_min, distance_to_wall_min=distance_to_wall_min, write_dir=out_dir / 'slice')

        # extract manway
        logging.info('extract_manway')
        station.extract_manway()
        
        # save station without points
        logging.info('save object')
        station.save_object(out_dir)

        # del station.ptc
        logging.info('clean up')
        station.ptc = None

        gc.collect()

    except Exception as e:
        logging.info(f'error in {station.name}')
        logging.info(e)
    
    logging.info('returning')
    return True