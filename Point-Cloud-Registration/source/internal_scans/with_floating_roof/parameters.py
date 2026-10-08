default_parameters = {
    'top': {
        'num_process': 6,
        'min_tank_height': 5,
        'z_slices': [2, 3, 4],
        'icp': {
            'auto_pair': True,
            'estimate_scanner_position': False,
            'mode': 'sequence',
            'voxel_size': 0.01,
            'max_correspondence_distance_coarse': 0.2,
            'max_correspondence_distance_fine': 0.01
        }
    },
    'bottom': {
        'num_process': 6,
        'min_tank_height': 1.5,
        'z_slices': [0.5,1,1.5],
        'icp': {
            'auto_pair': True,
            'estimate_scanner_position': False,
            'mode': 'sequence',
            'voxel_size': 0.01,
            'max_correspondence_distance_coarse': 0.2,
            'max_correspondence_distance_fine': 0.01
        }
    }
}
parameters = {
    'JB-452 60ft API Column': {
        'data_dir': 'D:\REGISTRATION DATA/Internal - with FR\JB-452 60ft API Column',
    },
    'IFR With Spheres in Hatch': {
        'data_dir': 'D:\REGISTRATION DATA\IFR With Spheres in Hatch',
    },
    'JB-447 45ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-447 45ft Cylinder Column\E57 Exports',
    },
    'JB-510 45ft API Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-510 45ft API Column\E57 Exports',
        'top': {
            'z_slices': [0.15, 3, 4]
        }
    },
    'JB-583 200ft ERF No Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-583 200ft ERF No Column',
        'top': {
            'icp': {
                'mode': 'sequence',
                'voxel_size': 0.01,
                'max_correspondence_distance_coarse': 0.5,
                'max_correspondence_distance_fine': 0.01
            }
        },
    }, # fail top
    'JB-603 61ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-603 61ft Cylinder Column',
    },
    'JB-617 60ft API Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-617 60ft API Column\E57 Exports',
    }, # fail top due to bad data
    'JB-621 95ft EFR No Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-621 95ft EFR No Column',
    },
    'JB-640 67ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-640 67ft Cylinder Column',
    },
    'JB-641 21ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-641 21ft Cylinder Column',
        'bottom': {
            'min_tank_height': 1.0,
            'z_slices': [0.5,0.65,0.8],
        }
    },
    'JB-652 150ft Multi-API Columns': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-652 150ft Multi-API Columns',
    },
    'JB-616 134ft Multi-Cylinder': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-616 134ft Multi-Cylinder',
    }, # fail due to bad data

    'JB-598 100ft EFR Spheres in Hatch': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-598 100ft EFR Spheres in Hatch',
    },
    'Internal- Tank0001 Top Scans': {
        'data_dir': 'D:\REGISTRATION DATA\Internal- Tank0001 Top Scans',
    },
    'JB-570 Internal Scans': {
        'data_dir': 'G:\JB-570 Internal Scans',
        'bottom': {
            'num_process': 6,
            'icp': {
                'mode': 'sequence',
                'voxel_size': 0.01,
                'max_correspondence_distance_coarse': 0.2,
                'max_correspondence_distance_fine': 0.01
            }
        }
    },
    'JB-417 IFR w EXT - 265ft - downsample': {
        'data_dir': 'D:\REGISTRATION DATA\JB-417 IFR w EXT - 265ft',
        'top': {
            'num_process': 12,
            'icp': {
                'max_correspondence_distance_fine': 0.005
            }
        },
        'bottom': {
            'num_process': 12,
            'icp': {
                'max_correspondence_distance_fine': 0.005
            }
        }
    },
    'JB-572 80ft API Column': {
        'data_dir': 'D:\REGISTRATION DATA/Internal - with FR\JB-572 80ft API Column/E57 Exports',
    },
    'JB-417 IFR w EXT - 265ft': {
        'data_dir': 'D:\REGISTRATION DATA\JB-417 IFR w EXT - 265ft\\NonSubsampled Scans',
        'top': {
            'num_process': 6,
            'icp': {
                'max_correspondence_distance_fine': 0.005
            }
        },
        'bottom': {
            'num_process': 6,
            'icp': {
                'max_correspondence_distance_fine': 0.005
            }
        }
    },
    'JB-635 110ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-635 110ft Cylinder Column',
    },
    'JB-548 90ft EFR No Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-548 90ft EFR No Column\E57 Exports',
    },
    'JB-570 IFR - 210ft': {
        'data_dir': 'D:\REGISTRATION DATA\JB-570 IFR - 210ft',
    },
    'Tank 10-14': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\Tank 10-14',
    },
    'iSTAR Files': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\iSTAR Files',
    },
    'JB-626': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - with FR\JB-626',
        'bottom': {
            'z_slices': [0.6,0.8,1],
            'num_process': 12,
            'icp': {
                'auto_pair': True,
                'estimate_scanner_position': True,
                'mode': 'sequence',
                'voxel_size': 0.01,
                'max_correspondence_distance_coarse': 0.2,
                'max_correspondence_distance_fine': 0.01
            }
        }
    },
}

def update_params(params, custom_params):
    for param, value in custom_params.items():
        if param in custom_params:
            if isinstance(value, dict):
                update_params(params[param], value)
            else:
                params[param] = value
        else:
            params[param] = value
    return params

