default_parameters = {
    'num_process': 6,
    'min_tank_height': 1.5,
    'z_slices': [1, 2, 3],
    'distance_to_wall_min': 0.2,
    'target_object': 'column',
    'icp': {
        'auto_pair': True,
        'estimate_scanner_position': False,
        'voxel_size': 0.003,
        'max_correspondence_distance_coarse': 0.2,
        'max_correspondence_distance_fine': 0.01
    },
    'ref_scan': 0
}
parameters = {
    'JB-569 120ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-569 120ft Cylinder Column',
        'num_process': 7,
    },
    'JB-582 90ft API Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-582 90ft API Column\E57 Exports',
        'num_process': 2,
    },
    'JB-608 60ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-608 60ft Cylinder Column\E57 Exports',
        'num_process': 6,
        'z_slices': [0.3, 0.5, 0.8],
    },
    'JB-513 43ft No Columns': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-513 43ft No Columns\E57 Exports',
        'num_process': 6,
        'distance_to_wall_min': 0.5,
        'z_slices': [0.3, 0.5, 0.8],
        'target_object': 'manway',
    },
    'JB-503 52ft No Columns': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-503 52ft No Columns',
        'num_process': 6,
        'distance_to_wall_min': 0.5,
        'target_object': 'manway',
    },
    'JB-642 93ft Cylinder Column': {
        'data_dir': 'D:\REGISTRATION DATA\Internal - No FR\JB-642 93ft Cylinder Column',
        'num_process': 10,
        'distance_to_wall_min': 0.5,
        'z_slices': [1, 2, 3],
        'target_object': 'manway',
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