import numpy as np

threshold = {
    "cell_size": 1.0,
    "min_ptc": 100,
    "min_tank_height": 5.,
    "point_gap": 0.2,
    "min_length": 5.0,
    "cylinder_estimation": {
        "voxel_size": 0.25,
        "voxel_min_ptc": 10,
        "vert_surface": np.cos(np.deg2rad(40.)),
        "cyl_norm": np.cos(np.deg2rad(7.5)),
        "ground_points": np.cos(np.deg2rad(22.5)),
        "cyl_max_dist": 0.01,
        "ransac": {
            "max_itr": 200,
            "min_ptc": 50,
            "max_norm": np.cos(np.deg2rad(10.)),
            "max_dist": 0.05
        },
        "min_num_ptc": 10000
    }
}