import requests
from common.utils import Task

DEFAULT_PARAMETERS_INFO={
    'cell_size': 'The size to downsample point cloud. Specifically, this cell size is used for creating a voxel grid', 
    'min_ptc': 'To filter voxels. Voxels with less than this "min_ptc" points is considered empty. This parameter acts similarly to a noise remover.', 
    'min_tank_height': 'This number is used to identify voxels that potentially belong to a tank. The target tank should be taller than this number. ',
    'ransac_max_itr': 'The maximum number of iterations for the RANSAC algorithm that is used to fit a set of points to a cylinder model.',
    'min_num_ptc': 'The minimum number of points that a cylinder should have. Clusters with less number of points are ignored.',
    'max_correspondence_distance_coarse': 'During the fine alignment process, points from two stations are matched which is only possible when their distances are not less then the predefined distance. This parameter is for the first fine alignment - so-called coarse alignment.', 
    'max_correspondence_distance_fine': 'During the fine alignment process, points from two stations are matched which is only possible when their distances are not less then the predefined distance. This parameter is for the second fine alignment - so-called fine alignment.'
}
DEFAULT_PARAMETERS={
    'cylinder_extraction': {
        'cell_size': 1.0,
        'min_ptc': 100,
        'min_tank_height': 5.0,
        'ransac_max_itr': 200,
        'min_num_ptc': 10000
    },
    'fine_alignment': {
        'max_correspondence_distance_coarse': 0.2,
        'max_correspondence_distance_fine': 0.01
    }
}

JOB_API_URI = 'https://ec2-3-141-161-185.us-east-2.compute.amazonaws.com/public_api.php?'


class Project():
    def __init__(self, local_storage=None, s3_dir=None, data_dir=None, parameters=DEFAULT_PARAMETERS, parameter_info=DEFAULT_PARAMETERS_INFO):
        self.data_location = 'Local'
        self.local_storage = local_storage
        self.s3_dir = s3_dir
        self.data_dir = data_dir
        # external or internal registration
        self.registration_type = 'External scans'
        # has a floating roof
        self.floating_roof = False
        # for internal with a floating roof
        self.tank_parts = ['bottom', 'top']
        self.pre_aligned = False
        self.data_summary = {}
        self.station_labels = {}
        self.parameters = parameters
        self.parameter_info = parameter_info
        self.job = {
            'summarise': False,
            'cylinder extraction': False,
            'rough alignment': False,
            'fine alignment': False,
            # for internal with floating roof
            'bottom': True,
            'top': True,
            'evaluation': False,
        }
        self.task = Task.INIT
        self.log = {
            'cylinder extraction': {
                'info': None,
                'error': None
            },
            'rough alignment': {
                'info': None,
                'error': None
            },
            'fine alignment': {
                'info': None,
                'error': None
            }
        }
        self.run_type = 'Step by step'
        self.run_registration = False
        self.ref_scan = 0

    def update_status(self, status, message=''):
        if (not hasattr(self, 'job_id')) or (self.job_id is None):
            return 'Job id is not specified'
        params = dict(
            action='update_status',
            id=self.job_id,
            status=status,
            message=message
        )
        res = requests.get(JOB_API_URI, params, verify=False)
        return res.text