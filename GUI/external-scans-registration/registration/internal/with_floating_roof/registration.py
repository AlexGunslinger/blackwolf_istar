import sys
import traceback

import os
import time
import pickle
from glob import glob
from pathlib import Path
from copy import deepcopy

from multiprocessing import Pool

from Internal.source.internal_scans.with_floating_roof.project import STAGE, InternalFloatRoofProject

###############################################################################
# CYLINDER EXTRACTION
###############################################################################
def pcr_cylinder_extraction(data_dir, custom_params={'bottom': {'min_tank_height': 0.3, 'z_slices': [0.5,0.65,0.8]}}, parts=['bottom', 'top']):
    proj = InternalFloatRoofProject(data_dir, custom_params)
    # 
    proj.run(
        phase=STAGE.CYLINDER_EXTRACTION,
        mode='single'
    )

###############################################################################
# ROUGH ALIGNMENT
###############################################################################
def pcr_rough_alignment(data_dir, custom_params={}, parts=['bottom', 'top'], pre_aligned=False):
    proj = InternalFloatRoofProject(data_dir, custom_params)
    # 
    proj.run(
        parts,
        phase=STAGE.ROUGH_ALIGNMENT,
        mode='single',
        pre_aligned=pre_aligned
    )


###############################################################################
# FINE ALIGNMENT
###############################################################################
def pcr_fine_alignment(data_dir, custom_params={}, parts=['bottom', 'top']):    
    proj = InternalFloatRoofProject(data_dir, custom_params)
    # 
    proj.run(
        parts,
        phase=STAGE.ICP,
        mode='single'
    )