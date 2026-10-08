from Internal.source.internal_scans.without_floating_roof.project import STAGE, InternalNoFloatRoofProject

###############################################################################
# CYLINDER EXTRACTION
###############################################################################
def pcr_cylinder_extraction(data_dir, custom_params={'target_object': 'column', 'distance_to_wall_min': 0.2}):
    proj = InternalNoFloatRoofProject(
        data_dir,
        custom_params
    )
    # 
    proj.run(
        phase=STAGE.CYLINDER_EXTRACTION,
        mode='single'
    )

###############################################################################
# ROUGH ALIGNMENT
###############################################################################
def pcr_rough_alignment(data_dir, custom_params={'target_object': 'column', 'distance_to_wall_min': 0.2}, pre_aligned=False):
    proj = InternalNoFloatRoofProject(
        data_dir,
        custom_params
    )
    # 
    proj.run(
        phase=STAGE.ROUGH_ALIGNMENT,
        mode='single',
        pre_aligned=pre_aligned
    )


###############################################################################
# FINE ALIGNMENT
###############################################################################
def pcr_fine_alignment(data_dir, custom_params={'target_object': 'column', 'distance_to_wall_min': 0.2, 'icp': {'estimate_scanner_position': True}}):
    proj = InternalNoFloatRoofProject(
        data_dir,
        custom_params
    )
    # 
    proj.run(
        phase=STAGE.ICP,
        mode='single'
    )