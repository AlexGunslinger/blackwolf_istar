import sys
sys.path.append('.source')

from source.internal_scans.without_floating_roof.project import STAGE, InternalNoFloatRoofProject

if __name__=="__main__":
    proj = InternalNoFloatRoofProject(
        data_dir='D:\REGISTRATION DATA\Test-GUI\JB-503-NoFR',
        custom_params={
            'target_object': 'manway',
            'distance_to_wall_min': 0.2,
        }
    )
    # 
    proj.run(
        phase=STAGE.CYLINDER_EXTRACTION,
        mode='all'
    )