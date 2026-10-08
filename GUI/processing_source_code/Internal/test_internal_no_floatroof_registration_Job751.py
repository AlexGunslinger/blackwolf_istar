import sys
sys.path.append('.source')

from source.internal_scans.without_floating_roof.project import STAGE, InternalNoFloatRoofProject

if __name__=="__main__":
    proj = InternalNoFloatRoofProject(
        data_dir='D:\REGISTRATION DATA\Test_S3\JB-751_TK_6115\\test_fine_alignment',
        custom_params={
            'target_object': 'manway',
            'distance_to_wall_min': 0.2,
            'icp': {'auto_pair': False}
        }
    )
    # 
    proj.run(
        phase=STAGE.ICP,
        mode='all'
    )