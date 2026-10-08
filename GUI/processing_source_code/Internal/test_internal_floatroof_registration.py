import sys
sys.path.append('.source')

from source.internal_scans.with_floating_roof.project import STAGE, InternalFloatRoofProject

if __name__=="__main__":
    proj = InternalFloatRoofProject(
        data_dir='D:\REGISTRATION DATA\Internal - with FR\JB-641 21ft Cylinder Column',
        custom_params={
            'bottom': {
                'min_tank_height': 1.0,
                'z_slices': [0.5,0.65,0.8],
            }
        }
    )
    # 
    proj.run(
        parts=['bottom'],
        phase=STAGE.ICP,
        mode='all'
    )