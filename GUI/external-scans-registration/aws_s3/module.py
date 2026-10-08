import streamlit as st
import os
from .s3_bucket import s3_upload_directory
from pathlib import Path
from common.utils import Task

def module_upload_results(Proj):
    st.markdown('<font color="blue">UPLOAD RESULTS</font>', unsafe_allow_html=True)
    if st.button('Upload results'):
        Proj.task = Task.UPLOAD_RESULTS
    if Proj.task == Task.UPLOAD_RESULTS:
        Proj.update_status('storing_files', '')
        if Proj.s3_dir is None:
            Proj.s3_dir = os.path.join(
                # 'Auto Registration Project/JB-GUI-Test',
                'iSTAR/',
                Path(Proj.data_dir).stem
            )
        if Proj.registration_type == 'External scans':
            # registered scans
            s3_upload_directory(
                src=os.path.join(
                    Proj.data_dir, 'target_tank', 'global', 'ground', 'icp', 'e57'
                ),
                dst=os.path.join(
                    Proj.s3_dir, 'registration', 'result'
                )
            )
            # evaluation
            s3_upload_directory(
                src=os.path.join(
                    Proj.data_dir, 'target_tank', 'global', 'ground', 'icp', 'evaluation'
                ),
                dst=os.path.join(
                    Proj.s3_dir, 'registration', 'evaluation'
                )
            )
        else:
            if Proj.floating_roof:
                for tank_part in Proj.tank_parts:
                    # registered scans
                    s3_upload_directory(
                        src=os.path.join(
                            Proj.data_dir, tank_part, 'fine_alignment', 'e57'
                        ),
                        dst=os.path.join(
                            Proj.s3_dir, tank_part, 'registration', 'result'
                        )
                    )
                    # evaluation
                    s3_upload_directory(
                        src=os.path.join(
                            Proj.data_dir, tank_part, 'fine_alignment', 'evaluation'
                        ),
                        dst=os.path.join(
                            Proj.s3_dir, tank_part, 'registration', 'evaluation'
                        )
                    )
            else:
                # registered scans
                s3_upload_directory(
                    src=os.path.join(
                        Proj.data_dir, 'fine_alignment', 'e57'
                    ),
                    dst=os.path.join(
                        Proj.s3_dir, 'registration', 'result'
                    )
                )
                # evaluation
                s3_upload_directory(
                    src=os.path.join(
                        Proj.data_dir, 'fine_alignment', 'evaluation'
                    ),
                    dst=os.path.join(
                        Proj.s3_dir, 'registration', 'evaluation'
                    )
                )
        Proj.update_status('ready_for_download', '')
        Proj.task = Task.END