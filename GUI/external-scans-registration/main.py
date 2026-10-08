import streamlit as st
from project import Project
from pathlib import Path

# source code for internal and external registration
# directory structure:
#
# processing_source_code
# ---- External
# ---- Internal
from config import PROCESSING_SOURCE_CODE_DIR
import sys
sys.path.append(PROCESSING_SOURCE_CODE_DIR)

from job_overview.module import module_job_overview
# from parameter.module import module_parameter
from registration.module import module_registration
from result_summary.module import module_result_summary
from aws_s3.module import module_upload_results

if __name__=="__main__":
    if 'proj' not in st.session_state:
        st.session_state.proj = Project(
            local_storage=Path('D:\REGISTRATION DATA\Test_S3'),
            s3_dir=None,
            data_dir=None,
        )
    Proj = st.session_state.proj
    if 'submit_registration' not in st.session_state:
        st.session_state.submit_registration = False

    st.set_page_config(layout="wide")
    
    with st.sidebar:
        left, right = st.columns([1,2.5])
        with left:
            st.write('\n\n')
            from PIL import Image
            logo = Image.open('logos\BW_iSTAR_Tank_Words.jpg')
            st.image(logo, use_column_width=True)
        with right:
            st.markdown('## intelligent Storage Tank Auto Registration')
        with st.form('Type of Registration'): #,expanded=True):
            # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            Proj.registration_type = st.radio('Type of Registration', ('External scans','Internal scans'))

            if st.form_submit_button('Submit'):
                st.session_state.submit_registration = True
        
        with st.container():
            st.subheader('General settings')
            Proj.run_type = st.radio('Run type', ('Run all','Step by step'),index=1)
                # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            Proj.unit     = st.radio('Unit for evaluation metrics', ('[ft]','[m]'),index=1) 

            if Proj.registration_type == 'Internal scans':
                # floating_roof = st.selectbox(
                #     'Floating roof:',
                #     ('Tank has a floating roof', 'Tank does not have a floating roof')
                # )
                # Proj.floating_roof = floating_roof == 'Tank has a floating roof'

                floating_roof = st.radio('Floating roof', ('Yes','No'),index=0) 
                Proj.floating_roof = floating_roof == 'Yes'
                
                if Proj.floating_roof:
                    st.write('Tank parts')
                    Proj.tank_parts = []
                    Proj.job['top'] = st.checkbox('ABOVE the floating roof', value=True)
                    if Proj.job['top']:
                        Proj.tank_parts.append('top')
                    Proj.job['bottom'] = st.checkbox('BELOW the floating roof', value=True)
                    if Proj.job['bottom']:
                        Proj.tank_parts.append('bottom')
                left,right = st.columns([1.5,1])
                with left:
                    Proj.ref_scan = int(st.text_input(f'Reference scan for levelling',Proj.ref_scan))                        
            Proj.pre_aligned = st.radio('Are the scans roughly aligned?', ('Yes','No'),index=1) == 'Yes'

    if st.session_state.submit_registration:
        if Proj.registration_type=='External scans':
            st.header('External Scans Registration')
            scan_type = 'External'
        elif Proj.registration_type=='Internal scans':
            st.header('Internal Scans Registration')
            scan_type = 'Internal'
            
        # =========================================================================
        # JOB OVERVIEW
        # =========================================================================
        module_job_overview(Proj)
        # =========================================================================
        # PARAMETERS
        # =========================================================================
        # module_parameter(Proj)
        
        # if Proj.run_registration:
        # =========================================================================
        # MAIN PROCESSES
        # =========================================================================
        module_registration(Proj)
        
        # =========================================================================
        # RESULT SUMMARY
        # =========================================================================
        module_result_summary(Proj,scan_type=scan_type)
        
        # =========================================================================
        # Upload result to S3
        # =========================================================================
        module_upload_results(Proj)

        #     # set run_type back to manual to avoid rerun all steps
        #     # if Proj.run_type == 'Run all':
        #     #     Proj.run_type = 'Step by step'
        #     #     Proj.job['summarise'] = False 
        #     #     Proj.run_registration = False

        # elif Proj.registration_type=='Internal scans':
        #     st.header('Internal Scans Registration')
        #     scan_type = 'Internal'

        #     # =========================================================================
        #     # JOB OVERVIEW
        #     # =========================================================================
        #     module_job_overview(Proj)
        #     # =========================================================================
        #     # PARAMETERS
        #     # =========================================================================
        #     # module_parameter(Proj)

        #     if Proj.run_registration:
        #         # =========================================================================
        #         # MAIN PROCESSES
        #         # =========================================================================
        #         module_registration(Proj)        

        #         # =========================================================================
        #         # RESULT SUMMARY
        #         # =========================================================================
        #         # data_dir = 'D:\REGISTRATION DATA\JB-641 21ft Cylinder Column\\bottom\\fine_alignment'
        #         module_result_summary(Proj,scan_type=scan_type)

        #         # =========================================================================
        #         # Upload result to S3
        #         # =========================================================================
        #         module_upload_results(Proj)
            
        #     # set run_type back to manual to avoid rerun all steps
        #     if Proj.run_type == 'Run all':
        #         Proj.run_type = 'Step by step'
        #         Proj.job['summarise'] = False 
        #         Proj.run_registration = False