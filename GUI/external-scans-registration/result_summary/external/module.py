import streamlit as st
import streamlit as st
from glob import glob

from common.utils import METER2FOOT
from result_summary.external.module_plot_evaluation_metrics import plot_evaluation_metrics
from result_summary.external.module_run_evaluation_metrics import run_evaluation_metrics 
from result_summary.external.module_plot_horizontal_cross_sections import run_plot_horizontal_cross_section
from result_summary.external.module_plot_vertical_cross_sections import run_plot_vertical_cross_section
from common.utils import FOOT2METER, METER2FOOT
from common.utils import Task

def result_summary_external_scan(Proj,scan_type='External',data_dir=None):

    st.markdown('<font color="blue">RESULTS SUMMARY</font>', unsafe_allow_html=True)
    if data_dir is None:
        try:
            # Proj.data_dir = Path('D:\Registration_Projects\Data\AWS_S3\JB_GUI_Test')
            data_dir = str(Proj.data_dir / 'target_tank' / 'global' / 'ground' / 'icp')
        except:
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available...</p>'
            st.markdown(message, unsafe_allow_html=True)
            # st.write('Registration data is not available...')
            return
    else:
        data_dir = data_dir
    
    if Proj.task != Task.EVALUATION:
        # Don't run evaluation. This is rather a quick fix for the current code
        run_type = 'Step by step'
    else:
        run_type=Proj.run_type
    if Proj.unit == '[ft]':
        unit = 'ft'
    elif Proj.unit == '[m]':
        unit = 'm' 
    # run to extract evaluation metrics
    run_evaluation_metrics(data_dir,run_type,scan_type,unit=unit)
    # plot evaluation metrics
    plot_evaluation_metrics(data_dir,run_type,scan_type,unit)
    # plot horizontal cross-sections
    run_plot_horizontal_cross_section(data_dir,run_type,unit)
    # plot vertical cross-sections
    run_plot_vertical_cross_section(data_dir,run_type,unit)


