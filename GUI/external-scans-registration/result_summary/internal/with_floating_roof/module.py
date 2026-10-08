import streamlit as st
from common.utils import METER2FOOT
from result_summary.internal.with_floating_roof.module_plot_evaluation_metrics import plot_evaluation_metrics_top, plot_evaluation_metrics_bottom
from result_summary.internal.with_floating_roof.module_run_evaluation_metrics import run_evaluation_metrics_top, run_evaluation_metrics_bottom
from result_summary.internal.with_floating_roof.module_plot_horizontal_cross_sections import run_plot_horizontal_cross_section_top, run_plot_horizontal_cross_section_bottom
from result_summary.internal.with_floating_roof.module_plot_vertical_cross_sections import run_plot_vertical_cross_section_top, run_plot_vertical_cross_section_bottom
from common.utils import FOOT2METER, METER2FOOT
from common.utils import Task

def result_summary_internal_scan_with_floating_roof(Proj,scan_type='External',data_dir=None):

    st.markdown('<font color="blue">RESULTS SUMMARY</font>', unsafe_allow_html=True)
    if data_dir is None:
        try:
            # Proj.data_dir = Path('D:\Registration_Projects\Data\AWS_S3\JB_GUI_Test')
            data_dir_top    = str(Proj.data_dir / 'top' / 'fine_alignment')
            data_dir_bottom = str(Proj.data_dir / 'bottom' / 'fine_alignment')
        except:
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available...</p>'
            st.markdown(message, unsafe_allow_html=True)
            # st.write('Registration data is not available...')
            return
    else:
        data_dir_top    = data_dir
        data_dir_bottom = data_dir
    
    if Proj.task != Task.EVALUATION:
        # Don't run evaluation. This is rather a quick fix for the current code
        run_type = 'Step by step'
    else:
        run_type=Proj.run_type
    if Proj.unit == '[ft]':
        unit = 'ft'
    elif Proj.unit == '[m]':
        unit = 'm' 
    
    if Proj.job['top'] == True:
        # =========================================================================
        # RESULTS SUMMARY with evaluation metrics for TOP part
        # =========================================================================    
        st.markdown('<font color="blue">Evaluation for the top part</font>', unsafe_allow_html=True)       
        # run to extract evaluation metrics
        run_evaluation_metrics_top(data_dir_top,run_type,scan_type,unit=unit)
        # plot evaluation metrics
        plot_evaluation_metrics_top(data_dir_top,run_type,scan_type,unit)
        # plot horizontal cross-sections
        run_plot_horizontal_cross_section_top(data_dir_top,run_type,unit)
        # plot vertical cross-sections
        run_plot_vertical_cross_section_top(data_dir_top,run_type,unit)

    if Proj.job['bottom'] == True:
        # =========================================================================
        # RESULTS SUMMARY with evaluation metrics for BOTTOM part
        # =========================================================================
        st.markdown('<font color="blue">Evaluation for the bottom part</font>', unsafe_allow_html=True)
        # run to extract evaluation metrics
        run_evaluation_metrics_bottom(data_dir_bottom,run_type,scan_type,unit=unit)
        # plot evaluation metrics
        plot_evaluation_metrics_bottom(data_dir_bottom,run_type,scan_type,unit)
        # plot horizontal cross-sections
        run_plot_horizontal_cross_section_bottom(data_dir_bottom,run_type,unit)
        # plot vertical cross-sections
        run_plot_vertical_cross_section_bottom(data_dir_bottom,run_type,unit)
