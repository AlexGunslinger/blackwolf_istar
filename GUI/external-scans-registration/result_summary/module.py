from result_summary.external.module import result_summary_external_scan
from result_summary.internal.with_floating_roof.module import result_summary_internal_scan_with_floating_roof 
from result_summary.internal.without_floating_roof.module import result_summary_internal_scan_without_floating_roof 
import streamlit as st
from result_summary.define_session_state import define_session_state
from common.utils import Task

def module_result_summary(Proj,scan_type='External',data_dir=None):
    if Proj.registration_type == 'Internal scans':
        if Proj.floating_roof:
            define_session_state(Proj) # initialize session state
            define_session_state(Proj,'with floating roof') # restart for others
            result_summary_internal_scan_with_floating_roof(Proj,scan_type=scan_type)
        else:
            define_session_state(Proj)
            define_session_state(Proj,'without floating roof')
            result_summary_internal_scan_without_floating_roof(Proj,scan_type=scan_type)
    else:
        define_session_state(Proj)
        define_session_state(Proj,'external')       
        result_summary_external_scan(Proj,scan_type=scan_type)
    # Proj.job['evaluation'] = True
    if (Proj.task == Task.EVALUATION) and (Proj.run_type == 'Run all'):
        Proj.task = Task.UPLOAD_RESULTS