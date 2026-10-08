import streamlit as st
from .registration import pcr_fine_alignment
from .utils import vis_fine_alignment_output
from common.utils import Task

def module_fine_alignment(Proj):
    with st.expander('Fine alignment'):
        left, right = st.columns([1,1])
        with left:
            if st.button('Run', key='btn_run_fine_alignment'):
                Proj.task = Task.FINE_ALIGNMENT
            print_holder = st.empty()
            logging_holder = st.empty()
            # if run_rough_alignment or Proj.run_type=='Run all':
            if Proj.task == Task.FINE_ALIGNMENT:
                msg_info = []
                msg_error = []
                with st.spinner('Fine aligning pairs of scans'):
                # with st_stdout("text", print_holder, msg_info), st_stderr("info", logging_holder, msg_error):
                    pcr_fine_alignment(Proj.data_dir)
                    Proj.update_status('registered', '')
                Proj.job['fine alignment'] = True
                Proj.log['fine alignment'].update({
                    'info': msg_info,
                    'error': msg_error
                })
                if Proj.run_type == 'Run all':
                    Proj.task = Task.EVALUATION
                else:
                    Proj.task = Task.END
            elif Proj.job['fine alignment']:
                print_holder.text(''.join(Proj.log['fine alignment']['info']))
        with right:
            if Proj.data_dir:
                try:
                    fig = vis_fine_alignment_output(Proj.data_dir, Proj.station_labels)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.write('Fine alignment data is not available yet')
    

    # with st.expander('Fine alignment'):
    #     run_rough_alignment = st.button('Run', key='btn_run_fine_alignment')
    #     print_holder = st.empty()
    #     logging_holder = st.empty()
    #     if run_rough_alignment:
    #         msg_info = []
    #         msg_error = []
    #         with st_stdout("text", print_holder, msg_info), st_stderr("info", logging_holder, msg_error):
    #             pcr_fine_alignment(Proj.data_dir)
    #         Proj.job['fine alignment'] = True
    #         Proj.log['fine alignment'].update({
    #             'info': msg_info,
    #             'error': msg_error
    #         })
    #     elif Proj.job['fine alignment']:
    #         print_holder.text(''.join(Proj.log['fine alignment']['info']))
            # logging_holder.info(''.join(Proj.log['rough alignment']['error']))
        