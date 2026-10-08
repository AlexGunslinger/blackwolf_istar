import streamlit as st
from common.st_logging import st_stdout, st_stderr
from .registration import pcr_rough_alignment
from .utils import vis_rough_alignment_output
from common.utils import Task

def module_rough_alignment(Proj):
    with st.expander('Rough alignment'):
        left, right = st.columns([1,1])
        with left:
            if st.button('Run', key='btn_run_rough_alignment'):
                Proj.task = Task.ROUGH_ALIGNMENT
            print_holder = st.empty()
            logging_holder = st.empty()
            # if run_rough_alignment or Proj.run_type=='Run all':
            if Proj.task == Task.ROUGH_ALIGNMENT:
                msg_info = []
                msg_error = []
                # with st_stdout("text", print_holder, msg_info), st_stderr("info", logging_holder, msg_error):
                run_parts = []
                if Proj.job['bottom']:
                    run_parts.append('bottom')
                if Proj.job['top']:
                    run_parts.append('top')                    
                pcr_rough_alignment(
                    Proj.data_dir,
                    parts=run_parts,
                    pre_aligned=Proj.pre_aligned
                )
                
                Proj.job['rough alignment'] = True
                Proj.log['rough alignment'].update({
                    'info': msg_info,
                    'error': msg_error
                })
                if Proj.run_type == 'Run all':
                    Proj.task = Task.FINE_ALIGNMENT
                else:
                    Proj.task = Task.END
            elif Proj.job['rough alignment']:
                print_holder.text(''.join(Proj.log['rough alignment']['info']))
                # logging_holder.info(''.join(Proj.log['rough alignment']['error']))
        # with right:
        #     if Proj.data_dir:
        #         sel = st.selectbox(
        #             'Show rough alignment outputs',
        #             ['all cylinders', 'target cylinder only']
        #         )
        #         try:
        #             fig = vis_rough_alignment_output(Proj.data_dir, sel, Proj.station_labels)
        #             st.plotly_chart(fig, use_container_width=True)
        #         except Exception:
        #             st.write('Rough alignment data is not available yet')
    
