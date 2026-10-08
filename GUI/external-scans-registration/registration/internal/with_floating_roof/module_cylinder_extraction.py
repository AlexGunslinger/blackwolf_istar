import streamlit as st
from common.st_logging import st_stdout, st_stderr
from .registration import pcr_cylinder_extraction
from .utils import vis_extracted_cylinders
from common.utils import Task

# Cylinder extraction
def module_cylinder_extraction(Proj):
    with st.expander('Cylinder extraction'):
        left, right = st.columns([1,1])
        run_cylinder_extraction = left.button('Run')
        if run_cylinder_extraction:
            Proj.task = Task.CYLINDER_EXTRACTION
        right_panel = right.empty()
        right_container = right_panel.container()
        # right_container = right.empty()
        print_holder = left.empty()
        logging_holder = left.empty()
        # if run_cylinder_extraction or Proj.run_type=='Run all':
        if Proj.task == Task.CYLINDER_EXTRACTION:
            Proj.update_status('registering', '')
            with st.spinner('Extracting tanks from all stations'):
                msg_info = []
                msg_error = []
                with st_stdout("text", print_holder, msg_info), st_stderr("info", logging_holder, msg_error):
                    # run_parts = []
                    # if Proj.job['bottom']:
                    #     run_parts.append('bottom')
                    # if Proj.job['top']:
                    #     run_parts.append('top')  
                    pcr_cylinder_extraction(
                        Proj.data_dir,
                        parts=Proj.tank_parts
                    )
                Proj.job['cylinder extraction'] = True
                Proj.log['cylinder extraction'].update({
                    'info': msg_info,
                    'error': msg_error
                })
                if Proj.run_type == 'Run all':
                    Proj.task = Task.ROUGH_ALIGNMENT
                else:
                    Proj.task = Task.END
        elif Proj.job['cylinder extraction']:
            print_holder.text(''.join(Proj.log['cylinder extraction']['info']))
            # logging_holder.info(''.join(Proj.log['cylinder extraction']['error']))

        # with right_panel.container():
        #     if Proj.data_dir:
        #         # scan for pickle files
        #         # processed_stations = []
        #         # for p in Proj.data_dir.iterdir():
        #         #     if p.is_dir():
        #         #         pkl_files = glob(os.path.join(p, '*.pkl'))
        #         #         if len(pkl_files) > 0:
        #         #             processed_stations.append(p.name)
        #         processed_stations = list(Proj.station_labels.keys())
        #         # processed_stations = [p.name for p in Proj.data_dir.iterdir() if p.is_dir()]
        #     else:
        #         processed_stations = []
        #     # with st.form('Visualise identified tanks'):
        #     sel_station = st.selectbox(
        #         'Select a scan to show identified tanks',
        #         processed_stations
        #     )
        #     # if st.form_submit_button('Plot'):
        #     if sel_station:
        #         try:
        #             fig = vis_extracted_cylinders(Proj.data_dir, sel_station)
        #             st.plotly_chart(fig, use_container_width=True)
        #         except:
        #             st.write('Failed loading extracted cylinders data for this station')

        #         # for i in foo([1,2,3]):
        #         #     st.write('main count {i}\n')
        #             # arr = np.random.normal(1, 1, size=i*10)
        #             # fig, ax = plt.subplots()
        #             # ax.hist(arr, bins=20)
        #             # st.pyplot(fig)
    