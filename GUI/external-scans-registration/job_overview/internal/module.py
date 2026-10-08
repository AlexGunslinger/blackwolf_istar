import traceback
import streamlit as st
from pathlib import Path
from aws_s3.s3_bucket import s3_download_directory
from .utils import load_data, visualise_thin_slice, write_summarised_data, load_summarised_data
import pandas as pd
import os
from common.utils import Task
                  
def module_job_overview(Proj):
    st.markdown('<font color="blue">JOB OVERVIEW</font>', unsafe_allow_html=True)
    if Proj.floating_roof:
        titles = {
            'bottom': 'Tank part **BELOW** the floating roof',
            'top': 'Tank part **ABOVE** the floating roof'
        }
    with st.expander('Job overview',expanded=True):
        left, right, = st.columns([1,1])
        with left:
            with st.form('job overview'):
                # st.write('##')
                # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                data_location = st.radio('Specify data storage location', ('Local', 'AWS S3'))
                # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                # Proj.run_type = st.radio('Select run type', ('Run all','Step by step'),index=1)
                # # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                # Proj.unit     = st.radio('Select unit type for metrics evaluation', ('[ft]','[m]'),index=1) 
                                
                path = st.text_input(
                    label='Data directory',
                    placeholder='Path to the directory with all scan data points')

                job_id = st.text_input(
                    label='Job ID',
                    placeholder='Job ID number')
                try:
                    Proj.job_id = int(job_id)
                except:
                    Proj.job_id = None

                if Proj.run_type == 'Run all':
                    btn_message = 'Set and Run all'
                else:
                    btn_message = 'Set'
                btn_summarise_press = st.form_submit_button(btn_message)
                if btn_summarise_press or Proj.job['summarise']:
                    if btn_summarise_press:
                        # button pressed
                        Proj.job['summarise'] = True
                        Proj.data_dir = Path(path)
                        Proj.run_registration = True 
                        if data_location == 'AWS S3':
                            Proj.s3_dir = path
                            # Downloading data
                            dst = Proj.local_storage / Proj.data_dir.stem
                            st.write('Downloading data from AWS S3.')
                            st.write(f'Saving to {str(dst)}')
                            Proj.update_status('retrieving_files', 'Started')
                            if Proj.floating_roof:
                                for tank_part in Proj.tank_parts:
                                    s3_download_directory(os.path.join(Proj.s3_dir, tank_part), dst / tank_part)
                            else:
                                s3_download_directory(Proj.s3_dir, dst)
                            Proj.update_status('retrieving_files', 'Finished')
                            Proj.data_dir = dst
                        
                    if Proj.data_dir:
                        # Proj.data_dir = Path(path)
                        # Only when the 'Summarise' button is pressed, check if this has already been done
                        # if btn_summarise_press:
                        #     data_summary_file = Proj.data_dir / 'job_overview' / 'data_summary.pkl'
                        #     if os.path.exists(data_summary_file):
                        #         with st.spinner('Loading summary data'):
                        #             try:
                        #                 Proj.data_summary = load_summarised_data(data_summary_file)
                        #                 Proj.station_labels = dict(zip(
                        #                     Proj.data_summary['file_names'], Proj.data_summary['station_labels']
                        #                 ))
                        #             except:
                        #                 Proj.data_summary = {}
                        #                 info_holder = st.empty()
                        #                 info_holder.write('Failed reading summary data. Going to read scan data ...')
                        if Proj.floating_roof:
                            try:
                                if Proj.data_summary == {}:
                                    Proj.data_summary = {
                                        'bottom': {},
                                        'top': {}
                                    }
                                if Proj.station_labels == {}:
                                    Proj.station_labels = {
                                        'bottom': {},
                                        'top': {}
                                    }
                                for tank_part in Proj.tank_parts:
                                    st.markdown(titles[tank_part])
                                    # Only when the 'Summarise' button is pressed, check if this has already been done
                                    if btn_summarise_press:
                                        data_summary_file = Proj.data_dir / tank_part / 'job_overview' / 'data_summary.pkl'
                                        if os.path.exists(data_summary_file):
                                            with st.spinner('Loading summary data'):
                                                try:
                                                    Proj.data_summary[tank_part] = load_summarised_data(data_summary_file)
                                                    Proj.station_labels[tank_part] = dict(zip(
                                                        Proj.data_summary[tank_part]['file_names'], Proj.data_summary[tank_part]['station_labels']
                                                    ))
                                                except:
                                                    Proj.data_summary[tank_part] = {}
                                                    info_holder = st.empty()
                                                    info_holder.write('Failed reading summary data. Going to read scan data ...')
                                                    
                                    if (Proj.data_summary[tank_part] is None) or (Proj.data_summary[tank_part] == {}):
                                        file_names, labels, ids, file_sizes, npoints, points = load_data(Proj.data_dir / tank_part)
                                        # save
                                        Proj.data_summary[tank_part] = {
                                            'file_names': file_names,
                                            'station_labels': labels,
                                            'station_ids': ids,
                                            'file_sizes': file_sizes,
                                            'number of points': npoints,
                                            'slices': points
                                        }
                                        Proj.station_labels[tank_part] = dict(zip(
                                            Proj.data_summary[tank_part]['file_names'], Proj.data_summary[tank_part]['station_labels']
                                        ))
                                        # store to file for future use
                                        data_summary_file = Proj.data_dir / tank_part / 'job_overview' / 'data_summary.pkl'
                                        write_summarised_data(Proj.data_summary[tank_part], data_summary_file)

                                    st.markdown('<font color="blue">Scan information</font>', unsafe_allow_html=True)

                                    data_summary = pd.DataFrame({
                                        'Scan label': Proj.data_summary[tank_part]['station_labels'],
                                        'File size': Proj.data_summary[tank_part]['file_sizes'],
                                        'Number of points': Proj.data_summary[tank_part]['number of points']
                                    })
                                    st.write(data_summary)
                            except:
                                message = f'<p style="font-family:sans-serif; color:Red; font-size: 14px;">Something went wrong. Check data path...</p>'
                                st.markdown(message, unsafe_allow_html=True)
                        else:
                            try:
                                # Only when the 'Summarise' button is pressed, check if this has already been done
                                if btn_summarise_press:
                                    data_summary_file = Proj.data_dir / 'job_overview' / 'data_summary.pkl'
                                    if os.path.exists(data_summary_file):
                                        with st.spinner('Loading summary data'):
                                            try:
                                                Proj.data_summary = load_summarised_data(data_summary_file)
                                                Proj.station_labels = dict(zip(
                                                    Proj.data_summary['file_names'], Proj.data_summary['station_labels']
                                                ))
                                            except:
                                                Proj.data_summary = {}
                                                info_holder = st.empty()
                                                info_holder.write('Failed reading summary data. Going to read scan data ...')
                                if (Proj.data_summary is None) or (Proj.data_summary == {}):
                                    file_names, labels, ids, file_sizes, npoints, points = load_data(Proj.data_dir)
                                    # save
                                    Proj.data_summary = {
                                        'file_names': file_names,
                                        'station_labels': labels,
                                        'station_ids': ids,
                                        'file_sizes': file_sizes,
                                        'number of points': npoints,
                                        'slices': points
                                    }
                                    Proj.station_labels = dict(zip(
                                        Proj.data_summary['file_names'], Proj.data_summary['station_labels']
                                    ))
                                    # store to file for future use
                                    write_summarised_data(Proj.data_summary, data_summary_file)

                                st.markdown('<font color="blue">Scan information</font>', unsafe_allow_html=True)

                                data_summary = pd.DataFrame({
                                    'Scan label': Proj.data_summary['station_labels'],
                                    'File size': Proj.data_summary['file_sizes'],
                                    'Number of points': Proj.data_summary['number of points']
                                })
                                st.write(data_summary)
                            except Exception as exc:
                                message = f'<p style="font-family:sans-serif; color:Red; font-size: 14px;">Something went wrong. Check data path...</p>'
                                st.markdown(message, unsafe_allow_html=True)
                                traceback.print_exc(exc)
                    
                    if btn_summarise_press and (Proj.run_type == 'Run all'):
                        Proj.task = Task.CYLINDER_EXTRACTION
            # Plot crossection
        with right:
            if Proj.floating_roof:
                for tank_part in Proj.tank_parts:
                    st.markdown(titles[tank_part])
                    if Proj.data_dir:
                        file_names = Proj.data_summary[tank_part].get('file_names', [])
                    else:
                        file_names = []
                        # with st.form('Plot'):
                    sel_station = st.selectbox(
                        'Select a station to view a horizontal slice',
                        file_names,
                        key=f'thin_slice_{tank_part}'
                    )   
                    # if st.form_submit_button('Apply'):
                    if sel_station:
                        fig = visualise_thin_slice(
                            sel_station,
                            Proj.data_summary[tank_part]['slices']
                        )
                        st.plotly_chart(fig, use_container_width=True)     
            else:
                if Proj.data_dir:
                    file_names = Proj.data_summary.get('file_names', [])
                else:
                    file_names = []
                    # with st.form('Plot'):
                sel_station = st.selectbox(
                    'Select a station to view a horizontal slice',
                    file_names
                )   
                # if st.form_submit_button('Apply'):
                if sel_station:
                    fig = visualise_thin_slice(
                        sel_station,
                        Proj.data_summary['slices']
                    )
                    st.plotly_chart(fig, use_container_width=True)                  