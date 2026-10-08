

import streamlit as st
import streamlit as st
from glob import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt
from PIL import Image
import math
from result_summary.utils import evaluation_metrics, load_and_combine_registation_data, plot_cross_section
from common.utils import METER2FOOT

#####################
#       TOP
#####################
def run_evaluation_metrics_top(data_dir,run_type='Manual',scan_type='External',unit='m'):
        
    with st.expander('Evaluation metric analyses'):
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        # unit = st.radio('Select an unit type for visualization', ['[m]','[ft]'])    
        with st.form('Evaluation metric'): 

            if unit == 'ft': #unit == '[ft]':
                vals = '0.1'
            else: 
                vals = '0.03'    
                
            left, right= st.columns([1,1])           
            with left:
                help_text = 'This is the max distance between every couple of cloest points from two scan station, which is used to determine sub-overlapping (ovl) between two scan stations'
                try:
                    dist_criterion = float(st.text_input(f'Select a distance criterion to determine sub-overlapping (ovl) data between two scan stations [{unit}]',value = f'{vals}',help=help_text))
                except:
                    dist_criterion = float(vals)

            with right:
                sel_ovl_data = st.multiselect('Select options to save overlapping data if needed [*this is optional]',['Full overlapping', 'Sub overlapping'],help='Either select one of them, both or none of them')

                write_ovl_data = [False,False]
                for i in sel_ovl_data:
                    if i == 'Full overlapping':
                        write_ovl_data[0]=True
                    if i == 'Sub overlapping':
                        write_ovl_data[1]=True

            key = st.form_submit_button('Run')        
        if key or (run_type == 'Run all'):
            star_time = time.time()
            message = evaluation_metrics(data_dir,dist_criterion,write_ovl_data,unit=unit,scan_type=scan_type)
            # message = f'<p style="font-family:sans-serif; color:Green; font-size: 13px;">Analyzing evaluation metrics is completed...[{round(time.time() - star_time,0)} s]!</p>'
            # st.markdown(message, unsafe_allow_html=True) 
            if message == True:
                st.write(f'Analyzing evaluation metrics took {int(time.time() - star_time)} seconds')  
            else:
                st.markdown(message, unsafe_allow_html=True) 

#####################
#      BOTTOM
#####################
def run_evaluation_metrics_bottom(data_dir,run_type='Manual',scan_type='External',unit='m'):
        
    with st.expander('Evaluation metric analyses '):
        # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        # unit = st.radio('Select an unit type for visualization', ['[m]','[ft]'])    
        with st.form('Evaluation metric '): 

            if unit == 'ft': #unit == '[ft]':
                vals = '0.1'
            else: 
                vals = '0.03'    
                
            left, right= st.columns([1,1])           
            with left:
                help_text = 'This is the max distance between every couple of cloest points from two scan station, which is used to determine sub-overlapping (ovl) between two scan stations'
                try:
                    dist_criterion = float(st.text_input(f'Select a distance criterion to determine sub-overlapping (ovl) data between two scan stations [{unit}]',value = f'{vals}',help=help_text))
                except:
                    dist_criterion = float(vals)

            with right:
                sel_ovl_data = st.multiselect('Select options to save overlapping data if needed [*this is optional]',['Full overlapping', 'Sub overlapping'],help='Either select one of them, both or none of them')

                write_ovl_data = [False,False]
                for i in sel_ovl_data:
                    if i == 'Full overlapping':
                        write_ovl_data[0]=True
                    if i == 'Sub overlapping':
                        write_ovl_data[1]=True

            key = st.form_submit_button('Run ')
        if key or (run_type == 'Run all'):
            star_time = time.time()
            message = evaluation_metrics(data_dir,dist_criterion,write_ovl_data,unit=unit,scan_type=scan_type)
            # message = f'<p style="font-family:sans-serif; color:Green; font-size: 13px;">Analyzing evaluation metrics is completed...[{round(time.time() - star_time,0)} s]!</p>'
            # st.markdown(message, unsafe_allow_html=True) 
            if message == True:
                st.write(f'Analyzing evaluation metrics took {int(time.time() - star_time)} seconds')  
            else:
                st.markdown(message, unsafe_allow_html=True) 