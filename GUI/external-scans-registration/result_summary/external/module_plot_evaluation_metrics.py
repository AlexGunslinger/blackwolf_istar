import streamlit as st
import streamlit as st
from glob import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from result_summary.utils import evaluation_metrics, load_and_combine_registation_data, plot_cross_section
from common.utils import METER2FOOT


def plot_evaluation_metrics(data_dir,run_type='Manual',scan_type='External',unit='m'):
    
    with st.expander('Plot evaluation metrics'):             
        
        if os.path.exists(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}')):
            option = st.selectbox(
                'Select an option to plot evaluation metrics',
                ['Full registration','Couple of stations']
                )
            
            if option == 'Full registration':   
                try:  
                    image = Image.open(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}','Full registrations.png'))           
                except:
                    # st.write('Selected data is not available yet!')
                    message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Selected data is not available yet...</p>'
                    st.markdown(message, unsafe_allow_html=True)

            elif option == 'Couple of stations':
                png_files = glob(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}','*[&]*.*')) 
                couple_names = []
                for file in png_files:
                    couple_names.append(Path(file).stem)

                key_sel = st.selectbox(
                    'Select a couple of stations',
                    range(len(couple_names)), 
                    format_func=lambda x: couple_names[x]
                    )  
                try:              
                    image = Image.open(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}',f'{couple_names[key_sel]}.png'))
                except:
                    # st.write('Selected data is not available yet!')
                    message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Selected data is not available yet...</p>'
                    st.markdown(message, unsafe_allow_html=True)

            # plot when press button
            button = st.button('Plot ')
            if option == 'Full registration':
                if ((st.session_state.plot_type != option) and button) or (st.session_state.plot_type or button) or run_type=='Run all':
                    st.session_state.plot_type = option  
                    st.image(image, caption='')
            else:

                if ((st.session_state.sel_metric != key_sel) and button) or (st.session_state.plot_metric and (st.session_state.sel_metric == key_sel)):
                    st.session_state.sel_metric = key_sel                
                    st.session_state.plot_metric = True 
                    st.image(image, caption='')
                    
            if scan_type=='Internal':
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.</p>'
                st.markdown(message, unsafe_allow_html=True)
                # st.write('Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.')
        else:
            # st.write('Selected data is not available yet! Run "Evaluation metric analyses" first!...') 
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Evaluation metrics are not available yet! Please run "Evaluation metric analyses" first!...</p>'
            st.markdown(message, unsafe_allow_html=True)            
    # except:
    #     pass
