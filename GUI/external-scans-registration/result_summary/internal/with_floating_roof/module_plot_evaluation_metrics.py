import streamlit as st
import streamlit as st
from glob import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from common.utils import METER2FOOT


#####################
#       TOP
#####################
def plot_evaluation_metrics_top(data_dir,run_type='Manual',scan_type='External',unit='m'):
    
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
            button = st.button('Plot',key='Plot_top')
            if option == 'Full registration':
                if ((st.session_state.plot_type_top != option) and button) or (st.session_state.plot_type_top or button) or run_type=='Run all':
                    st.session_state.plot_type_top = option    
                    st.image(image, caption='')
            else:

                if ((st.session_state.sel_metric_top != key_sel) and button) or (st.session_state.plot_metric_top and (st.session_state.sel_metric_top == key_sel)):
                    st.session_state.sel_metric_top = key_sel                
                    st.session_state.plot_metric_top = True 
                    st.image(image, caption='')
                    
            if scan_type=='Internal':
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.</p>'
                st.markdown(message, unsafe_allow_html=True)
                # st.write('Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.')
        else:
            # st.write('Selected data is not available yet! Run "Evaluation metric analyses" first!...') 
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Evaluation metrics are not available yet! Please run "Evaluation metric analyses" first!...</p>'
            st.markdown(message, unsafe_allow_html=True)            


#####################
#       BOTTOM
#####################
def plot_evaluation_metrics_bottom(data_dir,run_type='Manual',scan_type='External',unit='m'):
    
    with st.expander('Plot evaluation metrics'):             
        
        if os.path.exists(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}')):
            option_bot = st.selectbox(
                'Select an option to plot evaluation metrics',
                ['Full registration ','Couple of stations ']
                )
            
            if option_bot == 'Full registration ':   
                try:  
                    image = Image.open(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}','Full registrations.png'))           
                except:
                    # st.write('Selected data is not available yet!')
                    message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Selected data is not available yet...</p>'
                    st.markdown(message, unsafe_allow_html=True)

            elif option_bot == 'Couple of stations ':
                png_files = glob(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}','*[&]*.*')) 
                couple_names = []
                for file in png_files:
                    couple_names.append(Path(file).stem)

                key_sel_bot = st.selectbox(
                    'Select a couple of stations',
                    range(len(couple_names)), 
                    format_func=lambda x: couple_names[x]
                    )  
                try:              
                    image = Image.open(os.path.join(data_dir,'evaluation','full_evaluation',f'{unit}',f'{couple_names[key_sel_bot]}.png'))
                except:
                    # st.write('Selected data is not available yet!')
                    message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Selected data is not available yet...</p>'
                    st.markdown(message, unsafe_allow_html=True)

            # plot when press button
            button_bot = st.button('Plot',key='Plot_bot')
            if option_bot == 'Full registration ':
                if ((st.session_state.plot_type_bot != option_bot) and button_bot) or (st.session_state.plot_type_bot or button_bot) or run_type=='Run all':
                    st.session_state.plot_type_bot = option_bot    
                    st.image(image, caption='')
            else:

                if ((st.session_state.sel_metric_bot != key_sel_bot) and button_bot) or (st.session_state.plot_metric_bot and (st.session_state.sel_metric_bot == key_sel_bot)):
                    st.session_state.sel_metric_bot = key_sel_bot                
                    st.session_state.plot_metric_bot = True 
                    st.image(image, caption='')
                    
            if scan_type=='Internal':
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.</p>'
                st.markdown(message, unsafe_allow_html=True)
                # st.write('Note that ovl is determined based on 5 slices along vertical direction with a thickness of 5 [cm] to avoid large computational cost.')
        else:
            # st.write('Selected data is not available yet! Run "Evaluation metric analyses" first!...') 
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Evaluation metrics are not available yet! Please run "Evaluation metric analyses" first!...</p>'
            st.markdown(message, unsafe_allow_html=True)         
