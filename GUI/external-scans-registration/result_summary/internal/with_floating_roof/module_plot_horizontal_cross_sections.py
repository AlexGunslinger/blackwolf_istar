
import streamlit as st
import streamlit as st
from glob import glob
from pathlib import Path
import os
from PIL import Image
import math
from result_summary.utils import load_and_combine_registation_data, plot_cross_section
from common.utils import METER2FOOT
from External.source.utils.write_output import write_array2txt


#####################
#       TOP
#####################
def run_plot_horizontal_cross_section_top(data_dir,run_type='Manual',unit='m'):

    with st.expander('Plot horizontal cross-section'):
        # load a pcd file to define the high of the tank   
        try:   
            _, h_min, h_max = load_and_combine_registation_data(data_dir,get_hmax=True)
        except:                
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
            st.markdown(message, unsafe_allow_html=True)              
            return 

        if unit == 'ft':
            vals = [str(round(h_max*2.0/3.0,1)),'0.03']
            h_max = math.trunc(h_max*METER2FOOT)
        else: 
            vals = [str(round(h_max*2.0/3.0,1)),'0.01']
        # message = '<p style="font-family:sans-serif; color:Green; font-size: 13px;">Data preparation is completed!</p>'
        # st.markdown(message, unsafe_allow_html=True)
        
        left, right, = st.columns([1,1])        
        try: # define elevation
            elevation = float(left.text_input(f'Select an elevation within [{h_min}, {h_max}] [{unit}]',value=vals[0])) # elevation = left.number_input(f'Select an elevation within [{0}, {h_max}] [m]',value=1.0,format="%.1f",step=0.5)
        except:
            elevation = float(vals[0])
        
        try: # define thickness
            thickness = float(right.text_input(f'Select cross-section thickness [{unit}]',value=vals[1]))
        except:
            thickness = float(vals[1])          
        
        angle = None
        arc_length = None
        radius = None

        button_horizontal = st.button('Plot',key='plot_top')
        if (st.session_state.plot_horizontal_cross_sec_top or button_horizontal) or (run_type=='Run all'):
            st.session_state.plot_horizontal_cross_sec_top = True

            left, mid, right, = st.columns([0.5,4,0.5])
            left.write('')
            right.write('')

            try: 
                # get pcd data and define the max and min of the tank   
                pcd_full, _, _ = load_and_combine_registation_data(data_dir) 
            except:                
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
                st.markdown(message, unsafe_allow_html=True)              
                return 

            fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Horizontal',unit=unit)
            mid.plotly_chart(fig, use_container_width=True)
            
            # to save figure and write data
            left2,right2 = st.columns([5.5,1])
            folder = os.path.join(data_dir,'evaluation','full_evaluation',unit)
            if left2.button('Save figure',key='save_fig_h_top') or (run_type=='Run all'):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.png')
                fig.write_image(file)                
                # st.write(f'Note: figure/cross-section data are saved/written to {folder}')

            if right2.button('Write cross-section data',key='write_h_top') or (run_type=='Run all'): 
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.txt')
                write_array2txt(file,cross_sec_data)
                # st.write(f'Note: figure/cross-section data are saved/written to {folder}')

            #==============================================================
            # if run = auto, plot another cross-section at 2/3 high of tank
            #==============================================================
            if run_type=='Run all':
                elevation = round(h_max*2.0/3.0,1)
                fig,cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Horizontal',unit=unit)
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.png')
                fig.write_image(file) 

                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.txt')
                write_array2txt(file,cross_sec_data)
                folder = os.path.join(data_dir,'evaluation','full_evaluation',unit)

            # show where data/figure is stored    
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note: figure/cross-section data are saved/written to: {folder}</p>'
            st.markdown(message, unsafe_allow_html=True)
            
                

#####################
#       BOTTOM
#####################
def run_plot_horizontal_cross_section_bottom(data_dir,run_type='Manual',unit='m'):

    with st.expander('Plot horizontal cross-section'):
        # load a pcd file to define the high of the tank   
        try:   
            _, h_min, h_max = load_and_combine_registation_data(data_dir,get_hmax=True)
        except:                
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
            st.markdown(message, unsafe_allow_html=True)              
            return 

        if unit == 'ft':
            vals = [str(round(h_max*2.0/3.0,1)),'0.03']
            h_max = math.trunc(h_max*METER2FOOT)
        else: 
            vals = [str(round(h_max*2.0/3.0,1)),'0.01']
        # message = '<p style="font-family:sans-serif; color:Green; font-size: 13px;">Data preparation is completed!</p>'
        # st.markdown(message, unsafe_allow_html=True)
        
        left, right, = st.columns([1,1])        
        try: # define elevation
            elevation = float(left.text_input(f'Select an elevation within [{h_min}, {h_max}] [{unit}]',value=vals[0],key='elevation_bot')) # elevation = left.number_input(f'Select an elevation within [{0}, {h_max}] [m]',value=1.0,format="%.1f",step=0.5)
        except:
            elevation = float(vals[0])
        
        try: # define thickness
            thickness = float(right.text_input(f'Select cross-section thickness [{unit}]',value=vals[1],key='thickness_bot'))
        except:
            thickness = float(vals[1])          
        
        angle = None
        arc_length = None
        radius = None

        button_horizontal_bot = st.button('Plot',key='plot_bot')
        if (st.session_state.plot_horizontal_cross_sec_bot or button_horizontal_bot) or (run_type=='Run all'):
            st.session_state.plot_horizontal_cross_sec_bot = True

            left, mid, right, = st.columns([0.5,4,0.5])
            left.write('')
            right.write('')

            try: 
                # get pcd data and define the max and min of the tank   
                pcd_full, _, _ = load_and_combine_registation_data(data_dir) 
            except:                
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
                st.markdown(message, unsafe_allow_html=True)              
                return 

            fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Horizontal',unit=unit)
            mid.plotly_chart(fig, use_container_width=True)
            
            # to save figure and write data
            left2,right2 = st.columns([5.5,1])
            folder = os.path.join(data_dir,'evaluation','full_evaluation',unit)
            if left2.button('Save figure',key='save_fig_h_bot') or (run_type=='Run all '):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.png')
                fig.write_image(file)                
                # st.write(f'Note: figure/cross-section data are saved/written to {folder}')

            if right2.button('Write cross-section data',key='write_h_bot') or (run_type=='Run all'): 
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.txt')
                write_array2txt(file,cross_sec_data)
                # st.write(f'Note: figure/cross-section data are saved/written to {folder}')

            #==============================================================
            # if run = auto, plot another cross-section at 2/3 high of tank
            #==============================================================
            if run_type=='Run all':
                elevation = round(h_max*2.0/3.0,1)
                fig,cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Horizontal',unit=unit)
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.png')
                fig.write_image(file) 

                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Horizontal_elevation={elevation}_thickness={thickness}.txt')
                write_array2txt(file,cross_sec_data)
                folder = os.path.join(data_dir,'evaluation','full_evaluation',unit)

            # show where data/figure is stored    
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note: figure/cross-section data are saved/written to: {folder}</p>'
            st.markdown(message, unsafe_allow_html=True)