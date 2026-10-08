
import streamlit as st
import streamlit as st
from glob import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt
import plotly
import numpy as np
import os
import open3d as o3d
import pickle
from PIL import Image
import math
from result_summary.utils import load_and_combine_registation_data, plot_cross_section
from common.utils import METER2FOOT
from External.source.utils.write_output import write_array2txt


#####################
#       TOP
#####################
def run_plot_vertical_cross_section_top(data_dir,run_type='Manual',unit='m'):

    with st.expander('Plot vertical cross-section'):
        # load a pcd file to define the high of the tank   
        try:   
            _, h_min, h_max = load_and_combine_registation_data(data_dir,get_hmax=True)
        except:                
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
            st.markdown(message, unsafe_allow_html=True)              
            return 
                        
        if unit == 'ft':
            vals = ['0', '0.015'] # starting angle and thickness
            h_max = math.trunc(h_max*METER2FOOT)
        else: 
            vals = ['0','0.005']

        try:
            png_files = glob(os.path.join(data_dir,'fit_cylinder*')) 
            file = open(png_files[0], 'rb')
            fit_info = pickle.load(file)
            file.close() 
            radius = fit_info[-1][3]
        except:
            # st.write('Data is not available yet!')
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Data is not available yet...</p>'
            st.markdown(message, unsafe_allow_html=True)    

        left, right, = st.columns([1,1])
        
        try:
            angle = int(left.text_input('Select a starting angle within [0, 360]\N{DEGREE SIGN}',value=vals[0]))
        except:
            angle = 0

        try:
            arc_length = float(right.text_input(f'Select cross-section thickness [{unit}]',value=vals[1]))
        except:
            arc_length = float(vals[1])
        
        elevation = None
        thickness = None
        
        theta = angle*np.pi/180 + np.pi/2
        if theta<np.pi:
            theta = np.r_[theta, theta+np.pi]
        else:
            theta = np.r_[theta, theta-np.pi]
        r = np.ones([len(theta),1])

        button_vertical = st.button('Plot',key='plot_top_v')
        if (st.session_state.plot_vertical_cross_sec_top or button_vertical or run_type=='Run all'):
            st.session_state.plot_vertical_cross_sec_top = True

            left, mid, right, = st.columns([1,4,0.5])

            # creat empty space 
            # left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            right.write('')

            figp, ax = plt.subplots(subplot_kw={'projection': 'polar'})     
            theta_labels = np.array([0, 45, 90, 135, 180, 225, 270, 315])*np.pi/180    
            ax.set_xticks(theta_labels.tolist(),['270\N{DEGREE SIGN}', '315\N{DEGREE SIGN}','0\N{DEGREE SIGN}', '45\N{DEGREE SIGN}', '90\N{DEGREE SIGN}', '135\N{DEGREE SIGN}', '180\N{DEGREE SIGN}', '225\N{DEGREE SIGN}'])       
            figp.set_size_inches(3,4)                
            ax.scatter(theta, r) 
            ax.set_rticks([0.00001,1.1], labels = '')  # Less radial ticks  

            try: 
                # get pcd data and define the max and min of the tank   
                pcd_full, _, _ = load_and_combine_registation_data(data_dir) 
            except:
                # st.write('Registration data is not available yet. Please run "Fine alignment fist!..."')
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
                st.markdown(message, unsafe_allow_html=True)              
                return 
            fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',width=1000,height=500,unit=unit) 
            # fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',unit=unit) 
            mid.plotly_chart(fig, use_container_width=True)
            left.pyplot(figp, use_container_width=True) 

            # to save figure and write data            
            left2,right2 = st.columns([5.5,1])
            if left2.button('Save figure ') or (run_type=='Run all'):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.png')
                fig.write_image(file)
                file2 = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Location_Vertical_angle={angle}_thickness={arc_length}.png')
                plt.savefig(file2)                

            if right2.button('Write cross-section data',key='write_v_top') or (run_type=='Run all'):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.txt')
                write_array2txt(file,cross_sec_data)
                
            #==============================================================
            # if run = Run all, plot another cross-section 
            #==============================================================
            if run_type=='Run all':
                # location of cross-section
                angle = 90
                theta = angle*np.pi/180 + np.pi/2
                if theta<np.pi:
                    theta = np.r_[theta, theta+np.pi]
                else:
                    theta = np.r_[theta, theta-np.pi]
                r = np.ones([len(theta),1])
                figp, ax = plt.subplots(subplot_kw={'projection': 'polar'})     
                theta_labels = np.array([0, 45, 90, 135, 180, 225, 270, 315])*np.pi/180    
                ax.set_xticks(theta_labels.tolist(),['270\N{DEGREE SIGN}', '315\N{DEGREE SIGN}','0\N{DEGREE SIGN}', '45\N{DEGREE SIGN}', '90\N{DEGREE SIGN}', '135\N{DEGREE SIGN}', '180\N{DEGREE SIGN}', '225\N{DEGREE SIGN}'])       
                figp.set_size_inches(3,4)                
                ax.scatter(theta, r) 
                ax.set_rticks([0.00001,1.1], labels = '')  # Less radial ticks

                # cross section
                fig,cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',width=1000,height=500,unit=unit)
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.png')
                fig.write_image(file)
                file2 = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Location_Vertical_angle={angle}_thickness={arc_length}.png')
                plt.savefig(file2) 

                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.txt')
                write_array2txt(file,cross_sec_data)

                
            # show where data/figure is stored   
            folder = os.path.join(data_dir,'evaluation','full_evaluation',unit) 
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note: figure/cross-section data are saved/written to: {folder}</p>'
            st.markdown(message, unsafe_allow_html=True)                



#####################
#       BOTTOM
#####################
def run_plot_vertical_cross_section_bottom(data_dir,run_type='Manual',unit='m'):

    with st.expander('Plot vertical cross-section'):
        # load a pcd file to define the high of the tank   
        try:   
            _, h_min, h_max = load_and_combine_registation_data(data_dir,get_hmax=True)
        except:                
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
            st.markdown(message, unsafe_allow_html=True)              
            return 
                        
        if unit == 'ft':
            vals = ['0', '0.015'] # starting angle and thickness
            h_max = math.trunc(h_max*METER2FOOT)
        else: 
            vals = ['0','0.005']

        try:
            png_files = glob(os.path.join(data_dir,'fit_cylinder*')) 
            file = open(png_files[0], 'rb')
            fit_info = pickle.load(file)
            file.close() 
            radius = fit_info[-1][3]
        except:
            # st.write('Data is not available yet!')
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Data is not available yet...</p>'
            st.markdown(message, unsafe_allow_html=True)    

        left, right, = st.columns([1,1])
        
        try:
            angle = int(left.text_input('Select a starting angle within [0, 360]\N{DEGREE SIGN}',value=vals[0],key='angle_bot'))
        except:
            angle = 0

        try:
            arc_length = float(right.text_input(f'Select cross-section thickness [{unit}]',value=vals[1],key='arc_length_bot'))
        except:
            arc_length = float(vals[1])
        
        elevation = None
        thickness = None
        
        theta = angle*np.pi/180 + np.pi/2
        if theta<np.pi:
            theta = np.r_[theta, theta+np.pi]
        else:
            theta = np.r_[theta, theta-np.pi]
        r = np.ones([len(theta),1])

        button_vertical_bot = st.button('Plot',key='plot_bot_v')
        if (st.session_state.plot_vertical_cross_sec_bot or button_vertical_bot or run_type=='Run all'):
            st.session_state.plot_vertical_cross_sec_bot = True

            left, mid, right, = st.columns([1,4,0.5])

            # creat empty space 
            # left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            left.write('')
            right.write('')

            figp, ax = plt.subplots(subplot_kw={'projection': 'polar'})     
            theta_labels = np.array([0, 45, 90, 135, 180, 225, 270, 315])*np.pi/180    
            ax.set_xticks(theta_labels.tolist(),['270\N{DEGREE SIGN}', '315\N{DEGREE SIGN}','0\N{DEGREE SIGN}', '45\N{DEGREE SIGN}', '90\N{DEGREE SIGN}', '135\N{DEGREE SIGN}', '180\N{DEGREE SIGN}', '225\N{DEGREE SIGN}'])       
            figp.set_size_inches(3,4)                
            ax.scatter(theta, r) 
            ax.set_rticks([0.00001,1.1], labels = '')  # Less radial ticks  

            try: 
                # get pcd data and define the max and min of the tank   
                pcd_full, _, _ = load_and_combine_registation_data(data_dir) 
            except:
                # st.write('Registration data is not available yet. Please run "Fine alignment fist!..."')
                message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Registration data is not available yet. Please run "Fine alignment" fist!...</p>'
                st.markdown(message, unsafe_allow_html=True)              
                return 
            fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',width=1000,height=500,unit=unit) 
            # fig, cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',unit=unit) 
            mid.plotly_chart(fig, use_container_width=True)
            left.pyplot(figp, use_container_width=True) 

            # to save figure and write data            
            left2,right2 = st.columns([5.5,1])
            if left2.button('Save figure',key='save_fig_v_bot') or (run_type=='Run all'):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.png')
                fig.write_image(file)
                file2 = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Location_Vertical_angle={angle}_thickness={arc_length}.png')
                plt.savefig(file2)                

            if right2.button('Write cross-section data',key='write_v_bot') or (run_type=='Run all'):
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.txt')
                write_array2txt(file,cross_sec_data)
                
            #==============================================================
            # if run = Run all, plot another cross-section 
            #==============================================================
            if run_type=='Run all':
                # location of cross-section
                angle = 90
                theta = angle*np.pi/180 + np.pi/2
                if theta<np.pi:
                    theta = np.r_[theta, theta+np.pi]
                else:
                    theta = np.r_[theta, theta-np.pi]
                r = np.ones([len(theta),1])
                figp, ax = plt.subplots(subplot_kw={'projection': 'polar'})     
                theta_labels = np.array([0, 45, 90, 135, 180, 225, 270, 315])*np.pi/180    
                ax.set_xticks(theta_labels.tolist(),['270\N{DEGREE SIGN}', '315\N{DEGREE SIGN}','0\N{DEGREE SIGN}', '45\N{DEGREE SIGN}', '90\N{DEGREE SIGN}', '135\N{DEGREE SIGN}', '180\N{DEGREE SIGN}', '225\N{DEGREE SIGN}'])       
                figp.set_size_inches(3,4)                
                ax.scatter(theta, r) 
                ax.set_rticks([0.00001,1.1], labels = '')  # Less radial ticks

                # cross section
                fig,cross_sec_data = plot_cross_section(data_dir=data_dir,pcd_full=pcd_full,eleva=elevation,thickness=thickness,angle=angle,arc_length=arc_length,radius=radius,type='Vertical',width=1000,height=500,unit=unit)
                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.png')
                fig.write_image(file)
                file2 = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Location_Vertical_angle={angle}_thickness={arc_length}.png')
                plt.savefig(file2) 

                file = os.path.join(data_dir,'evaluation','full_evaluation',unit,f'CrossSec_Vertical_angle={angle}_thickness={arc_length}.txt')
                write_array2txt(file,cross_sec_data)

                
            # show where data/figure is stored   
            folder = os.path.join(data_dir,'evaluation','full_evaluation',unit) 
            message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Note: figure/cross-section data are saved/written to: {folder}</p>'
            st.markdown(message, unsafe_allow_html=True)                
    