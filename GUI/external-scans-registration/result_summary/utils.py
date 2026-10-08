import streamlit as st
from multiprocessing import Pool
import time
import logging
import sys
import os
import pandas as pd
import plotly.express as px
from glob import glob
import numpy as np
from pathlib import Path
import open3d as o3d
import math
import matplotlib.pyplot as plt
import re
import pickle

from common.utils import FOOT2METER, METER2FOOT, extract_station_ids
# sys.path.append('D:\\PointCloudRegistration\\codes\\User-Interfaces\\Point-Cloud-Registration\\source')
# sys.path.append('D:\\PointCloudRegistration\\codes\\User-Interfaces\\Point-Cloud-Registration')

from External.source.utils.math_methods import cart2pol, rotate_around_z
from External.source.evaluation.utils_evaluation import plot_thin_slice, evaluation_and_visualization 


@st.cache()
def evaluation_metrics(data_dir,dist_criterion,write_ovl_data,unit,scan_type='External'):     
    full_list_registration_files = glob(os.path.join(data_dir, '*.pcd'))
    # convert feet to meter to determine ovl
    if unit == 'ft':
        dist_criterion = dist_criterion*FOOT2METER

    # only run evaluation metrics for internal with a number of slices to reduce compuational time
    if scan_type=='Internal':
        approach='slice'
    else:
        approach='full'

    if len(full_list_registration_files) > 1:
        evaluation_and_visualization(data_dir,full_list_registration_files=full_list_registration_files,dist_criterion=dist_criterion,write_ovl_data=write_ovl_data,approach=approach,plot=True,unit=unit,pack_num=4,scan_type=scan_type,run_parallel=True)    
        message = True
    else:        
        message = f'<p style="font-family:sans-serif; color:Orange; font-size: 13px;">Data for running metric evaluation is not available yet. Please run "Fine alignment first!...</p>'
    return message
    
    # st_data_full_ovl, st_data_sub_ovl, couple_stations_full_ovl, couple_stations_sub_ovl, full_cross_sec_data, z_slice  = evaluation_and_visualization(data_dir,full_list_registration_files=full_list_registration_files,dist_criterion=dist_criterion,write_ovl_data=write_ovl_data,plot=True,unit=unit,pack_num=16)
    # # save all the data
    # filenames = ['st_data_full_ovl', 'st_data_sub_ovl', 'couple_stations_full_ovl', 'couple_stations_sub_ovl', 'full_cross_sec_data', 'z_slice']
    # for i_val in filenames:
    #     file = open(os.path.join(data_dir,'evaluation',i_val), 'wb')
    #     # dump information to that file
    #     pickle.dump(eval(i_val), file)
    #     # close the file
    #     file.close()  
    # return st_data_full_ovl, st_data_sub_ovl, couple_stations_full_ovl, couple_stations_sub_ovl, full_cross_sec_data, z_slice

def load_pcd_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    return pcd

@st.cache()
def load_and_combine_registation_data(data_dir,get_hmax=False):
    full_list = glob(os.path.join(data_dir, '*.pcd'))

    try:
        png_files = glob(os.path.join(data_dir,'fit_cylinder*')) 
        file = open(png_files[0], 'rb')
        fit_info = pickle.load(file)
        file.close() 
        radius = fit_info[-1][3]
    except:
        radius = []
    

    if get_hmax:
        full_list = [full_list[0]]

    pcd_full = np.empty(shape=[0,4])
    for id,file in enumerate(full_list):
        pcd = load_pcd_file(os.path.join(data_dir,Path(file).name))        
        A = np.asarray(pcd.points)
        B = np.c_[ A, np.ones( A.shape[0] )*id ]
        pcd_full = np.r_[pcd_full,B]
    
    if radius:
        pcd_full = pcd_full[np.max(abs(pcd_full[:,0:2]),axis=1) < (radius + 1.0),:]
    h_min = math.trunc(np.min(pcd_full[:,2]))
    h_max = math.trunc(np.max(pcd_full[:,2]))
    return pcd_full, h_min, h_max

@st.cache()
def data_cart2pol(pcd_full):
    rho, phi = cart2pol(pcd_full[:,0], pcd_full[:,1])
    return rho, phi

@st.cache()
def plot_cross_section(data_dir=None,pcd_full=None,eleva=1,thickness=0.01,type='Horizontal',angle=0,arc_length=0.01,radius=None,width=1200,height=900,unit='m'):
    full_list = glob(os.path.join(data_dir, '*.pcd'))
    file_names = [Path(full_list[i]).stem for i in range(len(full_list))]
    file_names = extract_station_ids(file_names, sorted=True)
    file_names = file_names[0].tolist()
    if type == 'Horizontal':
        ind = ((eleva <= pcd_full[:,2]) & (pcd_full[:,2] <= eleva + thickness))   
        cross_sec_data = pcd_full[ind,:]
        if unit == 'ft':
            cross_sec_data[:,0:3] = cross_sec_data[:,0:3]*METER2FOOT

        cs_df = pd.DataFrame(cross_sec_data, columns=['x', 'y', 'z', 'tank_id'])
        # cs_df['Station labels'] = cs_df.tank_id.apply(lambda i: Path(full_list[int(i)]).stem)
        cs_df['Scan label'] = cs_df.tank_id.apply(lambda i: file_names[int(i)])
        fig = plot_thin_slice(cs_df,f'Horizontal cross-section at elevation = {eleva} [{unit}] & thickness = {thickness} [{unit}]',unit=unit)

    elif type == 'Vertical': 
        rho, phi = data_cart2pol(pcd_full)
        #cart2pol(pcd_full[:,0], pcd_full[:,1])
        in_angle = arc_length/radius*180/np.pi 

        phi = phi*180/np.pi
        id1 = ((angle - in_angle/2 <= phi) & (phi <= angle + in_angle/2)) 

        if angle < 180:
            id2 = ((angle - in_angle/2 + 180 <= phi) & (phi <= angle + 180 + in_angle/2))
            angle2 = angle + 180
        else:
            id2 = ((angle - in_angle/2 - 180 <= phi) & (phi <= angle - 180 + in_angle/2))
            angle2 = angle - 180
        ind = id1 + id2

        cross_sec_data = pcd_full[ind,:]        
        cross_sec_data[:,:-1] = rotate_around_z(cross_sec_data[:,:-1], angle)

        text_pos = np.array([1.5,2])
        if unit == 'ft':
            cross_sec_data[:,0:3] = cross_sec_data[:,0:3]*METER2FOOT
            text_pos = text_pos*METER2FOOT

        cs_df = pd.DataFrame(cross_sec_data, columns=['x', 'z', 'y', 'tank_id'])
        # cs_df['Station labels'] = cs_df.tank_id.apply(lambda i: Path(full_list[int(i)]).stem)
        cs_df['Scan label'] = cs_df.tank_id.apply(lambda i: file_names[int(i)])
        fig = plot_thin_slice(cs_df,f'Vertical cross-section at angle = [{round(angle)}, {round(angle2)}]\N{DEGREE SIGN} & thickness = {arc_length} [{unit}]',width=width,height=height,unit=unit)
        
            
        # elevation 0.5-1m
        ind = (cs_df.y > text_pos[0]) & (cs_df.y < text_pos[1])
        # x_right = cs_df.x[ind].max()
        # x_left = cs_df.x[ind].min()
        x_right =  radius + 0.0*radius
        x_left  = -radius - 0.0*radius
        fig.add_annotation(dict(font=dict(color='white',size=15),
                                                x=x_left,
                                                y=-text_pos[0],
                                                showarrow=False,
                                                text=f'  {str(round(angle2))}\N{DEGREE SIGN}  ',
                                                bgcolor='#636363'))

        fig.add_annotation(dict(font=dict(color='white',size=15),
                                                bgcolor='#636363',
                                                x=x_right,
                                                y=-text_pos[0],
                                                showarrow=False,
                                                text=f'  {str(round(angle))}\N{DEGREE SIGN}  ')) 

                                                
        # fig['layout']['xaxis'].update(range=[cs_df.x.min()-4, cs_df.x.max()+4], autorange=False)
        # fig['layout']['xaxis'].update(autorange=True)
        # fig.update_yaxes(range=[0, 0.4])
        # fig.update_layout(xaxis_range=[cs_df.x.min()-0.5, cs_df.x.max()+0.5])
        # fig.update_xaxes(range = [cs_df.x.min()-0.5, cs_df.x.max()+0.5])

        # fig.update_xaxes(range = [cs_df.x.min()-0.5, cs_df.x.max()+0.5],
        #                 scaleanchor = "y",
        #                 scaleratio = 1,
        #                 tickfont_size=13,
        #                 )                     
        # fig.update_layout(xaxis_range=[-30, 30])
        # xaxis_range=[cs_df.x.min()-0.5, cs_df.x.max()+0.5]

    return fig, cross_sec_data


