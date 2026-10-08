
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import numpy as np
from matplotlib import gridspec
import pickle
import re
from shapely.geometry import Polygon, Point
import matplotlib.path as mpltPath
import plotly.express as px
import open3d as o3d
import time
import numpy as np
import matplotlib.pyplot as plt

from ..cylinder_extraction.cylinder import Line
from ..cylinder_extraction import config
from ..cylinder_extraction.utils import fit_cylinder
from ..utils.write_output import write_array2txt
from ..evaluation.geometry_2Dmbb import min_2Dbounding_box

METER2FOOT = 3.2808399
FOOT2METER = 0.3048

def find_closest_pts(src_pts, trg_pts,method='open3d'):
    """
    Demo:
        src_pts = np.array([[1, 2, 3], [2, 3, 4], [2, 3, 5], [2, 3, 4]])
        trg_pts = np.array([[2,3,1], [2,3,3]])
        
        src_pts = candidate_roof_pts_xyz
        trg_pts = roof_pts_xyz
    """

    leaf_size = 1000
    tree = KDTree(src_pts, leaf_size = leaf_size)
    distance, neighbour_pts_ids = tree.query(trg_pts, k = 1)
        
    return neighbour_pts_ids, distance

def calculate_statistical_data(ovl_trg_pts,ovl_src_pts,cyl_trg_info):
    # calculate delta between two point clouds in x, y, z direction and distance between 2 points cloud
    delta_xyz = ovl_trg_pts - ovl_src_pts
    distance = np.sqrt(np.sum(delta_xyz*delta_xyz, axis=1))
    p2p_data = np.hstack((delta_xyz, distance.reshape(-1, 1)))

    # save statistical data based on closest points
    statis_p2p_data = np.r_[
            np.min(p2p_data,axis=0).reshape([1,4]),
            np.max(p2p_data,axis=0).reshape([1,4]),
            np.mean(p2p_data,axis=0).reshape([1,4]),
            np.std(p2p_data,axis=0).reshape([1,4]),
            np.median(p2p_data,axis=0).reshape([1,4]),
            np.linalg.norm(p2p_data,axis=0).reshape([1,4])/np.sqrt(len(p2p_data))
            ]       

    ovltrg_2_clytrg = Line(cyl_trg_info[:3],cyl_trg_info[-3:]).distance_to_point(ovl_src_pts)
    ovlsrc_2_clytrg = Line(cyl_trg_info[:3],cyl_trg_info[-3:]).distance_to_point(ovl_trg_pts)

    # calculate the delta of distance from points to axes of two cylinders
    p2cyl_del_data = ovltrg_2_clytrg - ovlsrc_2_clytrg
    # save statistical data based on points to cylinder axes
    # p2cyl_del_data_ = abs(p2cyl_del_data)
    statis_p2cyl_data = np.r_[
                    np.min(p2cyl_del_data),
                    np.max(p2cyl_del_data),
                    np.mean(abs(p2cyl_del_data)),
                    np.std(p2cyl_del_data),
                    np.median(p2cyl_del_data),
                    np.linalg.norm(p2cyl_del_data)/np.sqrt(len(p2cyl_del_data))
                    ]                        
    
    return p2p_data, p2cyl_del_data, statis_p2p_data, statis_p2cyl_data

def registration_evaluation(data_dir,trg_filename,src_filename,cyl_trg_info=None,approach='full',nslice=5,dist_criterion=0.02,write_full_ovl_data=False,write_sub_ovl_data=False,scan_type='External'):
    
    pcd_trg = o3d.io.read_point_cloud(os.path.join(data_dir,trg_filename))
    pcd_src = o3d.io.read_point_cloud(os.path.join(data_dir,src_filename))
    points_trg = np.asarray(pcd_trg.points)
    points_src = np.asarray(pcd_src.points)    
    # get points 0.5m above the ground 
    id1 = points_trg[:,-1] > (np.min(points_trg[:,-1])+0.5)
    points_trg = points_trg[id1]
    id2 = points_src[:,-1] > (np.min(points_src[:,-1])+0.5)
    points_src = points_src[id2]
    
    # take nslices of cross-section for visualization
    h_min = np.min(np.r_[points_trg[:,-1],points_src[:,-1]])
    h_max = np.max(np.r_[points_trg[:,-1],points_src[:,-1]])
    increasment = round((h_max - h_min)/nslice,0)
    l = round(h_min + h_min/3,0) 
    thickness = 0.01
    cross_sec_data = {f'{trg_filename}':[],
                      f'{src_filename}':[]}
    z_slice = [l]
    for i in range(0,nslice):
        # print(f'SLICE {i}')
        ind1 = ((l <= points_trg[:,2]) & (points_trg[:,2] <= l + thickness))        
        cross_sec_data[f'{trg_filename}'].append(points_trg[ind1,:])
        ind2 = ((l <= points_src[:,2]) & (points_src[:,2] <= l + thickness))
        cross_sec_data[f'{src_filename}'].append(points_src[ind2,:]) 
        l = l + increasment
        z_slice.append(l)

    # if use slices to calculate evaluation metrics
    if approach == 'slice':
        print(f'****** Number of slice = {nslice} ******')
        p1=np.empty([0,3])
        p2=np.empty([0,3])
        # h_min = np.min(np.r_[points_trg[:,-1],points_src[:,-1]])
        # h_max = np.max(np.r_[points_trg[:,-1],points_src[:,-1]])
        # increasment = (h_max - h_min)/nslice
        l = round(h_min + h_min/3,0) 
        thickness = 0.05
        for i in range(0,nslice):
            #print(f'SLICE {i}')
            ind1 = ((l <= points_trg[:,2]) & (points_trg[:,2] <= l + thickness))      
            p1 = np.r_[p1,points_trg[ind1,:]]
            ind2 = ((l <= points_src[:,2]) & (points_src[:,2] <= l + thickness))
            p2 = np.r_[p2,points_src[ind2,:]] 
            l = l + increasment
        points_trg = p1
        points_src = p2 

    if scan_type=='External':
        # define polygon based on min bounding box of data
        poly_trg = Polygon(min_2Dbounding_box(points_trg)['rectangle_corners'])
        poly_src = Polygon(min_2Dbounding_box(points_src)['rectangle_corners'])
        # define intersection between two polygons
        intersection = poly_src.intersection(poly_trg)
        # print("Define intersection polygon, elapsed time: " + str(time.time()-start_time))

        # define overlaping data using Matplotlib mplPath
        polygon = list(intersection.exterior.coords)
        path_poly = mpltPath.Path(polygon)
        inside1 = path_poly.contains_points(points_trg[:,:-1])
        inside2 = path_poly.contains_points(points_src[:,:-1])
        ovl_trg_pts = points_trg[inside1]
        ovl_src_pts = points_src[inside2]
        # print("Define ovl data using Matplotlib, elapsed time: " + str(time.time()-start_time))

    elif scan_type=='Internal': # for internal scan both stations are full ovl
        ovl_trg_pts = points_trg
        ovl_src_pts = points_src

    if (len(ovl_trg_pts) > 1 and len(ovl_src_pts) > 1):
        # find set of closest points
        cl_points,dist = find_closest_pts(ovl_src_pts, ovl_trg_pts)   
        # extract the couple of closest points < 2 cm
        id = dist <= dist_criterion
        sub_cl_points = cl_points[id.flatten()]
        sub_ovl_trg_pts = ovl_trg_pts[id.flatten()] 
        sub_ovl_src_pts = ovl_src_pts[sub_cl_points.flatten()]    

        # distance closest points to the target cylinder
        if cyl_trg_info is None:
            if os.path.exists(os.path.join(data_dir,f'fit_cylinder_{trg_filename[:-4]}.pkl')):
                file = open(os.path.join(data_dir,f'fit_cylinder_{trg_filename[:-4]}.pkl'), 'rb')

                print('Read cylinder fit information')
                # load information to that file
                cyl_trg_fit = pickle.load(file)
                # close the file
                file.close() 
                cyl_trg_info = cyl_trg_fit[-1]
                print(cyl_trg_info)

            else:
                print('Fit cylinder')
                # cyl_trg_fit  = fit_cylinder(np.asarray(pcd_trg.points),threshold=config.threshold)
                cyl_trg_fit  = fit_cylinder(ovl_trg_pts,threshold=config.threshold)
                cyl_trg_info = cyl_trg_fit[-1]
                
                # save fit information for next time
                file = open(os.path.join(data_dir,f'fit_cylinder_{trg_filename[:-4]}.pkl'), 'wb')
                print('Save cylinder fit information')
                # dump information to that file
                pickle.dump(cyl_trg_fit, file)
                # close the file
                file.close()  
                
        else:
            cyl_trg_info = cyl_trg_fit[-1]
        
        if isinstance(cyl_trg_info,list):
            cyl_trg_info = np.array(cyl_trg_info)

        # data for sub overlap
        p2p_data, p2cyl_del_data, statis_p2p_data, statis_p2cyl_data = calculate_statistical_data(sub_ovl_trg_pts,sub_ovl_src_pts,cyl_trg_info)
        st_data_sub_ovl = {
            'p2p':p2p_data,
            'p2cyl_del':p2cyl_del_data,
            'statistical_p2p':statis_p2p_data,
            'statistical_p2cyl':statis_p2cyl_data,
            'trg_filename':trg_filename,
            'src_filename':trg_filename
                        }
        # data for full overlap                
        p2p_data, p2cyl_del_data, statis_p2p_data, statis_p2cyl_data = calculate_statistical_data(ovl_trg_pts,ovl_src_pts[cl_points.flatten()],cyl_trg_info)
        st_data_full_ovl = {
            'p2p':p2p_data,
            'p2cyl_del':p2cyl_del_data,
            'statistical_p2p':statis_p2p_data,
            'statistical_p2cyl':statis_p2cyl_data,
            'trg_filename':trg_filename,
            'src_filename':trg_filename
                        }

        write_dir = os.path.join(data_dir, 'evaluation', 'ovl_data')
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)  

        if write_full_ovl_data:      
            pcd0 = o3d.geometry.PointCloud()
            pcd1 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(ovl_trg_pts)
            pcd1.points = o3d.utility.Vector3dVector(ovl_src_pts)

            o3d.io.write_point_cloud(os.path.join(write_dir, trg_filename[:-4] + '-[&]-' + src_filename[:-4] + '_full_ovl.pcd'), pcd0)
            o3d.io.write_point_cloud(os.path.join(write_dir, src_filename[:-4] + '-[&]-' + trg_filename[:-4] + '_full_ovl.pcd'), pcd1)

        if write_sub_ovl_data:
            pcd0 = o3d.geometry.PointCloud()
            pcd1 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(sub_ovl_trg_pts)
            pcd1.points = o3d.utility.Vector3dVector(sub_ovl_src_pts)

            o3d.io.write_point_cloud(os.path.join(write_dir, trg_filename[:-4] + '-[&]-' + src_filename[:-4] + '_sub_ovl.pcd'), pcd0)
            o3d.io.write_point_cloud(os.path.join(write_dir, src_filename[:-4] + '-[&]-' + trg_filename[:-4] + '_sub_ovl.pcd'), pcd1)
    else:
        st_data_sub_ovl  = []
        st_data_full_ovl = []

    return st_data_full_ovl, st_data_sub_ovl, cross_sec_data, z_slice

def plot_evaluation_data(st_data_full_ovl,st_data_sub_ovl,tt_name,dist_criterion=None,all_metrics=False,unit='m'):
    
    if len(st_data_full_ovl): # only calculate when there is ovl data
        if unit == 'ft':
            t_unit = 'ft'
            # convert data from meter to feet
            st_data_full_ovl['statistical_p2p'] = st_data_full_ovl['statistical_p2p']*METER2FOOT        
            st_data_full_ovl['p2p']             = st_data_full_ovl['p2p']*METER2FOOT
            st_data_full_ovl['p2cyl_del']       = st_data_full_ovl['p2cyl_del']*METER2FOOT

            st_data_sub_ovl['statistical_p2p']  = st_data_sub_ovl['statistical_p2p']*METER2FOOT
            st_data_sub_ovl['p2p']              = st_data_sub_ovl['p2p']*METER2FOOT
            st_data_sub_ovl['p2cyl_del']        = st_data_sub_ovl['p2cyl_del']*METER2FOOT

            number        = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
            number_labels = number_labels = number*12 # np.array([0.12, 0.36, 0.60 , 0.84, 1.08]) # 
            col_labels_=['Number of ovl points',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[0])} [in]',
                        f'Percentage of ovl points <= {"{:.2f}".format(number_labels[1])} [in]',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[2])} [in]',
                        f'Percentage of ovl points <= {"{:.2f}".format(number_labels[3])} [in]',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[4])} [in]']
            footnote = f'*sub ovl is determined based on the distance bettwen two closest points <= {round(dist_criterion*3.2808399,3)} [ft]'        
            # define small ovl 
            s_ovl = 0.016 
            nbin  = 0.01*METER2FOOT
        else:
            t_unit = 'm'
            number_labels = np.array([0.5, 1.0, 1.5, 2.5, 3.0])
            number        = number_labels/100
            col_labels_=['Number of ovl points',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[0])} [cm]',
                        f'Percentage of ovl points <= {"{:.2f}".format(number_labels[1])} [cm]',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[2])} [cm]',
                        f'Percentage of ovl points <= {"{:.2f}".format(number_labels[3])} [cm]',f'Percentage of ovl points <= {"{:.2f}".format(number_labels[4])} [cm]']
            footnote = f'*sub ovl is determined based on the distance bettwen two closest points <= {round(dist_criterion,3)} [m]'     
            s_ovl = 0.005
            nbin  = 0.01

        col_labels=['min','max','mean','std','median','rmse']
        row_labels=['full ovl','sub ovl']
        title     =['Delta X (full ovl)','Delta Y (full ovl)','Delta Z (full ovl)','Distance (full ovl)','Delta_P2Cly (full ovl)']
        colors    =['b','g','c','r','y']
        xlabel    =[f'Distance between two closest points in X direction [{t_unit}]',f'Distance between two closest points in Y direction [{t_unit}]',f'Distance between two closest points in Z direction [{t_unit}]',
                    f'Abs distance between two closest points [{t_unit}]', f'Delta between two closest points to the reference cylinder [{t_unit}]']
        ylabel    =['Frequency of data falls into a certain delta group', 'Frequency of data falls into a certain delta group', 'Frequency of data falls into a certain delta group',
                    'Frequency of data falls into a certain distance group', 'Frequency of data falls into a certain delta group']
        
        fig = plt.figure()
        #plt.subplots(1,3,figsize=(28,7), gridspec_kw={'width_ratios': [1.3,1,1]})    
        if all_metrics:
            fig.set_size_inches(28,14)
            fig_id = [4,5,6,2,3]
            start_id = 1
            subplot = [2,3]
        else:
            fig.set_size_inches(30,7)
            spec = gridspec.GridSpec(ncols=3, nrows=1,
                            width_ratios=[1.3,1,1])
            fig_id = [1,2]
            start_id = 4
            subplot = [1,3]

        tt = 0
        for i in range(start_id, 6):
            tt = tt + 1
            # ax = fig.add_subplot(subplot[0], subplot[1], fig_id[tt-1])
            ax = fig.add_subplot(spec[fig_id[tt-1]])
            table_vals = []
            if i < 5:
                data = st_data_full_ovl['p2p'][:,i-1]
                for j in range(6):
                        table_vals.append(["{:.4f}".format(st_data_full_ovl['statistical_p2p'][j,i-1]), "{:.4f}".format(st_data_sub_ovl['statistical_p2p'][j,i-1])])     
            else: 
                data = st_data_full_ovl['p2cyl_del'] 
                for j in range(6):   
                        table_vals.append(["{:.4f}".format(st_data_full_ovl['statistical_p2cyl'][j]), "{:.4f}".format(st_data_sub_ovl['statistical_p2cyl'][j])])
            
            # data_plot = data[(data > (data.mean()-4*data.std())) & ((data.mean()+4*data.std()) < data) ]
            data_plot = data
            # use the percentage of distance to determine the bins
            if len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= nbin])/len(st_data_full_ovl['p2p'])*100 < 50:
                binwidth = (data_plot.max()-data_plot.min())/100   
            else:
                binwidth = (data_plot.max()-data_plot.min())/500  
            ax.hist(data_plot,bins=np.arange(data_plot.min(), data_plot.max() + binwidth, binwidth),edgecolor='w',color = colors[i-1])
            ax.title.set_text(title[i-1])  
            min_ylim, max_ylim = plt.ylim() 
            
            # plot 1, 2, and 3 sigma lines
            xtick_data_left  = []
            xtick_data_right = []
            if i == 4:
                for k in range(1,4):
                    xtick_data_left.append(data.mean()-k*data.std()) 
                    xtick_data_right.append(data.mean()+k*data.std()) 

                    plt.plot([data.mean()+k*data.std(),data.mean()+k*data.std()],[0, max_ylim*0.25],color='m', linestyle='dashed', linewidth=2)     
                    plt.text((data.mean()+k*data.std()), max_ylim*0.255, f'+{k}' + r'$\sigma$')   
                plt.xlim(-data.mean(), data.mean() + 3.3*data.std())   
            else:
                for k in range(1,4):
                    xtick_data_left.append(data.mean()-k*data.std()) 
                    xtick_data_right.append(data.mean()+k*data.std()) 

                    plt.plot([data.mean()+k*data.std(),data.mean()+k*data.std()],[0, max_ylim*0.25],color='m', linestyle='dashed', linewidth=2)
                    plt.plot([data.mean()-k*data.std(),data.mean()-k*data.std()],[0, max_ylim*0.25],color='m', linestyle='dashed', linewidth=2)     
                    plt.text((data.mean() - k*data.std()), max_ylim*0.255, f'-{k}' + r'$\sigma$')
                    plt.text((data.mean() + k*data.std())*0.9, max_ylim*0.255, f'+{k}' + r'$\sigma$')   
                plt.xlim(data.mean() - 3.3*data.std(), data.mean() + 3.3*data.std()) 

            plt.xlabel(f'{xlabel[i-1]}')
            plt.ylabel(f'{ylabel[i-1]}')     

            # Xtick data
            xtick_data  = []
            if i != 4:
                for m in range(1,4):
                    xtick_data.append(data.mean()-m*data.std()) 
            xtick_data.append(data.mean())
            for m in range(1,4):
                xtick_data.append(data.mean()+m*data.std())     
            label = []
            for l in range(len(xtick_data)):
                label.append(str("{:.4f}".format(xtick_data[l])))
            plt.xticks(xtick_data,label)

            # plot table data on the figure
            if i == 5:
                col_labels[2] = 'mean abs'           
            the_table = plt.table(cellText=table_vals,             
                        rowLabels=col_labels,
                        colLabels=row_labels,
                        bbox = [0.72,0.6,0.2,0.4],                  
                        cellLoc='center',
                        edges = 'horizontal',
                        rowLoc='right',
                        )                  
            the_table.set_fontsize(18)

            # set the fomrat of y axis 
            current_values = ax.get_yticks()
            ax.set_yticks(current_values)
            # plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])   
            ax.set_yticklabels(['{:.0f}'.format(x) for x in current_values])   
            
        # show summary evalation data 
        # ax = fig.add_subplot(subplot[0], subplot[1], 1)   
        ax = fig.add_subplot(spec[0])  

        row_labels_=['full ovl','sub ovl*']
        np_full_ovl = len(st_data_full_ovl['p2p'])
        np_sub_ovl  = len(st_data_sub_ovl['p2p'])
        table_vals_=[[np_full_ovl, np_sub_ovl],
                    [str(round(len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= number[0]])/np_full_ovl*100,1))  + '%', str(round(len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < number[0]])/np_sub_ovl*100,1))  + '%'],
                    [str(round(len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= number[1]])/np_full_ovl*100,1))  + '%', str(round(len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < number[1]])/np_sub_ovl*100,1))  + '%'],
                    [str(round(len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= number[2]])/np_full_ovl*100,1))  + '%', str(round(len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < number[2]])/np_sub_ovl*100,1))  + '%'],
                    [str(round(len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= number[3]])/np_full_ovl*100,1))  + '%', str(round(len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < number[3]])/np_sub_ovl*100,1))  + '%'],
                    [str(round(len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= number[4]])/np_full_ovl*100,1))  + '%', str(round(len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < number[4]])/np_sub_ovl*100,1))  + '%']
                ]

        the_table_ = plt.table(cellText=table_vals_,             
                    rowLabels=col_labels_,
                    colLabels=row_labels_,
                    bbox = [0.55,0.35,0.4,0.6],                  
                    cellLoc='center',
                    edges = 'horizontal',
                    loc   = 'center'
                    # rowLoc='right',
                    )                  
        the_table_.set_fontsize(12)    
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

        y_axis = ax.axes.get_yaxis()
        y_axis.set_visible(False)
        ax.title.set_text('=-------------------> EVALUATION SUMMARY <-------------------=') 
        plt.axis('off')
        min_xlim, max_xlim = plt.xlim()
        # plt.text(min_xlim + abs(min_xlim)*0.2, max_xlim*0.3, footnote) 
        plt.text(0.1, max_xlim*0.3, footnote)   

        fig.suptitle(tt_name, fontsize=20)
        # fig.show()
        # plt.setp(tt_name, color='r') 

        # collect percentage of sub ovl data <= 0.5 [cm] or ~0.1 ft
        small_ovl_percentage = [len(st_data_full_ovl['p2p'][st_data_full_ovl['p2p'][:,3] <= s_ovl])/np_full_ovl*100, len(st_data_sub_ovl['p2p'][st_data_sub_ovl['p2p'][:,3] < s_ovl])/np_sub_ovl*100]
    
    else:
        fig = plt.figure()        
        fig.suptitle('-> There is no ovl data from selected cross-sections between: ' + tt_name + '<-', fontsize=20)
        min_xlim, max_xlim = plt.xlim()
        plt.text(0.1, max_xlim*0.3, ' ')   
        fig = plt.figure()
        fig.suptitle(tt_name, fontsize=20)
        small_ovl_percentage = []

    return small_ovl_percentage

def plot_thin_slice(df,title,width=1200,height=900,unit='m'):
    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2
    fig = px.scatter(df, x='x', y='y', color='Scan label', range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y],
                     labels={
                     "x": f"x [{unit}]",
                     "y": f"y [{unit}]"})
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color='station_name', range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y])
    fig.update_layout(
        width = width,
        height = height,
        title = title,
        # xaxis_range=[-13, 13] #xaxis_range=[df.x.min()-0.5, df.x.max()+0.5]
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        tickfont_size=13,
    )

    fig.update_xaxes(
        tickfont_size=13,
    )

    # fig['layout']['xaxis'].update(range=[-5,5], autorange=False)    
    fig['layout']['yaxis'].update(autorange = True)
    fig.update_traces(marker=dict(size=5))
    return fig

def extract_station_ids(file_names, sorted=True):
    # extract station id numbers from a list of station names
    labels = [re.findall('[0-9]+', fn) for fn in file_names]
    df = pd.DataFrame(labels)
    for i in df.columns:
        if df[i].nunique() > 1:
            names = df[i].values
            ids = names.astype(np.int8)
            if sorted:
                ind = np.argsort(ids)
                names = names[ind]
                ids = ids[ind]
                file_names = np.array(file_names)[ind]
                return names, ids, file_names
    return np.array(file_names), np.arange(len(file_names)), file_names
    

def evaluation_and_visualization(data_dir,full_list_registration_files=None,couple_of_registration_files=None,write_ovl_data=[False,False],pack_num=4,dist_criterion=0.04,approach='full',nslices=5,plot=True,unit='m',run_parallel=True,scan_type='External'):

    write_full_ovl_data=write_ovl_data[0]
    write_sub_ovl_data =write_ovl_data[1]

    write_dir = os.path.join(data_dir, 'evaluation')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir) 

    #=====================================
    # =    Evaluation of all stations    =   
    # ====================================    
    if full_list_registration_files is not None:
        write_dir = os.path.join(write_dir, 'full_evaluation', f'{unit}')
        if not os.path.exists(write_dir):
            os.makedirs(write_dir) 

        # run_parallel = False
        if run_parallel:
        # define argument inputs for each workers
            args=[]
            for j in range(0,len(full_list_registration_files)-1):    
                pcd_trg_filename = Path(full_list_registration_files[j]).name
                pcd_src_filename = Path(full_list_registration_files[j+1]).name   
                args.append([data_dir,pcd_trg_filename,pcd_src_filename,None,approach,nslices,dist_criterion,write_full_ovl_data,write_sub_ovl_data,scan_type])
                
            # run parallel
            if pack_num > len(args):
                pack_num = len(args)
            with Pool(pack_num) as p:
                results = p.starmap(
                    registration_evaluation,
                    args
                )
        else:
            results=[]
            for j in range(0,len(full_list_registration_files)-1):    
                pcd_trg_filename = Path(full_list_registration_files[j]).name
                pcd_src_filename = Path(full_list_registration_files[j+1]).name   
                results.append(registration_evaluation(data_dir,pcd_trg_filename,pcd_src_filename,None,approach,nslices,dist_criterion,write_full_ovl_data,write_sub_ovl_data,scan_type))

        # loop to colleact data and plot metric    
        st_data_full_ovl = {
            'p2p':np.array([0,0,0,0]).reshape([1,4]),
            'p2cyl_del':[],
            'statistical_p2p':[],
            'statistical_p2cyl':[]
        }
        st_data_sub_ovl = {
            'p2p':np.array([0,0,0,0]).reshape([1,4]),
            'p2cyl_del':[],
            'statistical_p2p':[],
            'statistical_p2cyl':[]
        }

        # only get station number
        file_names = [Path(full_list_registration_files[i]).stem for i in range(len(full_list_registration_files))]
        file_names = extract_station_ids(file_names, sorted=True)
        file_names = file_names[0].tolist()

        couple_stations_full_ovl = []
        couple_stations_sub_ovl = []
        collect_cross_sec_data = []
        small_ovl_percentage_all = []
        tt_name = []
        for i in range(len(results)):  
            # tt_name.append(f'{Path(full_list_registration_files[i]).stem} -[&]- {Path(full_list_registration_files[i+1]).stem}')
            tt_name.append(f'Scan {file_names[i]} -[&]- Scan {file_names[i+1]}')
            print(tt_name[i])
            # collect data after runing parallel 
            if len(results[i][0]):
                # --- full overlapping data --- 
                st_data_full_ovl['p2p'] = np.r_[st_data_full_ovl['p2p'],results[i][0]['p2p']]
                st_data_full_ovl['p2cyl_del'] = np.r_[st_data_full_ovl['p2cyl_del'],results[i][0]['p2cyl_del']]
                # --- sub overlapping data ---
                st_data_sub_ovl['p2p']  = np.r_[st_data_sub_ovl['p2p'],results[i][1]['p2p']]
                st_data_sub_ovl['p2cyl_del']  = np.r_[st_data_sub_ovl['p2cyl_del'],results[i][1]['p2cyl_del']]

            # collect cross section data
            collect_cross_sec_data.append(results[i][-2])   
            
            # store the data for every couple of stations
            couple_stations_full_ovl.append(results[i][0])
            couple_stations_sub_ovl.append(results[i][1])

            # plot every couple of stations
            if plot:
                # small_ovl_percentage = plot_evaluation_data(results[i][0],results[i][1],tt_name[i],dist_criterion)
                small_ovl_percentage = plot_evaluation_data(results[i][0],results[i][1],'',dist_criterion,unit=unit)
                plt.ioff()
                plt.savefig(os.path.join(write_dir, tt_name[i]),bbox_inches="tight" )
                small_ovl_percentage_all.append(small_ovl_percentage)

        # calculate statistical data
        if len(st_data_full_ovl['p2cyl_del']):
            # --- full overlapping data --- 
            st_data_full_ovl['statistical_p2p'] = np.r_[ np.min(st_data_full_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.max(st_data_full_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.mean(st_data_full_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.std(st_data_full_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.median(st_data_full_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.linalg.norm(st_data_full_ovl['p2p'],axis=0).reshape([1,4])/np.sqrt(len(st_data_full_ovl['p2p']))]
            st_data_full_ovl['statistical_p2cyl'] = [st_data_full_ovl['p2cyl_del'].min(),
                                                    st_data_full_ovl['p2cyl_del'].max(),
                                                    st_data_full_ovl['p2cyl_del'].mean(),
                                                    st_data_full_ovl['p2cyl_del'].std(),
                                                    np.median(st_data_full_ovl['p2cyl_del']),
                                                    np.linalg.norm(st_data_full_ovl['p2cyl_del'])/np.sqrt(len(st_data_full_ovl['p2cyl_del']))]
            # --- sub overlapping data ---
            st_data_sub_ovl['statistical_p2p'] = np.r_[ np.min(st_data_sub_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.max(st_data_sub_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.mean(st_data_sub_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.std(st_data_sub_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.median(st_data_sub_ovl['p2p'],axis=0).reshape([1,4]),
                                                        np.linalg.norm(st_data_sub_ovl['p2p'],axis=0).reshape([1,4])/np.sqrt(len(st_data_sub_ovl['p2p']))]      
            st_data_sub_ovl['statistical_p2cyl'] = [st_data_sub_ovl['p2cyl_del'].min(),
                                                    st_data_sub_ovl['p2cyl_del'].max(),
                                                    st_data_sub_ovl['p2cyl_del'].mean(),
                                                    st_data_sub_ovl['p2cyl_del'].std(),
                                                    np.median(st_data_sub_ovl['p2cyl_del']),
                                                    np.linalg.norm(st_data_sub_ovl['p2cyl_del'])/np.sqrt(len(st_data_sub_ovl['p2cyl_del']))] 
        if plot:
            # plot statistical data for all station
            # plot_evaluation_data(st_data_full_ovl,st_data_sub_ovl,'===========> Full registrations <=====',dist_criterion)
            plot_evaluation_data(st_data_full_ovl,st_data_sub_ovl,'',dist_criterion,unit=unit)
            
            id_cp = np.argsort(np.asarray(small_ovl_percentage_all)[:,0])
            min_xlim, max_xlim = plt.xlim()
            names = ['1st','2nd','3rd']
            nums  = [0.25,0.20,0.15]
            for i in range(min(len(id_cp),3)):     
                # plt.text(min_xlim + abs(min_xlim)*0.2, max_xlim*0.3, footnote)             
                # plt.text(min_xlim+abs(min_xlim)*0.2, max_xlim*nums[i], f'-> The {names[i]} worst couple: {tt_name[id_cp[i]]}')
                plt.text(0.1, max_xlim*nums[i], f'-> The {names[i]} worst couple: {tt_name[id_cp[i]]}')
                # plt.text(min_xlim-abs(min_xlim)*0.02, max_xlim*0.20, f'-> The 2nd worst couple: {tt_name[id_cp[1]]}')
                # plt.text(min_xlim-abs(min_xlim)*0.02, max_xlim*0.15, f'-> The 3rd worst couple: {tt_name[id_cp[2]]}')

            plt.ioff()
            plt.savefig(os.path.join(write_dir, 'Full registrations'),bbox_inches="tight" )

        # plot cross-sections
        z_slice = results[i][-1]
        full_cross_sec_data = []
        # # nslices = len(z_slice)
        # cross_sec_name = []
        # for i in range(1,nslices): # loop through each cross-section
        #     plot_data = np.empty(shape=[0,4])    
        #     for j in range(len(collect_cross_sec_data)):
        #         keys = list(collect_cross_sec_data[j])
        #         cross_sec  = collect_cross_sec_data[j][keys[0]][i]
        #         cross_sec_ = np.c_[cross_sec, np.ones(cross_sec.shape[0])*j]
        #         plot_data  = np.r_[plot_data,cross_sec_]
        #     # get the last station at the last couple
        #     cross_sec  = collect_cross_sec_data[j][keys[1]][i]
        #     cross_sec_ = np.c_[cross_sec, np.ones(cross_sec.shape[0])*j]
        #     plot_data  = np.r_[plot_data,cross_sec_]

        #     # plot cross-section at each slice
        #     cs_df = pd.DataFrame(plot_data, columns=['x', 'y', 'z', 'tank_id'])
        #     cs_df['Station labels'] = cs_df.tank_id.apply(lambda i: Path(full_list_registration_files[int(i)]).stem)
            
        #     cross_sec_name.append(f'Cross-section at elevation = {z_slice[i]:.2f} [m].png')
        #     if plot:
        #         fig = plot_thin_slice(cs_df,cross_sec_name[i-1][:-4])
        #         fig.write_image(os.path.join(write_dir, cross_sec_name[i-1]))

        #     # store full cross-section data
        #     full_cross_sec_data.append(cs_df)
    #=====================================
    # = Evaluation of couple of stations =   
    # ====================================     
    if couple_of_registration_files is not None:
        write_dir = os.path.join(write_dir, 'couple_evaluation')
        if not os.path.exists(write_dir):
            os.mkdir(write_dir) 
        trg_filename = couple_of_registration_files[0]
        src_filename = couple_of_registration_files[1]  
        tt_name = f"Sations: {Path(trg_filename).name} -[&]- {Path(src_filename).name}"
        print(tt_name)
        st_data_full_ovl, st_data_sub_ovl, collect_cross_sec_data, z_slice = registration_evaluation(
                                        data_dir,
                                        trg_filename,
                                        src_filename,
                                        write_data=True,
                                        approach='full',
                                        nslice=nslices
                                    )
        if plot:
            fig = plot_evaluation_data(st_data_full_ovl,st_data_sub_ovl,tt_name,dist_criterion)
            plt.ioff()
            plt.savefig(os.path.join(write_dir, f'{Path(trg_filename).stem} & {Path(src_filename).stem}'),bbox_inches="tight" )   

    return st_data_full_ovl, st_data_sub_ovl, couple_stations_full_ovl, couple_stations_sub_ovl, full_cross_sec_data, z_slice 