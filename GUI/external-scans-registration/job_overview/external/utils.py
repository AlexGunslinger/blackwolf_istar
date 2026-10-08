import streamlit as st
import pandas as pd
import plotly.express as px
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from External.source.utils.read_pts import read_input_data
from common.utils import extract_station_ids
import numpy as np
import os
import pickle

@st.cache
def load_data(data_dir, type='e57'): 
    files = glob(str(Path(data_dir) / f'*.{type}'))  
    file_name = []
    file_size = []
    npoints   = []
    points    = {}
    pack_num  = 6
    
    labels, ids, files = extract_station_ids(files)
    args = []
    for i,f in enumerate(files):
        file_name.append(Path(f).stem) #
        file_size.append(f'{round(Path(f).stat().st_size/1e6,2)} MB')
        args.append([data_dir, Path(f).name, False])
            
    # run parallel
    with Pool(pack_num) as p:
        results = p.starmap(
            read_input_data,
            args
        )

    for j in range(len(results)):
        # _ = read_input_data(data_dir, Path(f).name, read_e57_raw=False)
        points[file_name[j]] = results[j][['x', 'y', 'z']].values

        # pcd = o3d.io.read_point_cloud(os.path.join(data_dir,Path(f).name))
        # points[file_name[i]] = np.asarray(pcd.points)

        len_points = len(points[file_name[j]])
        if len_points > 1:
            npoints.append(len_points)
        else: 
            npoints.append(None)

    return file_name, labels, ids, file_size, npoints, points

def write_summarised_data(data, file):
    if not os.path.exists(file.parent):
        os.mkdir(file.parent)
    with open(file, 'wb') as f:
        pickle.dump(data, f)

@st.cache
def load_summarised_data(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
@st.cache
def visualise_thin_slice(sel_station,points):

    point_plot = points[sel_station]
    # l = 5
    thickness = 0.02
    z = point_plot[:,2]
    l = (z.max() + z.min()) / 2
    ind = ((l <= point_plot[:,2]) & (point_plot[:,2] <= l + thickness))      

    df = pd.DataFrame(point_plot[ind,:], columns=['x', 'y', 'z'])

    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2
    fig = px.scatter(df, x='x', y='y', range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y])
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color='station_name', range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y])
    fig.update_layout(
        width = 800,
        height = 800,
        #title = title #"thin slice"
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=400)
    return fig