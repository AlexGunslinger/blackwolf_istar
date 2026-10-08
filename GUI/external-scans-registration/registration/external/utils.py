import streamlit as st
import pandas as pd
import plotly.express as px
from glob import glob
from pathlib import Path
import numpy as np
from External.source.cylinder_alignment.bundle_station import BundleStation

@st.cache
def get_extracted_cylinder_slices(station):
    points = np.empty([0,3]) # x,y,id
    for c in station.refined_cylinders:
        c.read_points()
        z = c.points[:,2]
        z_cut = (z.max() + z.min()) / 2
        thickness = 0.01
        p = c.points[
            (z >= z_cut) & (z <= (z_cut + thickness)), :2
        ]
        points = np.r_[
            points,
            np.c_[
                p, c.id * np.ones([p.shape[0], 1])
            ]
        ]
    return points

@st.cache
def vis_extracted_cylinders(data_dir, station_name, add_title=False):
    bundle = BundleStation(data_dir=data_dir)
    bundle.read_extracted_cylinder_data(file_types=['e57', 'pcd'], print_info=False)
    s = bundle.get_station(station_name)
    points = get_extracted_cylinder_slices(s)
    df = pd.DataFrame(points, columns=['x', 'y', 'local_tank_id'])
    df['Tank (local) id'] = df.local_tank_id.apply(lambda sid: str(int(sid)))

    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2
    if add_title:
        fig = px.scatter(
            df, x='x', y='y', 
            range_x=[x_min-delta_x, x_max+delta_x], 
            range_y=[y_min-delta_y, y_max+delta_y],
            color='Tank (local) id',
            title=station_name
        )
    else:
        fig = px.scatter(
            df, x='x', y='y', 
            range_x=[x_min-delta_x, x_max+delta_x], 
            range_y=[y_min-delta_y, y_max+delta_y],
            color='Tank (local) id',
            # title=station_name
        )
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

@st.cache
def vis_rough_alignment_output(data_dir, output, station_labels, scan_station_names=True):
    rough_align_dir = data_dir / 'target_tank' / 'global' / 'thin_slice'
    # station name
    station_names = []
    if scan_station_names:
        pcd_files = glob(str(rough_align_dir / '*.pcd'))
        for fn in pcd_files:
            if 'cluster' not in fn:
                station_names.append(Path(fn).stem)

    if output == 'all cylinders':
        fig_title = 'Rough alignment of all cylinders from all stations'
        file_path = rough_align_dir / 'station_cylinders_phase_1.txt'
    else:
        fig_title = 'Rough alignment of the target cylinder from all stations'
        file_path = rough_align_dir / 'target_tank_phase_1.txt'
    
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['x', 'y', 'z', 'station_id'])
    df['Scan label'] = df.station_id.apply(lambda sid: station_labels[station_names[int(sid)]])
    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2

    fig = px.scatter(
        df, x='x', y='y', 
        range_x=[x_min-delta_x, x_max+delta_x], 
        range_y=[y_min-delta_y, y_max+delta_y],
        color='Scan label',
        # title=fig_title
    )
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

@st.cache
def vis_fine_alignment_output(data_dir, station_labels, scan_station_names=True):
    fine_align_dir = data_dir / 'target_tank' / 'global' / 'ground'
    # station name
    station_names = []
    if scan_station_names:
        pcd_files = glob(str(fine_align_dir / '*.pcd'))
        for fn in pcd_files:
            # extract station names from file names (format: [station_name] cluster[])
            # remove the cluter parts
            tmp = Path(fn).stem
            station_names.append(tmp.split(' cluster')[0])

    file_path = fine_align_dir / 'icp_slice_down.txt'
    
    df = pd.read_csv(file_path, delimiter=' ', header=None, names=['x', 'y', 'z', 'station_id'])
    df['Scan label'] = df.station_id.apply(lambda sid: station_labels[station_names[int(sid)]])
    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2

    fig = px.scatter(
        df, x='x', y='y', 
        range_x=[x_min-delta_x, x_max+delta_x], 
        range_y=[y_min-delta_y, y_max+delta_y],
        color='Scan label',
        # title=fig_title
    )
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