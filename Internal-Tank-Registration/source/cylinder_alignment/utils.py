import gc
import plotly.express as px

def extract_target_cylinder(cyl, write_dir):
    # corresponding station
    print(cyl.station)
    cyl.station.read_points()
    # cylinder
    print(cyl.ptc_filepath)
    cyl.read_points()
    cyl.calc_z_min()
    cyl.extend_ground()
    cyl.write_points(center=False, write_dir=write_dir, data_type='pcd')
    # clean up
    del cyl.station.ptc
    cyl.station.ptc = None
    del cyl.points
    cyl.points = None
    gc.collect()

def draw_thin_slice(df, x='x', y='y', color='tank_label', title=''):
    x_min = df.x.min()
    x_max = df.x.max()
    delta_x = (x_max - x_min) * 0.2
    y_min = df.y.min()
    y_max = df.y.max()
    delta_y = (y_max - y_min) * 0.2
    fig = px.scatter(df, x=x, y=y, color=color, range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y])
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color='station_name', range_x=[x_min-delta_x, x_max+delta_x], range_y=[y_min-delta_y, y_max+delta_y])
    fig.update_layout(
        width = 800,
        height = 800,
        title = title
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    return fig