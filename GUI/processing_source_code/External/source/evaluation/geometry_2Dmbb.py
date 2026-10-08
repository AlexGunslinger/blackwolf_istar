# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 08:08:44 2021

@author: Linh Truong-Hong
"""
import numpy as np
from ..evaluation.utils_geometry import vector_unit, orthogonal_2Dvector

def colinear(pts_xyz):
    """This function is to check if the data set have at least two colinear
    Input:
        pts                 : [x,y,z]
    Output:
        False               : not colinear
        True                : coliner
        
    Demo:
        pts_xyz = cluster_subcell_pts_xyz[:50]
    """
    # Call libs
    import itertools
    
    # Constant
    min_pts = 3
    # Random select 3 points
    local_ids = np.arange(pts_xyz.shape[0])
    # Get local indices of three points
    tri_local_ids = list(itertools.combinations(local_ids, min_pts))
    tri_local_ids = np.asarray(tri_local_ids, dtype=np.uint16) 
    # Check co-linear
#    count_colinear = True
    colinear = True
    count = 0
    while (count < tri_local_ids.shape[0]) & (colinear):
        # Get the points
        three_points = pts_xyz[tri_local_ids[count]]     
        # Colinear check
        det = (three_points[0, 0]*(three_points[1, 1] - three_points[2, 1]) +
               three_points[1, 0]*(three_points[2, 1] - three_points[0, 1]) +
               three_points[2, 0]*(three_points[0, 1] - three_points[1, 1]))
        if det != 0:
            colinear = False
        # Update for next iteration
        count = count + 1
    # Determine colinear
    return colinear
#    if colinear:
#        return False
#    else:
#        return True
    
def min_2Dbounding_box(pts_xyz, option = 'area'):
    """
    This function is to estimate the 2D minimum bounding box of the data
    Parameters:
    ------------------------
        pts_xyz                                 : [Nx3], x-, y-, z-

    Return:
    ------------------------
        min_rectangle.area                  : Area
        min_rectangle.parallel_length       : Length of edge 1 - scalar
        min_rectangle.orthogonal_length     : Length of edge 2 - scalar
        min_rectangle.rectangle_center      : Center - [1x2]
        min_rectangle.rectangle_corners     : Corners [4x2] [xi,yi]
        min_rectangle.parallel_edge_vector  : a unit vector of edge 1

    Demo:
    pts_xyz = ptc_xyz[:,:2]
    pts_xyz = sect_pts_xyz
    option = 'fit_edge'
    option = 'area'
    option = 'edge'
    
    pts_xyz = region_proj_ptc_xyz
    
    """
    # Call libs
    from scipy.spatial import ConvexHull
    # Check input variable
    pts_xyz = np.asarray(pts_xyz, dtype = float)
    if pts_xyz.shape[0] < 4:
        # print('Points {:d} are less than required points {:d}'.format(pts_xyz.shape[0], 4))
        return None
    pts_xy = pts_xyz[:, :2]
    
    # remove check colinear
    pts_xy = np.unique(pts_xy, axis=0)
    if pts_xy.shape[0] < 50:
        if colinear(pts_xy):
            return None
    # Generate convexhull
    hull_pts_ids = ConvexHull(pts_xy).vertices
    hull_vert_xy = pts_xy[hull_pts_ids]
    hull_vert_xy = np.insert(hull_vert_xy, hull_vert_xy.shape[0], hull_vert_xy[0], axis = 0)

    # Find minimum rectangle
    optimize_val = 1000.
    for i in range(0, len(hull_vert_xy)-1):
        # Estimate the rectangle
        rectangle = bounding_rectangle(i, hull_vert_xy)
        # Determine the best
        if option.casefold() == 'area'.casefold():
            if rectangle['area'] < optimize_val:
                optimize_val = rectangle['area']
                min_rectangle = rectangle          
        elif option.casefold() == 'edge'.casefold():
            if (rectangle['parallel_length'] + rectangle['orthogonal_length']) < optimize_val:
                optimize_val = (rectangle['parallel_length'] + rectangle['orthogonal_length'])
                min_rectangle = rectangle
        elif option.casefold() == 'fit_edge'.casefold():
            # Measure the fit
            fit_val = fit_edges(pts_xyz, rectangle)
            if fit_val < optimize_val:
                optimize_val = fit_val
                min_rectangle = rectangle
        else:
            raise ValueError("Option {:s} is Invalid".format(option))
                
    # Get rot matrrix
    map_array =  min_2d_mbb_get_rotation_matrix(min_rectangle['parallel_edge_vector'])
    # Get the corners
    rect_corner_pts = min_2d_mbb_rect_corners(min_rectangle['rectangle_center'], min_rectangle['parallel_length'], min_rectangle['orthogonal_length'], map_array)
    min_rectangle['rectangle_corners'] = rect_corner_pts
    # Update center
    min_rectangle['rectangle_center'] = np.mean(rect_corner_pts, axis=0)
    # return
   
    return min_rectangle
#%%
def fit_edges(pts_xyz, rectangle):
    """This function is to determine how fit of the 2dmmb 
    """
    # Call libs/function
    from utils_geometry.utils_geometry import dist_points_line, residual
    # Constant
    # Get edge 
    rot_mat = min_2d_mbb_get_rotation_matrix(rectangle['parallel_edge_vector'])
    # Get the corners
    rect_corner_pts = min_2d_mbb_rect_corners(rectangle['rectangle_center'], rectangle['parallel_length'], rectangle['orthogonal_length'], rot_mat)
    rect_cent = np.mean(rect_corner_pts, axis=0)
    # Set up edges
    rect_close_verts = np.vstack((rect_corner_pts, rect_corner_pts[0]))
    rect_edges_vect = vector_unit(rect_close_verts[1:] - rect_close_verts[:-1])
    
    # Count the points within buffer
    opt_val = 0
    for edge_p, edge_vect in zip(rect_corner_pts, rect_edges_vect):
        # Compute distance points to an edge
        dist_p_edge = dist_points_line(pts_xyz[:, :2], edge_p, edge_vect)
        
        # Compute distance center to an edge
        dist_cent_edge = dist_points_line(rect_cent, edge_p, edge_vect)
        
        # Count points within buffer
        mask = np.abs(dist_p_edge) <= np.abs(dist_cent_edge)
        opt_val += residual(dist_p_edge[mask])#
    # return
    return opt_val
        
# In[]
def min_2d_mbb_rect_corners(rect_cent, parallel_length, orthogonal_length, rot_mat):
    """The function is to obtain corners of a rectangle
    """
    rect_corner_pts = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1.*i1):
            corner_pt = np.array([rect_cent[0] + i1*parallel_length,
                                  rect_cent[1] + i2*orthogonal_length])
            corner_pt = np.sum(corner_pt*rot_mat, axis=1)
            rect_corner_pts = np.append(rect_corner_pts, corner_pt)
    rect_corner_pts = rect_corner_pts.reshape(-1,2)
    return rect_corner_pts
# In[]
def min_2d_mbb_get_rotation_matrix(edge_vector):
    """This function is to get a rotation matric for 2D mbb
    edge_vector = rectangle['parallel_edge_vector']
    """
    # Get rot matrrix
    parallel_edge_angle = np.arctan2(edge_vector[1], edge_vector[0])
    orthogonal_edge_angle = parallel_edge_angle + np.pi/2.
    rot_mat = np.array([[np.cos(parallel_edge_angle), np.cos(orthogonal_edge_angle)],
                          [np.sin(parallel_edge_angle), np.sin(orthogonal_edge_angle)]])
    return rot_mat
    
#%%
def bounding_rectangle(vert_id, hull_verts):
    """
    This function is to get a rectangle where an edge is one edge of 2D polygon
    Parameters:
    ------------------------
        vert_id                             : index of vertices of a 2D polygon
        hull_verts                          : vertices of the 2D polygon - [Nx2] - [xi, yi]

    Return:
    ------------------------
        min_rectangle.area                  : Area
        min_rectangle.parallel_length       : Length of edge 1 - scalar
        min_rectangle.orthogonal_length     : Length of edge 2 - scalar
        min_rectangle.rectangle_center      : Center - [1x2]
        min_rectangle.parallel_edge_vector  : a unit vector of edge 1

    Demo:

    vert_id = 0
    hull_verts = hull_vert_xy
    """

    # Get the first edge through vert_id
    edge_vect = hull_verts[vert_id + 1] - hull_verts[vert_id]
    edge_unit_vect = vector_unit(edge_vect)
    edge_ortho_vect = orthogonal_2Dvector(edge_unit_vect)

    # Call distance from the hull vertices to the vector
    dist_verts_unit_vect = [np.dot(edge_unit_vect, pt) for pt in hull_verts]
    dist_verts_ortho_vect = [np.dot(edge_ortho_vect, pt) for pt in hull_verts]

    # Determine the vertices on the edges of the min rectangle
    vert_unit_vect = np.min(dist_verts_unit_vect)
    vert_ortho_vect = np.min(dist_verts_ortho_vect)
    len_edge_unit_vect = np.max(dist_verts_unit_vect) - vert_unit_vect
    len_edge_ortho_vect = np.max(dist_verts_ortho_vect) - vert_ortho_vect

    rect = {"area": len_edge_unit_vect*len_edge_ortho_vect,
            "parallel_length": len_edge_unit_vect,
            "orthogonal_length": len_edge_ortho_vect,
            "rectangle_center": np.array([vert_unit_vect + len_edge_unit_vect/2.0, vert_ortho_vect + len_edge_ortho_vect/2.0]),
            "parallel_edge_vector": edge_unit_vect}

    return rect
