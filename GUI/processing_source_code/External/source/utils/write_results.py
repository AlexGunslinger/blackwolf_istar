# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:55:40 2021

@author: LinhTruongHong
"""
import os
import numpy as np
from .write_output import write_array2txt

def write_bin_result(cellgrids, extracted_bins, path):
    """This function is to write the results of the bin extraction
    Input:
        cellgrid                    : Cell grid structure
        extracted_bins              : Results of extracted bins. For each bin it contains:
            ..["pts_ids"]           : Indices of the points of the bin
            ..["pts_seg_ids"]       : Result of segmentation based on the points of the bin
            ..["surface"]           : Results points of each surface of the bin. See refine_boundary.py for output
        path                        : path
    Output:
        all_bin_pts.txt             : Point clouds of bins
        all_bin_surfaces_pts.txt    : Point clouds of bins associated with surface ids
        
    Demo:
    """
    
    #%% Write results
    bins_pts_surf_ids = []
#    bin_pts_xyz = []
    for count, bin_id in enumerate(list(extracted_bins.keys())):
        # Get the candidate points of the bin
        single_bin = extracted_bins[bin_id]
        
        # Get the points of the vertical surfaces
        single_bin_vert_surfs_ids = np.array(list(single_bin["vert_surface"].keys()))
        single_bin_vert_surfs_pts = []
        for vert_surf_id in single_bin_vert_surfs_ids:
            vert_surf_pts_ids = single_bin["vert_surface"][vert_surf_id]["surface_ptc_ids"]
            vert_surf_pts_ids = np.vstack((vert_surf_pts_ids, np.full(vert_surf_pts_ids.shape[0], vert_surf_id, dtype=np.uint32))).T  
            single_bin_vert_surfs_pts = np.append(single_bin_vert_surfs_pts, vert_surf_pts_ids)
        single_bin_vert_surfs_pts = np.asarray(single_bin_vert_surfs_pts, dtype=np.uint32).reshape(-1, 2)
        
        # Get the points of the other surfaces
        if len(single_bin["other_surface"]) > 0:
            single_bin_other_surfs_ids = np.array(list(single_bin["other_surface"].keys()))
            single_bin_other_surfs_pts = []
            for other_surf_id in single_bin_other_surfs_ids:
                other_surf_pts_ids = single_bin["other_surface"][other_surf_id]["surface_ptc_ids"]
                other_surf_pts_ids = np.vstack((other_surf_pts_ids, np.full(other_surf_pts_ids.shape[0], other_surf_id, dtype=np.uint32))).T  
                single_bin_other_surfs_pts = np.append(single_bin_other_surfs_pts, other_surf_pts_ids)
            # Get pts_ids of the bin
            if single_bin_other_surfs_pts.shape[0] > 10:
                single_bin_other_surfs_pts = np.asarray(single_bin_other_surfs_pts, dtype=np.uint32).reshape(-1, 2)
                single_bin_pts_surfs_ids = np.vstack((single_bin_vert_surfs_pts, single_bin_other_surfs_pts))
        else:
            single_bin_pts_surfs_ids = single_bin_vert_surfs_pts
        # Assigned bin number
        single_bin_pts_surfs_ids = np.hstack((single_bin_pts_surfs_ids, np.full((single_bin_pts_surfs_ids.shape[0], 1), count + 1, dtype=np.uint32)))  
         
        # Update detect_bin_pts
        bins_pts_surf_ids = np.append(bins_pts_surf_ids, single_bin_pts_surfs_ids)
        
#        # Get pts-xyz of the bin
#        bin_pts_xyz = np.append(bin_pts_xyz, single_bin["ptc_xyz"])
#        bin_pts_xyz = np.hstack((bin_pts_xyz, np.full((bin_pts_xyz.shape[0], 1), count + 1, dtype=np.uint32)))  
        

    # Point surface indices + bin ids
    bins_pts_surf_ids = np.asarray(bins_pts_surf_ids, dtype=np.uint32).reshape(-1,3)
    # Get points + bin ids
    bin_pts = np.hstack((cellgrids.points[bins_pts_surf_ids[:, 0]], bins_pts_surf_ids[:, 1:])) 
    
    #
    file_name = '_bin_surfaces_pts.txt'
    if path is None:
        from write_output import get_path_file
        path = get_path_file(None, 'temp_result')
    write_array2txt(path + file_name, bin_pts) 

    