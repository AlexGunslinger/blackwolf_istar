# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:05:14 2021

@author: ltruonghong
"""
import copy
import numpy as np
from utils.write_output import write_array2txt
def shell_wall_extraction(voxelgrid, cyl_threshold, constant):
    """
    Input:
        voxelgrid               :
        cyl_threshold           :
        constant                :
        
    Output:
        shell_wall_pts_ids      : [N x 1] - indicies of points
        shell_wall_pts_xyz      : [N x 3] - x, y,z
        shell_voxels            : dict stored voxels of the cylnder
        cyl_paras               : [1 x 7] [xc, yc, zc, rc, tx, ty, tz] - cylinder parameters
    
    
    
    
    cyl_threshold = threshold["cylinder_estimation"]
    """
    from utils.utils_geometry import cart2pol, cosin_vectors, rot_vectors, proj_3Dpoints_3Dline, vector_unit
    from utils.utils_KDE import peak_shape_range
    
    # Cal features of voxels
    voxel_linear_ids, voxels_cent, voxels_norm = cal_voxel_features(voxelgrid, cyl_threshold)
    
    # Filter voxels representing a vertical surface
    vert_voxels_local_ids, nonvert_voxels_local_ids = filter_vert_voxels(voxels_norm, cyl_threshold, constant)
    vert_voxel_linear_ids, vert_voxels_cent, vert_voxels_norm = voxel_linear_ids[vert_voxels_local_ids], voxels_cent[vert_voxels_local_ids], voxels_norm[vert_voxels_local_ids]
    
    # Extract shell wall
    cyl_paras = est_cylinder_parameters(vert_voxel_linear_ids, vert_voxels_cent, vert_voxels_norm, cyl_threshold, constant)
    cyl_cent, cyl_radius, cyl_direct = cyl_paras[:3], cyl_paras[3], cyl_paras[4:7]
    
    # fine extraction
    vert_voxels_proj_cent = proj_3Dpoints_3Dline(vert_voxels_cent, np.append(cyl_cent, cyl_direct))
    vert_voxels_cyl_dir = vector_unit(vert_voxels_cent - vert_voxels_proj_cent)
    cosin_norm = cosin_vectors(vert_voxels_norm, vert_voxels_cyl_dir)
    mask = cyl_threshold["cyl_norm"] <= np.abs(cosin_norm)
    angle_filter_vert_voxel_linear_ids, angle_filter_vert_voxels_cent = vert_voxel_linear_ids[mask], vert_voxels_cent[mask]
    
    # Filter based on radius
    cyl_rot_mat = rot_vectors(constant['oz'], cyl_direct)
    oz_vert_voxels_cent = np.matmul(angle_filter_vert_voxels_cent, cyl_rot_mat)
    oz_vert_voxels_cent_rho, __ = cart2pol(oz_vert_voxels_cent[:, 0] - cyl_cent[0], oz_vert_voxels_cent[:, 1] - cyl_cent[1])
    # peak_shape = peak_shape_range(oz_vert_voxels_cent_rho, bandwidth = cyl_cyl_threshold["cyl_max_dist"], num_locs = 200)
    # peak_voxel_local_ids, largest_peak_range = get_largest_peak(oz_vert_voxels_cent_rho, peak_shape[:, 1:])    
    # radius_filter_vert_voxel_linear_ids = angle_filter_vert_voxel_linear_ids[peak_voxel_local_ids]
    
    radius_filter_vert_voxel_linear_ids, range_rho = remove_floating_cluster_points(oz_vert_voxels_cent_rho, angle_filter_vert_voxel_linear_ids, cluster_gap=0.01, cluster_min_pts=20)
    
    # Filter the points of the shell based on voxels
    shell_voxels_pts_xyz, shell_voxels_pts_ids, shell_voxels = point_voxel_filter(voxelgrid, radius_filter_vert_voxel_linear_ids, cyl_cent, cyl_radius, cyl_rot_mat, cyl_threshold, constant)
    
    # Get the point from attached voxels
    shell_add_voxel_pts_xyz, shell_add_voxel_pts_ids, shell_add_voxels = add_voxel_points(voxelgrid, voxel_linear_ids, shell_voxels, cyl_cent, cyl_radius, cyl_rot_mat, cyl_threshold, constant)
   
    # Combinate the final 
    shell_wall_add_voxel_ids = np.array(list(shell_add_voxels.keys()), dtype=np.uint32)
    for add_voxel_linear_id in shell_wall_add_voxel_ids:
        # get points
        add_voxel_pts_ids = shell_add_voxels[add_voxel_linear_id]
        shell_voxels[add_voxel_linear_id] = add_voxel_pts_ids
    shell_wall_pts_ids = np.unique(np.append(shell_voxels_pts_ids, shell_add_voxel_pts_ids))
    shell_wall_pts_xyz = voxelgrid.points[shell_wall_pts_ids]
    
    return shell_wall_pts_ids, shell_wall_pts_xyz, shell_voxels, cyl_paras
# In[]
def add_voxel_points(voxelgrid, voxel_linear_ids, shell_voxels, cyl_cent, cyl_radius, cyl_rot_mat, cyl_threshold, constant):
    """This function is to filter to points of the shell wall
    
    candidate_voxel_linear_ids = voxel_linear_ids
    shell_voxel_linear_ids = rho_filter_vert_voxel_linear_ids
    cyl_rot_mat = rot_mat_oz
    
    """
    #% call functions/libs
    from itertools import chain
    from utils.utils_geometry import cart2pol
    
    #% preallocation
    searching_method = {'range_search': 0.01}
    shell_add_voxels = dict()
    
    # Filtering
    shell_wall_voxel_ids = np.array(list(shell_voxels.keys()), dtype=np.uint32)
    nonshell_wall_voxel_ids = np.setdiff1d(voxel_linear_ids, shell_wall_voxel_ids)
    for nonshell_wall_voxel_id in nonshell_wall_voxel_ids:
        # print(nonshell_wall_voxel_id)
        # Get the neighbour voxels
        __, neighbour_shell_wall_voxel_linear_ids, __ = voxelgrid.searching_adjoin_voxels(nonshell_wall_voxel_id, shell_wall_voxel_ids, index_type="linear", searching_opt='xyz', voxel_prop="full")
         
        # Determine if the need to search add points of the shell wall
        if (neighbour_shell_wall_voxel_linear_ids.shape[0] > 0):
            # extract the points of the shell within those voxels
            neighbour_shell_wall_voxel_pts_ids = list(chain.from_iterable(shell_voxels[ind] for ind in neighbour_shell_wall_voxel_linear_ids))
            neighbour_shell_wall_voxel_pts_ids = np.array(neighbour_shell_wall_voxel_pts_ids, dtype=np.uint32)
            neighbour_shell_wall_voxel_pts_xyz = voxelgrid.points[neighbour_shell_wall_voxel_pts_ids]    
            # Convert to polar for the shell points
            neighbour_shell_wall_voxel_pts_xyz = np.matmul(neighbour_shell_wall_voxel_pts_xyz, cyl_rot_mat)
            neighbour_shell_wall_voxel_pts_rho, __ = cart2pol(neighbour_shell_wall_voxel_pts_xyz[:, 0] - cyl_cent[0], neighbour_shell_wall_voxel_pts_xyz[:, 1] - cyl_cent[1])
            neighbour_shell_wall_voxel_pts_delta_rho = neighbour_shell_wall_voxel_pts_rho - cyl_radius
            
            __, range_rho = remove_floating_cluster_points(neighbour_shell_wall_voxel_pts_delta_rho, neighbour_shell_wall_voxel_pts_ids)
        

            # Check each voxels attached to the shell
            # get the points within the voxel
            nonshell_wall_voxel_pts_ids = voxelgrid.ptc_ids[nonshell_wall_voxel_id]
            nonshell_wall_voxel_pts_xyz = voxelgrid.points[nonshell_wall_voxel_pts_ids]
            
            if nonshell_wall_voxel_pts_xyz.shape[0] >= cyl_threshold["voxel_min_ptc"]:
                # Convert to polar for the shell points
                nonshell_wall_voxel_pts_xyz = np.matmul(nonshell_wall_voxel_pts_xyz, cyl_rot_mat)
                nonshell_wall_voxel_pts_rho, __ = cart2pol(nonshell_wall_voxel_pts_xyz[:, 0] - cyl_cent[0], nonshell_wall_voxel_pts_xyz[:, 1] - cyl_cent[1])
                nonshell_wall_voxel_pts_delta_rho = nonshell_wall_voxel_pts_rho - cyl_radius
                # Filtering
                mask = (range_rho[0] <= nonshell_wall_voxel_pts_delta_rho) & (nonshell_wall_voxel_pts_delta_rho <= range_rho[1])
                shell_wall_add_voxel_pts_ids = nonshell_wall_voxel_pts_ids[mask]
                shell_wall_add_voxel_pts_xyz = nonshell_wall_voxel_pts_xyz[mask]
                shell_wall_voxel_pts_delta_rho = nonshell_wall_voxel_pts_delta_rho[mask]
            
                # Remove the horizontal points
                if shell_wall_add_voxel_pts_ids.shape[0] >= cyl_threshold["voxel_min_ptc"]:
                    local_ids = remove_horizontal_points(shell_wall_add_voxel_pts_xyz, searching_method, cyl_threshold, constant)
                    shell_wall_add_voxel_pts_ids = shell_wall_add_voxel_pts_ids[local_ids]
#                        neighbour_add_voxel_add_pts_xyz = neighbour_add_voxel_add_pts_xyz[local_ids]
                    shell_wall_voxel_pts_delta_rho = shell_wall_voxel_pts_delta_rho[local_ids]
                    # Self filter 
                    if shell_wall_add_voxel_pts_ids.shape[0] >= cyl_threshold["voxel_min_ptc"]:
                        shell_wall_add_voxel_pts_ids, __ = remove_floating_cluster_points(shell_wall_voxel_pts_delta_rho, shell_wall_add_voxel_pts_ids, cluster_gap=0.005, cluster_min_pts=20)
        
                        # Update the shell_voxel
                        # print(shell_wall_add_voxel_pts_ids.shape[0])
                        if shell_wall_add_voxel_pts_ids.shape[0] >= cyl_threshold["voxel_min_ptc"]:
                            shell_add_voxels[nonshell_wall_voxel_id] = shell_wall_add_voxel_pts_ids
            
    # Get the points of the shell
    shell_add_voxel_ids = list(shell_add_voxels.keys())
    shell_add_voxel_add_pts_ids = list(chain.from_iterable(shell_add_voxels[ind] for ind in shell_add_voxel_ids))
    shell_add_voxel_add_pts_ids = np.array(shell_add_voxel_add_pts_ids, dtype=np.uint32)
    shell_add_voxel_add_pts_xyz = voxelgrid.points[shell_add_voxel_add_pts_ids]    
#     
    # return
    return shell_add_voxel_add_pts_xyz, shell_add_voxel_add_pts_ids, shell_add_voxels    


# In[]
def remove_horizontal_points(pts_xyz, searching_method, cyl_threshold, constant):
    """This function is to remove horizontal points
    
    pts_xyz = neighbour_add_voxel_pts_xyz
    
    
    """
    from sklearn.neighbors import KDTree
    from utils.utils_geometry import PCA_NormalVector, cosin_vectors
    # Searching neighbour points
    leaf_size = 50
    tree = KDTree(pts_xyz, leaf_size = leaf_size)
    if 'range_search' in searching_method.keys():
        # Range search
        searching_para = searching_method['range_search'] if searching_method['range_search'] > 0 else 0.1
        kdtree_neighbour_pts_ids = tree.query_radius(pts_xyz[:,:3], r = searching_para)
    elif 'kNN' in searching_method.keys():
		# kNN search
        searching_para = int(searching_method['kNN']) if int(searching_method['kNN']) > 5 else 20
        if searching_para >= pts_xyz.shape[0]:
            searching_para = pts_xyz.shape[0] - 1
        __, kdtree_neighbour_pts_ids = tree.query(pts_xyz[:,:3], k = searching_para)
    else:
        raise ValueError('The searching method does not support')

    # Compute the features
    pts_ids = np.arange(pts_xyz.shape[0])
    pts_normal = np.full((pts_xyz.shape[0], 3), np.inf, dtype = np.float32)
    for count in range(pts_xyz.shape[0]):
        # Get the neighbour points
        neighbour_pts_ids = kdtree_neighbour_pts_ids[count]
        neighbour_pts_xyz = pts_xyz[neighbour_pts_ids]
        # Estimate normal
        pt_normal = PCA_NormalVector(neighbour_pts_xyz, min_ptc = 5)

        # Update features
        pts_normal[count,:3] = pt_normal
        
    # remove singularity
    mask = np.any(np.isinf(pts_normal), axis = 1)
    pts_ids = pts_ids[~mask]
    pts_normal = pts_normal[~mask]
    
    # Compute cosin
    cosin_norm = cosin_vectors(pts_normal, constant["oz"])
    mask = np.abs(cosin_norm)  <= cyl_threshold["ground_points"]
    local_ids = np.where(mask == True)[0]
    
    # return
    return local_ids
    
    
    
# In[]
def point_voxel_filter(voxelgrid, shell_voxel_linear_ids, cyl_cent, cyl_radius, cyl_rot_mat, cyl_threshold, constant):
    """This function is to filter to points of the shell wall
    
  
    shell_voxel_linear_ids = radius_filter_vert_voxel_linear_ids
     
    
    
    """
    # call functions/libs
    from itertools import chain
    from utils.utils_geometry import cart2pol
    from utils.utils_KDE import peak_shape_range
    
    #% preallocation
    shell_voxels = dict()
    # Filtering
    for shell_voxel_linear_id in shell_voxel_linear_ids:
        # print(shell_voxel_linear_id)
        
        # get the points of the currrent voxel
        shell_voxel_pts_ids = voxelgrid.ptc_ids[shell_voxel_linear_id]
        shell_voxel_pts_xyz = voxelgrid.points[shell_voxel_pts_ids]
        
        # Convert to polar for the current voxel
        shell_voxel_pts_xyz = np.matmul(shell_voxel_pts_xyz, cyl_rot_mat)
        shell_voxel_pts_rho, __ = cart2pol(shell_voxel_pts_xyz[:, 0] - cyl_cent[0], shell_voxel_pts_xyz[:, 1] - cyl_cent[1])
        shell_voxel_pts_delta_rho = shell_voxel_pts_rho - cyl_radius
        # Filter the points based KDE
        shell_voxel_filter_pts_ids, __ = remove_floating_cluster_points(shell_voxel_pts_delta_rho, shell_voxel_pts_ids)
        
#        peak_shape = peak_shape_range(shell_voxel_pts_delta_rho, bandwidth = cyl_threshold["cyl_max_dist"], num_locs = 200)
#        local_ids, largest_peak_range = get_largest_peak(shell_voxel_pts_delta_rho, peak_shape[:, 1:])    
#        shell_voxel_filter_pts_ids = shell_voxel_pts_ids[local_ids]
#        
        # Update the shell_voxel
        if shell_voxel_filter_pts_ids.shape[0] >= cyl_threshold["voxel_min_ptc"]:
            shell_voxels[shell_voxel_linear_id] = shell_voxel_filter_pts_ids
        
    # Get the points of the shell
    shell_voxel_ids = list(shell_voxels.keys())
    shell_voxels_pts_ids = list(chain.from_iterable(shell_voxels[ind] for ind in shell_voxel_ids))
    shell_voxels_pts_ids = np.array(shell_voxels_pts_ids, dtype=np.uint32)
    shell_voxesl_pts_xyz = voxelgrid.points[shell_voxels_pts_ids]    
     
    # return
    return shell_voxesl_pts_xyz, shell_voxels_pts_ids, shell_voxels


#%%write_array2txt('shell_voxel_pts_xyz.txt', shell_voxel_pts_xyz) 
def remove_floating_cluster_points(pts_val, pts_ids, cluster_gap=0.001, cluster_min_pts=20):
    """This function is to remove a group floating points within the member based on length and density
    Parameters:
    ------------------------
        voxelgrid                       ; 3D voxelgrid
        mbb                             : 3D mbb of the member
        ptc_ids                         : indices of 3D points
        threshold                      : threshold
        constant                       : constant
       
    Return:
    ------------------------
        cluster_ptc                     : remain points
        cluster_ptc_ids                 : remain point indices
        
    Demo:
    ------------------------
    ptc = member_ptc
    ptc_ids = member_ptc_ids
    pts_val, pts_ids = neighbour_shell_voxel_pts_xyz_delta_rho, neighbour_shell_voxel_pts_ids
    
    mbb, ptc_ids, threshold, constant = member_mbb, extend_member_pts_ids, threshold, constant
    pts_val, pts_ids = neighbour_add_voxel_pts_xyz_delta_rho, neighbour_add_voxel_pts_ids
            
    """
    # Constants
    # CLUSTER_MIN_PTS = 20
    # CLUSTER_GAP = 0.001
    
    # sort data in z-direction 
    mask = np.argsort(pts_val)
    pts_ids = pts_ids[mask]
    pts_val = pts_val[mask]

    # Cluster the points in z-direction
    delta = np.abs(np.diff(pts_val))
    mask = cluster_gap < delta
    local_ids = np.where(mask == True)[0]

    if local_ids.shape[0] > 0:
        # Create indices of the long member
        s_ids = np.unique(np.append([0], local_ids + 1))
        e_ids = np.unique(np.append(local_ids, [pts_val.shape[0] - 1]))
        cluster_num_pts = e_ids - s_ids

        # Searching the connect cluster
        max_id = np.argmax(cluster_num_pts) # Find the biggest cluster
        biggest_cluster_s_id, biggest_cluster_e_id = s_ids[max_id], e_ids[max_id]
        biggest_cluster_delta_rho = pts_val[biggest_cluster_s_id:biggest_cluster_e_id+1]
        if biggest_cluster_delta_rho.shape[0] < 5: # No peak
            return np.array([]), np.array([])
        # Generate histogram
        num_bin = np.ceil((np.max(biggest_cluster_delta_rho) - np.min(biggest_cluster_delta_rho))/(cluster_min_pts*cluster_gap)).astype(np.uint32)
        hist_rho = np.histogram(biggest_cluster_delta_rho, bins=num_bin, density=False)
        hist_rho_no_pts = hist_rho[0]
        hist_rho_edges = hist_rho[1] 
        
        # Get the candidate bin
        hist_bin_id = np.argmax(hist_rho_no_pts)
        resulting_hist_bin_ids = np.array([hist_bin_id])
        seeding_hist_bin_ids = resulting_hist_bin_ids
        check_hist_bin_ids = np.setdiff1d(np.arange(hist_rho_no_pts.shape[0]), seeding_hist_bin_ids)
        while (check_hist_bin_ids.shape[0] > 0) & (seeding_hist_bin_ids.shape[0] > 0):
            # Get seeding cluster
            seeding_hist_bin_id = seeding_hist_bin_ids[0]
            # Get adjacent cluster
            mask = (seeding_hist_bin_id - 1 <= check_hist_bin_ids) & (check_hist_bin_ids <= seeding_hist_bin_id + 1)
            adj_hist_bin_ids = check_hist_bin_ids[mask]
            # Check if they can connect or not
            if adj_hist_bin_ids.shape[0] > 0:
                # Check the number of the points
                adj_hist_bin_np_pts = hist_rho_no_pts[adj_hist_bin_ids]
                mask = cluster_min_pts <= adj_hist_bin_np_pts
                adj_hist_bin_ids = adj_hist_bin_ids[mask]
                # Update the cluster
                if adj_hist_bin_ids.shape[0] > 0:
                    # Update seeding
                    seeding_hist_bin_ids = np.append(seeding_hist_bin_ids, adj_hist_bin_ids)
                    # Update check_cluster_ids
                    check_hist_bin_ids = np.setdiff1d(check_hist_bin_ids, adj_hist_bin_ids)
                    # Update resulting_cluster_ids 
                    resulting_hist_bin_ids = np.append(resulting_hist_bin_ids, adj_hist_bin_ids)
            # Remove seeding cluster
            seeding_hist_bin_ids = np.setdiff1d(seeding_hist_bin_ids, seeding_hist_bin_id) 
        # get the points
        resulting_hist_bin_edges = [hist_rho_edges[np.min(resulting_hist_bin_ids)], hist_rho_edges[np.max(resulting_hist_bin_ids) + 1]]
        # get the inlier ids
        mask = (resulting_hist_bin_edges[0] <= pts_val) & (pts_val <= resulting_hist_bin_edges[1])
        inlier_pts_ids = pts_ids[mask]
        
    else:
        resulting_hist_bin_edges = np.array([np.min(pts_val), np.max(pts_val)])
        inlier_pts_ids = pts_ids
     
    # return
    return inlier_pts_ids, resulting_hist_bin_edges

# In[Get the largest peaks]
def get_largest_peak(data_val, data_range):
    """This function is to get the indices of data within the range having the largest  data
    """
    
    # count the points within the range
    data_stat = np.full((data_val.shape[0], 2), np.inf, dtype=np.uint16)
    data_stat[:, 0] = np.arange(data_val.shape[0], dtype=np.uint16)
    for count, range_val in enumerate(data_range):
        # get point within the range
        mask = (range_val[0] <= data_val) &(data_val <= range_val[1])
        data_stat[count, 1] = np.sum(mask)
    # get the largest
    mask = np.argmax(data_stat[:, 1])
    largest_data_range = data_range[mask]
    # get points within the largest
    mask = (largest_data_range[0] <= data_val) &(data_val <= largest_data_range[1])
    local_ids = np.where(mask == True)[0]
    
    # return
    return local_ids, largest_data_range
        

#%%    
def est_cylinder_parameters(pts_linear_ids, pts_xyz, pts_norm, cyl_threshold, constant):
    """This function is to estimate the shell wall parameters
    
    __, all_pts_xyz = voxelgrid.get_voxelptc(pts_linear_ids, index_type='linear')
    pts_linear_ids, pts_xyz, pts_norm = vert_voxel_linear_ids, vert_voxels_cent, vert_voxels_norm
    
    
    """
    # call functions/libs
    from utils.utils_geometry import cart2pol, cosin_vectors, residual, ransac_plane, ransac_circle, proj_3Dpoints_plane, rot_vectors
    # Get cyl_threshold
    ransac_threshold = cyl_threshold["ransac"]
    # Copy data
    randon_pts_linear_ids = copy.deepcopy(pts_linear_ids)
    # preallocation
    score = np.full((ransac_threshold["max_itr"], 8), np.inf, dtype=np.float32) #[xc.yc,zc,rc,tx,ty,tz,rmse]
    for itr in range(ransac_threshold["max_itr"]):
        # random extract pts_ids
        np.random.shuffle(randon_pts_linear_ids)
        fit_model_pts_ids = randon_pts_linear_ids[:ransac_threshold["min_ptc"]]
        
        # get the sub_pts
        fit_model_pts_ids = np.sort(fit_model_pts_ids)
        mask = np.in1d(pts_linear_ids, fit_model_pts_ids)
        fit_model_pts_xyz = pts_xyz[mask]
        fit_model_pts_norm = pts_norm[mask]
    
        # Estimate a direction
        best_fit_plane, inlier_ptc_ids = ransac_plane(fit_model_pts_norm, model_tol = 0.2*ransac_threshold["max_dist"], model_min_ptc = 3, max_iters = 100, inlier_ratio = 0.95, debug = False)
        
        # project the points to the plane
        fit_model_proj_pts_xyz = proj_3Dpoints_plane(fit_model_pts_xyz[inlier_ptc_ids], best_fit_plane)
        
        # Align the proj_ptc to xy plane
        rot_mat_oz = rot_vectors(constant['oz'], best_fit_plane[3:])
        fit_model_align_pts_xyz = np.matmul(fit_model_proj_pts_xyz, rot_mat_oz)
        
        # Fit the circle
        if fit_model_align_pts_xyz.shape[0] < 5: continue
        xc, yc, rc, fit_inlier_ptc_ids = ransac_circle(fit_model_align_pts_xyz[:, :2], model_tol =  0.2*ransac_threshold["max_dist"], outlier_tolerance = 0.2*ransac_threshold["max_dist"], max_iters = 250)
        fit_circle_cent = [xc, yc, np.mean(fit_model_align_pts_xyz[:,2])]
        # Convert to cylinder direction
        rot_mat_cyl = rot_vectors(best_fit_plane[3:], constant['oz'])
        fit_circle_cent = np.matmul(fit_circle_cent, rot_mat_cyl)
        
        # Update the cylinder cent and direction
        cyl_cent, cyl_dir = fit_circle_cent, best_fit_plane[3:]
        
        # Remove outlier voxels
        cosin_norm = cosin_vectors(pts_norm, cyl_dir)
        mask = np.abs(cosin_norm) <= cyl_threshold["cyl_norm"]
        cyl_pts_xyz = pts_xyz[mask]
#        cyl_pts_xyz, cyl_pts_norm = pts_xyz[mask], pts_norm[mask]
        
        # Compute the error
        cyl_align_pts_xyz = np.matmul(cyl_pts_xyz, rot_mat_oz)
        cyl_align_pts_rho, __ = cart2pol(cyl_align_pts_xyz[:, 0] - xc, cyl_align_pts_xyz[:, 1] - yc)
        cyl_align_pts_err = np.abs(cyl_align_pts_rho - rc)
        cyl_align_pts_rms = residual(cyl_align_pts_err)
        
        # Update the score
        score[itr, :3] = cyl_cent
        score[itr, 3] = rc
        score[itr, 4:7] = cyl_dir
        score[itr, 7] = cyl_align_pts_rms
    # Find the best
    mask = np.argmin(score[:, 7])
    cyl_paras = score[mask, :7]
    
    # return
    return cyl_paras
        
# In[Filter voxels_vertical surface]
def filter_vert_voxels(voxel_norm, cyl_threshold, constant):
    """This function is to filter vertical surfaces
    
    """
    # Call libs
    from utils.utils_geometry import cosin_vectors
    
    # Compute cosin
    cosin_norm = cosin_vectors(voxel_norm, constant["oz"])
    # Filter
    mask = np.abs(cosin_norm) <= cyl_threshold["vert_surface"]
    vert_voxels_local_ids = np.where(mask == True)[0]
    nonvert_voxels_local_ids = np.where(mask == False)[0]
    # return
    return vert_voxels_local_ids, nonvert_voxels_local_ids
#%%
def cal_voxel_features(voxelgrid, cyl_threshold):
    """This function is to compute features of the voxels
    """
    
    # Call libs
    from utils.utils_geometry import PCA_NormalVector
    
    # Preallocation
    voxel_linear_ids = np.arange(voxelgrid.ids.shape[0], dtype=np.uint32)
    voxels_cent = np.full((voxel_linear_ids.shape[0], 3), np.inf, dtype=np.float32)
    voxels_norm = np.full((voxel_linear_ids.shape[0], 3), np.inf, dtype=np.float32)
    
    # cal features
    for count, voxel_linear_id in enumerate(voxel_linear_ids):
        # Get the points within the voxel
        voxel_pts_ids = voxelgrid.ptc_ids[voxel_linear_id]
        voxel_pts_xyz = voxelgrid.points[voxel_pts_ids]
        
        # Voxel features
        voxel_cent = np.mean(voxel_pts_xyz, axis=0)
        voxel_norm = PCA_NormalVector(voxel_pts_xyz)
        
        # Update
        voxels_cent[count] = voxel_cent
        voxels_norm[count] = voxel_norm
    # Update features
    mask = np.any(np.isinf(voxels_norm), axis=1) | np.any(np.isnan(voxels_norm), axis=1)
    voxel_linear_ids = voxel_linear_ids[~mask]
    voxels_cent = voxels_cent[~mask]
    voxels_norm = voxels_norm[~mask]
    
    # return
    return voxel_linear_ids, voxels_cent, voxels_norm
#%%
