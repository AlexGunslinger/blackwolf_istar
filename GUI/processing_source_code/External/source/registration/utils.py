from glob import glob
import os

import open3d as o3d
import numpy as np
import pandas as pd
from pathlib import Path

# ===========================
# Multiway

def multiway_filter_point_clouds(cylinders, radius_offset_max=0.2):
    inlier_cylinders = []
    outlier_cylinders = []
    all_radius = [c.radius for c in cylinders]
    # estimate radius
    r = np.median(np.array(all_radius))
    for cyl in cylinders:
        if abs(cyl.radius - r) <= radius_offset_max:
            inlier_cylinders.append(cyl)
        else:
            outlier_cylinders.append(cyl)
    return inlier_cylinders, outlier_cylinders

def multiway_load_point_clouds(cylinders, update_filepath=True, file_dir=None, voxel_size=0.02, radius=0.2, max_nn=10, full_normal=False):
    pcds_full = []
    pcds_down = []
    for cyl in cylinders:
        # p = Path(cyl.ptc_filepath)
        # filepath = p.parents[1] / f'tank_{cyl.tank_id}' / f'tank_{cyl.tank_id}_{p.name}'
        if update_filepath:
            if file_dir is None:
                file_dir = f'tank_{cyl.tank_id}'
            cyl.update_filepath(file_dir, num_keep_level=1)
        # filepath = cyl.ptc_filepath
        p = Path(cyl.ptc_filepath)
        if p.suffix == '.txt':
            filepath = p.parents[0] / f'{cyl.station.name}.txt'
            print(filepath)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                pd.read_csv(filepath, delimiter=" ").values
            )
        else:
            # pcd
            print(cyl.ptc_filepath)
            pcd =  o3d.io.read_point_cloud(cyl.ptc_filepath)
        # if os.path.getsize(filepath) < 2000414715:
        #     pcd = o3d.io.read_point_cloud(str(filepath),format='xyz')
        # else:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(
        #         pd.read_csv(filepath, delimiter=" ").values
        #     )
        if full_normal:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=max_nn)
            )
        pcds_full.append(pcd)
        # down sample
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,max_nn=max_nn))
        pcds_down.append(pcd_down)
    return pcds_down, pcds_full

def multiway_load_point_clouds_2(
    data_dir, station_ids, fmt='*{:03d}*.txt', voxel_size=0.0, radius=0.2, max_nn=10):
    pcds_full = []
    pcds_down = []
    for i in station_ids:  
        filepath = glob(os.path.join(
            data_dir,
            fmt.format(i)
        ))[0]
        print(filepath)
        pcd = o3d.io.read_point_cloud(filepath,format='xyz')
        pcds_full.append(pcd)
        # down sample
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,max_nn=max_nn))
        pcds_down.append(pcd_down)
    return pcds_down, pcds_full

def pairwise_registration(
    source, target,
    max_correspondence_distance_coarse, max_correspondence_distance_fine1, max_correspondence_distance_fine2, method='p2l'):
    # print("Apply point-to-plane ICP")
    print("Coarse alignment")  
    if method == 'p2l':
        print("Apply point-to-plane ICP")
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        print("Apply point-to-point ICP")
    # coarse alignment
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        icp_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    # fine aligment phase I  
    print("FINE ALIGNMENT PHASE I")  
    icp_fine1 = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine1,
        icp_coarse.transformation,
        icp_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)) 

    # fine aligment phase II  
    print("FINE ALIGNMENT PHASE II")  
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine2,
        icp_fine1.transformation,
        icp_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        
    transformation_icp = icp_fine.transformation
    # get transformation matrix
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine2,
        icp_fine.transformation)
    return transformation_icp, information_icp

# def find_neighbour_pcd(src_id, pcds, distance_min = 0.01):
#     num_pcds = len(pcds)
#     for ind in range(src_id+1, num_pcds):
#         # calculate the minimum distance to the source pcd
#         d = min(pcds[src_id].compute_point_cloud_distance(pcds[ind]))
#         if 

def check_neighbour_pcd(src_id, target_id, pcds, distance_min = 0.1, percentage_overlap_min = 0.1):
    # calculate the minimum distance to the source pcd
    d = pcds[src_id].compute_point_cloud_distance(pcds[target_id])
    # print(src_id, target_id, d)
    num_points_min = min(len(pcds[src_id].points), len(pcds[target_id].points))
    # num overlapping points
    d = np.array(d)
    d = d < distance_min
    num_overlap = d.sum()
    print(num_overlap, num_points_min)
    if num_overlap > (num_points_min * percentage_overlap_min):
        return True
    else:
        return False

def multiway_full_registration_with_sequence(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, num_overlap=1, method='p2l'):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        print(f'----SOURCE {source_id}----')
        # for target_id in range(source_id + 1, n_pcds):
        for target_id in range(source_id + 1, source_id + 1 + num_overlap):
            # if not check_neighbour_pcd(source_id, target_id%n_pcds, pcds):
            #     continue
            if target_id >= n_pcds:
                continue
            # target_id = target_id % n_pcds
            print(f'===== source_id={source_id}, target_id={target_id % n_pcds} =====')
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id % n_pcds], 
                max_correspondence_distance_coarse, max_correspondence_distance_fine,
                method=method)
            print("Build o3d.pipelines.registration.PoseGraph")
            if True:#target_id == ((source_id + 1) % n_pcds):  # odometry case
                print('odemetry')
                odometry = np.dot(transformation_icp, odometry)
                if target_id > source_id:
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                print('loop')
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def multiway_full_registration_with_pairs(pcds, pair_info, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine1,max_correspondence_distance_fine2, method='p2l'):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id, target_id in pair_info:
        print(f'----SOURCE, TARGET = {source_id}, {target_id}')
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id], pcds[target_id % n_pcds], 
            max_correspondence_distance_coarse, max_correspondence_distance_fine1,
            max_correspondence_distance_fine2,
            method=method)
        print("Build o3d.pipelines.registration.PoseGraph")
        # odometry case
        print('odemetry')
        odometry = np.dot(transformation_icp, odometry)
        # if target_id > source_id:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                        target_id,
                                                        transformation_icp,
                                                        information_icp,
                                                        uncertain=False))
    return pose_graph

def combine_point_cloud(pcds, pose_graph):
    # combine all points clouds, adding ids as a scalar field
    pcd_full = np.empty(shape=[0,4])
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        A = np.asarray(pcds[point_id].points)
        B = np.c_[ A, np.ones( A.shape[0] )*point_id ]
        pcd_full = np.r_[pcd_full,B]
    return pcd_full