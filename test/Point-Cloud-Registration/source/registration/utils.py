from glob import glob
import os

from pandas.core.indexes.base import InvalidIndexError
import open3d as o3d
import numpy as np
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

def multiway_load_point_clouds(cylinders, voxel_size=0.02, radius=0.2, max_nn=10):
    pcds_full = []
    pcds_down = []
    for cyl in cylinders:
        p = Path(cyl.ptc_filepath)
        # filepath = p.parents[1] / f'tank_{cyl.tank_id}' / f'tank_{cyl.tank_id}_{p.name}'
        filepath = p.parents[1] / f'tank_{cyl.tank_id}' / f'{cyl.station.name}.txt'
        print(filepath)
        pcd = o3d.io.read_point_cloud(str(filepath),format='xyz')
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
    max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    print("Coarse alignment")  
    # coarse alignment
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    # fine aligment  
    print("Fine alignment")  
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        
    transformation_icp = icp_fine.transformation
    # get transformation matrix
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def multiway_full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, source_id + 2):
            target_id = target_id % n_pcds
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], 
                max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
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
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
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