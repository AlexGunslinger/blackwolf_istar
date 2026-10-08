from glob import glob
import os
from multiprocessing import Pool
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

def _load_pcd(file_path, keep_original=True, downsample=True, voxel_size=0.02, radius=0.2, max_nn=10):
    """load point cloud - for use in parallel processing

    Args:
        file_path (str): path to scan data file
        keep_original (bool, optional)  : keep the original points. Defaults to True.
        downsample (bool, optional)     : down sample points. Defaults to True.
        voxel_size (float, optional)    : sampling size. Defaults to 0.02.
        radius (float, optional)        : radius for calculating normals. Defaults to 0.2.
        max_nn (int, optional)          : maximum number of neighbour points. Defaults to 10.

    Returns:
        ori_points, smpl_points, smpl_normals: original points, sampled points, sampled normals
    """
    print(file_path)
    pcd =  o3d.io.read_point_cloud(file_path)
    if keep_original:
        ori_points = np.asarray(pcd.points)
    else:
        ori_points = None
    # down sample
    if downsample:
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,max_nn=max_nn))
        smpl_points = np.asarray(pcd_down.points)
        smpl_normals = np.asarray(pcd_down.normals)
    else:
        smpl_points = None
        smpl_normals = None
    return ori_points, smpl_points, smpl_normals

def parallel_load_pcd_point_clouds(files, keep_original=True, downsample=True, voxel_size=0.02, radius=0.2, max_nn=10, num_proces=8):
    """load point clouds 

    Args:
        files (list)                    : list of source files
        keep_original (bool, optional)  : keep the original points. Defaults to True.
        downsample (bool, optional)     : down sample points. Defaults to True.
        voxel_size (float, optional)    : sampling size. Defaults to 0.02.
        radius (float, optional)        : radius for calculating normals. Defaults to 0.2.
        max_nn (int, optional)          : maximum number of neighbour points. Defaults to 10.
        num_proces (int, optional)      : number of processes. Defaults to 8.

    Returns:
        pcds_down   : down-sampled point clouds
        pcds_full   : original point clouds
    """
    pcds_full = []
    pcds_down = []
    n = len(files)
    pcds_full = []
    pcds_down = []
    with Pool(num_proces) as p:
        for i in range(0, n, num_proces):
            file_pack = files[i:i + num_proces]
            chunk_size = len(file_pack)
            res = p.starmap(
                _load_pcd,
                zip(
                    file_pack,
                    [keep_original] * chunk_size,
                    [downsample] * chunk_size,
                    [voxel_size] * chunk_size,
                    [radius] * chunk_size,
                    [max_nn] * chunk_size
                )
            )
            for ori_points, smpl_points, smpl_normals in res:
                if keep_original:
                    pcd_full = o3d.geometry.PointCloud()
                    pcd_full.points = o3d.utility.Vector3dVector(ori_points)
                    pcds_full.append(pcd_full)
                if downsample:
                    pcd_down = o3d.geometry.PointCloud()
                    pcd_down.points = o3d.utility.Vector3dVector(smpl_points)
                    pcd_down.normals = o3d.utility.Vector3dVector(smpl_normals)
                    pcds_down.append(pcd_down)
    
    return pcds_down, pcds_full

def load_pcd_point_clouds(files, voxel_size=0.02, radius=0.2, max_nn=10):
    pcds_full = []
    pcds_down = []
    for file_path in files:
        print(file_path)
        pcd =  o3d.io.read_point_cloud(file_path)
        pcds_full.append(pcd)
        # down sample
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,max_nn=max_nn))
        pcds_down.append(pcd_down)
    return pcds_down, pcds_full

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

def pairwise_registration(
    source, target,
    max_correspondence_distance_coarse, max_correspondence_distance_fine, method='p2l'):
    """fine align all stations in 2 steps - coarse and fine

    Args:
        source (PointCloud)                         : source point cloud
        target (PointCloud)                         : target point cloud
        max_correspondence_distance_coarse (float)  : searching distance for closet point in the course alignment step
        max_correspondence_distance_fine (float)    : searching distance for closet point in the first fine alignment step
        method (str, optional)                      : point to point or point to plane. Defaults to 'p2l'.

    Returns:
        transformation_icp, information_icp
    """
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
    print("FINE ALIGNMENT PHASE")  
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        icp_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)) 
        
    transformation_icp = icp_fine.transformation
    # get transformation matrix
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def pairwise_registration_3_phase(source, target,
    max_correspondence_distance_coarse, max_correspondence_distance_fine1, max_correspondence_distance_fine2, method='p2l'):
    """fine align all stations in 3 steps - coarse, fine1, and fine2

    Args:
        source (PointCloud)                         : source point cloud
        target (PointCloud)                         : target point cloud
        max_correspondence_distance_coarse (float)  : searching distance for closet point in the course alignment step
        max_correspondence_distance_fine (float)    : searching distance for closet point in the first fine alignment step
        max_correspondence_distance_fine2 (float)   : searching distance for closet point in the second fine alignment step
        method (str, optional)                      : point to point or point to plane. Defaults to 'p2l'.

    Returns:
        transformation_icp, information_icp
    """
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

def multiway_full_registration_internal(pcds, 
        max_correspondence_distance_coarse, 
        max_correspondence_distance_fine,
        pairs=None,
        method='p2l', num_steps=2):
    """fine align all stations

    Args:
        pcds ([PointCloud,])                        : list of point clouds
        max_correspondence_distance_coarse (float)  : searching distance for closet point in the course alignment step
        max_correspondence_distance_fine (float)    : searching distance for closet point in the fine alignment step
        pairs (list, optional)                      : list of tupple - source, target, and certainty indicator of matching 2 scans. Defaults to None.
        method (str, optional)                      : point to point or point to plane. Defaults to 'p2l'.
        num_steps (int, optional)                   : number of alignment steps. Defaults to 2.

    Returns:
        pose_graph  : PoseGraph after running fine alignment
        nodes       : maps nodes in pose_graph back to list of point clouds
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    n_pcds = len(pcds)
    nodes = []
    if pairs is None:
        # use input file order
        pairs = list(zip(
            range(0, n_pcds-1),
            range(1,n_pcds),
            [0] * (n_pcds - 1)
        ))
        # add a closure loop
        pairs.append((n_pcds-1, 0, 1))
    for source_id, target_id, is_closure in pairs:
        print(f'----SOURCE {source_id}----')
        print(f'===== source_id={source_id}, target_id={target_id} =====')

        if num_steps == 3:
            transformation_icp, information_icp = pairwise_registration_3_phase(
                pcds[source_id], pcds[target_id], 
                max_correspondence_distance_coarse, 
                max_correspondence_distance_fine * 3,
                max_correspondence_distance_fine,
                method=method)
        else:
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], 
                max_correspondence_distance_coarse, 
                max_correspondence_distance_fine,
                method=method)
        print("Build o3d.pipelines.registration.PoseGraph")
        if source_id not in nodes:
            odometry = np.identity(4)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
            nodes.append(source_id)
        else:
            odometry = pose_graph.nodes[nodes.index(source_id)].pose
        if target_id not in nodes:
            odometry = np.dot(transformation_icp, odometry)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(odometry)))
            nodes.append(target_id)
        
        source_node_ind = nodes.index(source_id)
        target_node_ind = nodes.index(target_id)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_ind,
                target_node_ind,
                transformation_icp,
                information_icp,
                uncertain=is_closure == 0
            )
        )
    return pose_graph, nodes

def load_transform_and_write(file_path, pose, write_dir):
    """reads, transform, then write scans back to disk

    Args:
        file_path (str) : path to file
        pose (Pose)     : a node in fine-aligned pose_graph which has a transformation matrix
        write_dir (str) : location to write aligned scan
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    # read
    pcd =  o3d.io.read_point_cloud(str(file_path))
    # transform
    pcd.transform(pose)
    # write
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    o3d.io.write_point_cloud(
        str(write_dir / file_path.name),
        pcd, print_progress=True
    )
