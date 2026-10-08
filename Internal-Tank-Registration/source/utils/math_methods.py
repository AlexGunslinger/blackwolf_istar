# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:03:13 2021

@author: Vinh Ho
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN

def perms(vec):
    n  = list(itertools.permutations(vec))
    nc = np.array(n)
    return nc 

def rms(a,b):
    err = a - b
    err = err * err
    err = np.mean(err)
    return np.sqrt(err)

def rigid_transform_2D(from_pt, to_pt):
    # Fill the matrices
    A_data = []
    for pt in from_pt:
      A_data.append( [-pt[1], pt[0], 1, 0] )
      A_data.append( [ pt[0], pt[1], 0, 1] )
    
    b_data = []
    for pt in to_pt:
      b_data.append(pt[0])
      b_data.append(pt[1])
    
    # Solve
    A = np.matrix( A_data )
    b = np.matrix( b_data ).T
    c = np.linalg.lstsq(A, b,rcond=-1)[0].T
    c = np.array(c)[0]
    
    R = np.array([[c[1],-c[0]],
                  [c[0],c[1]]])
    t = np.array([[c[2]],
                  [c[3]]])
    # p2 = np.array(from_pt).transpose()  
    # p3 = (R@p2) + t

    return R, t


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def pair_matching(A,B, perm=True, epsilon=0.1):
    if perm:
        perm_id = perms(range(A.shape[1]))
    else:
        perm_id = [range(A.shape[1])]
    RMS  = []
    trans_matrix = []
    for perm in perm_id:
        aa = (A[:,perm])
        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(aa, B)
        
        # check rotation around x,y
        # theta_x = np.arctan2(ret_R[2,1], ret_R[2,2])
        # theta_y = np.arctan2(-ret_R[2,0], np.sqrt(ret_R[2,1] ** 2 + ret_R[2,2] ** 2))
        # print('theta', theta_x * 180 / np.pi, theta_y * 180 / np.pi)

        # if True:#(abs(theta_x) < 30):# and (abs(theta_y) < 30):
        if abs(ret_R[-1,-1] - 1) < epsilon:
            # allow only no or very small changes to the z axis
            # Compare the recovered R and t with the original
            B2 = (ret_R@aa) + ret_t
            err = rms(B2,B)
            # print(f'rms = {err}')
        else:
            # B2 = (ret_R@aa) + ret_t
            # err = rms(B2,B)
            # print(err)
            # print(ret_R)
            # print('rotating around z')
            # print(ret_R)
            err = 100 # just some big number
        # Find the root mean squared error
        RMS.append(err)
        trans_matrix.append([ret_R, ret_t])
        
    return RMS, perm_id, trans_matrix
  
def trans_2D_to_3D(A):
    if A.shape[0] !=2:
        A = A.transpose()
    A = np.concatenate([A,np.zeros((1,A.shape[1]))]) 
    return A
    # print(rms(B2,B))    
    # plt.scatter(B2[0,:],B2[1,:]);
    # plt.scatter(B[0,:],B[1,:])
    # plt.show()


def rotate_vectors(vec1, vec2):
    """ Find the rotation matrix (R) that aligns vec1 to vec2
    vec2 @ R = vec1
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    Source: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def project_point_onto_plane(apoint, plane_coefficients): # essentially finds the point on the plane that is closest to the specified point
        # the plane_coefficients are [a,b,c,d], where the plane is ax + by + cz + d = 0

        # First, define a plane using cooeficients a, b, c, d such that ax + by + cz + d = 0
        a, b, c, d = plane_coefficients
        
        # Now, define a point in space (s,u,v)
        s, u, v = apoint
        
        # calculate distance
        distance = abs((a * s + b * u + c * v + d)) / math.sqrt(a * a + b * b + c * c)

        # the formula of a line perpendicular to the plan passing through (s,u,v) is:
        #x = s + at
        #y = u + bt
        #z = v + ct       
        t = (-d - a*s - b*u - c*v) / (a*a + b*b + c*c)
        
        # here's the point closest on the plane
        x = s + a*t
        y = u + b*t
        z = v + c*t 
        
        return np.array([x,y,z]), distance

def rotate_around_z(points, theta):
    """
    points: nx3
    """
    theta = math.pi / 180.0 * theta
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1.0]])
    
    return points @ R

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = (phi + 4 * np.pi) % (2 * np.pi)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def check_overlap_line(s0, s1):
    n0 = len(s0)
    n1 = len(s1)
    if n0 == 2 and n1 == 2:
        # calc overlapping distance
        _, d0 = check_overlap_line([s0[0]], [s1[0]])
        _, d1 = check_overlap_line([s0[1]], [s1[1]])
        #########################
        return True, d0+d1
    elif n0 == 1 and n1 == 1:
        s0 = s0[0]
        s1 = s1[0]
        s_max = max(max(s0), max(s1))
        s_min = min(min(s0), min(s1))
        if s_max - s_min < s0[1] - s0[0] + s1[1] - s1[0]:
            left = max(s0[0], s1[0])
            right = min(s0[1], s1[1])
            return True, right - left
    else:
        if n0 == 2:
            s0, s1 = s1, s0
        # s0: 1 fragment
        # s1: 2 fragments
        overlap0, d0 = check_overlap_line(s0, [s1[0]])
        if overlap0:
            return True, d0
        else:
            overlap1, d1 = check_overlap_line(s0, [s1[1]])
            if overlap1:
                return True, d1
    return False, 0
    
def calc_overlap_curve(c0, c1):
    # c0,c1: 2-column matrices
    # this function is 
    # from cartesian to polar coordinates
    _, phi_0 = cart2pol(c0[:,0], c0[:,1])
    _, phi_1 = cart2pol(c1[:,0], c1[:,1])
    # get 'connected' segments
    def get_segments(phi):
        phi = np.sort(phi)
        diff = np.diff(phi)
        ind = np.where(diff > np.pi / 2)[0] + 1
        if ind.size == 0:
            s = [phi[[0,-1]]]
        else:
            # 2 fragments
            ind = ind[0]
            s = [phi[[0, ind-1]], phi[[ind, -1]]]
        return s
    s0 = get_segments(phi_0)
    s1 = get_segments(phi_1)
    # get overlap
    # print(s0)
    # print(s1)
    _, d = check_overlap_line(s0, s1)
    return d

def calc_circle_span(coord):
    _, phi = cart2pol(coord[:,0], coord[:,1])
    # phi = phi.reshape([-1,1])
    # max distance: 5 degree
    # db_model = DBSCAN(eps=5/180*np.pi, min_samples=2)
    # db_model.fit(phi)
    # span = 0
    # sample_labels = np.array(db_model.labels_)
    # for lbl in np.unique(sample_labels):
    #     cluster = phi[sample_labels == lbl]
    #     span += cluster.max() - cluster.min()
    # return span / (2 * np.pi)
    eps = 5/180*np.pi
    phi = np.sort(phi)
    phi = np.r_[0, phi, 2*np.pi]
    diff = np.diff(phi)
    # diff = np.abs(diff)
    diff[diff < eps] = 0

    gap = np.sum(diff)
    
    return 1 - gap / (2*np.pi)