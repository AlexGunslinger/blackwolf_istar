# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:03:13 2021

@author: Vinh Ho
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    for perm in perm_id:
        aa = (A[:,perm])
        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(aa, B)
        
        if abs(ret_R[-1,-1] - 1) < epsilon:
            # no rotation around the z axis
            # Compare the recovered R and t with the original
            B2 = (ret_R@aa) + ret_t
            err = rms(B2,B)
        else:
            # B2 = (ret_R@aa) + ret_t
            # err = rms(B2,B)
            # print(err)
            # print(ret_R)
            err = 100 # just some big number
        # Find the root mean squared error
        RMS.append(err) 
        
    return RMS, perm_id
  
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