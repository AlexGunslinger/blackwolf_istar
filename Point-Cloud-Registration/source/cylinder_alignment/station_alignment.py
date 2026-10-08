# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:27:37 2021

@author: Vinh Ho
@contributors: Tin Nguyen
"""
from multiprocessing import Pool
import numpy as np
import itertools
from copy import deepcopy
from sklearn.cluster import DBSCAN

from ..utils.math_methods import rigid_transform_3D, pair_matching, rotate_around_z

class StationAlignment():
    def __init__(self, station_0, station_1):
        self.station_0 = station_0
        self.station_1 = station_1
    def align(self, max_delta_radius=1, max_radius=100, refined=True, top_surface=True, default_num_points=3, scan_radius=None, RMS_min=1, radius_diff_min=0.1):
        """align external stations

        Args:
            max_delta_radius (int, optional)    : only match cylinders within this distance. Defaults to 1.
            max_radius (int, optional)          : ignore cylinders with larger radius. Defaults to 100.
            refined (bool, optional)            : use refined or rough fit cylinders. Defaults to True.
            top_surface (bool, optional)        : whether to use center of the top surfaces. Defaults to True.
            default_num_points (int, optional)  : number of centers for matching. Defaults to 3.
            scan_radius (float, optional)       : estimate target tank's radius. Defaults to None.
            RMS_min (int, optional)             : upper bound RMS error for matching 2 sets of centers (from 2 stations). Defaults to 1.
            radius_diff_min (float, optional)   : condition on radiuses for matching 2 cylinders - should not be too different. Defaults to 0.1.
        """
        # get tank information
        s0 = self.station_0.to_numpy(refined, top_surface)
        s1 = self.station_1.to_numpy(refined, top_surface)
        # find potential matching sets
        # find maximum cylinders
        valid_cyls_0 = s0[:,1] < max_radius
        valid_cyls_1 = s1[:,1] < max_radius
        num_cyls_0 = sum(valid_cyls_0)
        num_cyls_1 = sum(valid_cyls_1)
        # better approach needed
        num_match_cyls = min(num_cyls_0, num_cyls_1)
        num_match_cyls = min(num_match_cyls, default_num_points)
        if num_match_cyls < 3:
            s0 = s0[valid_cyls_0, :]
            s1 = s1[valid_cyls_1, :]
            match_radius = True
            perm = False
        else:
            match_radius = False
            perm = True

        # this flag indicates if more than one matching options are possible (and it is not currently possible to determine which is the correct one)
        uncertain_flag = False

        if num_match_cyls > 1:
            s0_cb = list(itertools.combinations(s0, num_match_cyls))
            s1_cb = list(itertools.combinations(s1, num_match_cyls))
            index_pair = []
            al_RMS = []
            al_trans_matrix = []
            al_perm_id = []
            al_cb_pair = []
            for ia,A in enumerate(s0_cb):
                try:
                    A = np.array(A)
                    radius_A = A[:,1]
                    if match_radius:
                        A = A[np.argsort(A[:,1]),:]
                        # to keep z un-rotated
                        A = np.r_[A, A[-1,:].reshape([1,-1])]
                        A[-1, -1] = A[-1, -1] + 1
                    A = np.transpose(A[:,-3:])
                    
                    for ib,B in enumerate(s1_cb):
                        B = np.array(B)
                        radius_B = B[:,1]
                        if match_radius:
                            B = B[np.argsort(B[:,1]),:]
                            # to keep z un-rotated
                            B = np.r_[B, B[-1,:].reshape([1,-1])]
                            B[-1, -1] = B[-1, -1] + 1
                        B = np.transpose(B[:,-3:])
                        if match_radius:
                            close_radius = True
                            for rA, rB in zip(radius_A, radius_B):
                                if abs(rA - rB) > max_delta_radius:
                                    close_radius = False
                                    break
                            if not close_radius:
                                continue
                        RMS, perm_id, trans_matrix = pair_matching(A, B, perm)
                        
                        # print(RMS)
                        al_cb_pair.append([A,B])
                        al_RMS.append(RMS)
                        al_trans_matrix.append(trans_matrix)
                        al_perm_id.append(perm_id)
                        index_pair.append([ia,ib])
                except Exception as exc:
                    print(A)
                    print(B)
            if len(al_RMS) == 0:
                if num_match_cyls > 1:
                    self.align(
                        max_delta_radius, max_radius, refined,
                        top_surface, num_match_cyls-1, scan_radius)
                return
            if num_match_cyls == 2:
                ibest = np.argmin(np.array([min(e) for e in al_RMS]))
                cb    = index_pair[ibest]
                cbest0 = np.array(s0_cb[cb[0]])
                cbest1 = np.array(s1_cb[cb[1]])

                # order radius
                cbest0 = cbest0[np.argsort(cbest0[:,1]),:]
                cbest1 = cbest1[np.argsort(cbest1[:,1]),:]
                # to keep z un-rotated
                cbest0 = np.r_[cbest0, cbest0[-1,:].reshape([1,-1])]
                cbest0[-1, -1] = cbest0[-1, -1] + 1
                cbest1 = np.r_[cbest1, cbest1[-1,:].reshape([1,-1])]
                cbest1[-1, -1] = cbest1[-1, -1] + 1

                # find the best order of 2 sets
                if match_radius:
                    RMS = al_RMS
                    iRMS_best = ibest
                best_RMS = RMS[iRMS_best]

                # check radius difference
                if abs(cbest0[0,1] - cbest0[1,1]) < radius_diff_min*max(cbest0[0,1], cbest0[1,1]):
                    # there are two possible pairing options
                    # returnning both
                    uncertain_flag = True
                    matching_options = [
                        [cbest0, cbest1, best_RMS]
                    ]
                    # reverse order
                    cbest1r = cbest1[1::-1, :]
                    # to keep z un-rotated
                    cbest1r = np.r_[cbest1r, cbest1r[-1,:].reshape([1,-1])]
                    cbest1r[-1, -1] = cbest1r[-1, -1] + 1
                    matching_options.append(
                        [cbest0, cbest1r, best_RMS]
                    )

            else:
                # get all potential candidates, which has RMS lower than the given threshold
                # print(al_RMS)
                potential_groups_ind = [i for i,rms in enumerate(al_RMS) if min(rms) < RMS_min]
                # all points in each station
                all_points_0 = deepcopy(s0)
                all_points_0 = np.array(all_points_0)
                P0 = np.transpose(all_points_0[:,-3:])
                all_points_1 = deepcopy(s1)
                all_points_1 = np.array(all_points_1)
                P1 = np.transpose(all_points_1[:,-3:])
                max_matching_pairs = 0
                best_RMS = 100
                cbest0 = []
                cbest1 = []
                # for each group pair
                for gind in potential_groups_ind:
                    gr0_ind, gr1_ind = index_pair[gind]
                    # 2 groups from 2 stations
                    gr0 = np.array(s0_cb[gr0_ind])
                    gr1 = np.array(s1_cb[gr1_ind])
                    # RMS
                    gr_RMS = al_RMS[gind]
                    gr_trans_matrix = al_trans_matrix[gind]
                    gr_perm_id = al_perm_id[gind]
                    gr_cb_pair = al_cb_pair[gind]
                    # iterate through permutation
                    for _rms, _perm_id, _trans_matrix in zip(gr_RMS, gr_perm_id, gr_trans_matrix):
                        if _rms > RMS_min:
                            # print(f'{_rms} is larger than the chosen RMS_min threshold = {RMS_min}')
                            continue
                        pass_radius_check = True
                        c0 = gr0[_perm_id,:]
                        c1 = gr1
                        for i in range(num_match_cyls):
                            delta_radius = abs(c0[i][1] - c1[i][1])
                            if delta_radius > max_delta_radius:
                                pass_radius_check = False
                                break
                        if not pass_radius_check:
                            # print('fail radius')
                            # print(c0), print(c1)
                            continue
                        
                        ret_R, ret_t = _trans_matrix
                        P0_transformed = (ret_R@P0) + ret_t
                        # group points
                        P = np.c_[P0_transformed, P1]
                        db_model = DBSCAN(eps=1, min_samples=2)
                        db_model.fit(P)
                        # number of pairs
                        num_matching_pairs = max(db_model.labels_) + 1
                        if num_matching_pairs > max_matching_pairs:
                            cbest0 = gr0[_perm_id,:]
                            cbest1 = gr1
                            best_RMS = _rms
                        elif num_matching_pairs == max_matching_pairs:
                            if _rms < best_RMS:
                                cbest0 = gr0[_perm_id,:]
                                cbest1 = gr1
                                best_RMS = _rms
                                
            if cbest0 == []:
                if num_match_cyls > 1:
                    self.align(
                        max_delta_radius, max_radius, refined,
                        top_surface, num_match_cyls-1, scan_radius)
                return
            if isinstance(best_RMS, list):
                best_RMS = best_RMS[0]
            if num_match_cyls < 3:
                RMS_best = best_RMS + 50
            else:
                RMS_best = best_RMS
            # print(f'RMS = {RMS_best}')
            # check radius
            # print(cbest0)
            # print(cbest1)
            for i in range(num_match_cyls):
                delta_radius = abs(cbest0[i][1] - cbest1[i][1])
                if delta_radius > max_delta_radius:
                    print(cbest0[i][1], cbest1[i][1], delta_radius)
                    if num_match_cyls > 1:
                        self.align(
                            max_delta_radius, max_radius, refined,
                            top_surface, num_match_cyls-1, scan_radius)
                    return
        else:
            return

        # get the transformation matrices
        # 01: 1: target (fixed), 0: aligned (transformed)
        if num_match_cyls > 1:
            if uncertain_flag:
                # 2 options for matching 2 pairs
                best_r_01 = []
                best_t_01 = []
                best_r_10 = []
                best_t_10 = []
                for _match in matching_options:
                    _cbest0, _cbest1, _rms = _match
                    _r_01, _t_01 = rigid_transform_3D(
                        _cbest0[:,-3:].transpose(),
                        _cbest1[:,-3:].transpose()
                    )
                    _r_10, _t_10 = rigid_transform_3D(
                        _cbest1[:,-3:].transpose(),
                        _cbest0[:,-3:].transpose()
                    )
                    best_r_01.append(_r_01)
                    best_t_01.append(_t_01)
                    best_r_10.append(_r_10)
                    best_t_10.append(_t_10)
            else:
                best_r_01, best_t_01 = rigid_transform_3D(
                    cbest0[:,-3:].transpose(),
                    cbest1[:,-3:].transpose()
                )
                best_r_10, best_t_10 = rigid_transform_3D(
                    cbest1[:,-3:].transpose(),
                    cbest0[:,-3:].transpose()
                )
        else:
            best_r_01 = np.identity(3)
            best_t_10 = np.reshape(cbest0[-3:] - cbest1[-3:], [3,1])
            best_r_10 = np.identity(3)
            best_t_01 = np.reshape(cbest1[-3:] - cbest0[-3:], [3,1])

        # store information
        # station level
        self.transformation = {
            '01': {
                'rotation': best_r_01,
                'translation': best_t_01,
                'RMS': RMS_best,
                'uncertain_flag': uncertain_flag,
                'num_matching_cylinders': num_match_cyls
            },
            '10': {
                'rotation': best_r_10,
                'translation': best_t_10,
                'RMS': RMS_best,
                'uncertain_flag': uncertain_flag,
                'num_matching_cylinders': num_match_cyls
            }
        }
        self.station_0.transformation.append({
            'station': self.station_1,
            'rotation': best_r_10,
            'translation': best_t_10,
            'RMS': RMS_best,
            'uncertain_flag': uncertain_flag,
            'num_matching_cylinders': num_match_cyls
        })
        self.station_1.transformation.append({
            'station': self.station_0,
            'rotation': best_r_01,
            'translation': best_t_01,
            'RMS': RMS_best,
            'uncertain_flag': uncertain_flag,
            'num_matching_cylinders': num_match_cyls
        })
        # cylinder level
        if refined:
            cyl_type = 'refined_cylinders'
        else:
            cyl_type = 'cylinders'
        cylinders_0 = getattr(self.station_0, cyl_type)
        cylinders_1 = getattr(self.station_1, cyl_type)
        # a fix for matching 1 pair of cylinders
        if cbest0.ndim == 1:
            cbest0 = [cbest0]
            cbest1 = [cbest1]
        # print(cbest0), print(cbest1)
        for t0, t1 in zip(cbest0, cbest1):
            i0 = [i for i,c in enumerate(cylinders_0) if c.id == t0[0]][0]
            i1 = [i for i,c in enumerate(cylinders_1) if c.id == t1[0]][0]
            
            cylinders_1[i1].match.append({
                'cylinder': cylinders_0[i0],
                'rotation': best_r_01,
                'translation': best_t_01,
                'RMS': RMS_best,
                'uncertain_flag': uncertain_flag,
                'num_matching_cylinders': num_match_cyls
            })
            
            cylinders_0[i0].match.append({
                'cylinder': cylinders_1[i1],
                'rotation': best_r_10,
                'translation': best_t_10,
                'RMS': RMS_best,
                'uncertain_flag': uncertain_flag,
                'num_matching_cylinders': num_match_cyls
            })

    def align_internal(self, alpha=1, max_distance=0.5, object='column'):
        """align internal stations

        Args:
            alpha (int, optional)           : angle step size for rotating a cylinder. Defaults to 1.
            max_distance (float, optional)  : only match centers within this distance. Defaults to 0.5.
            object (str, optional)          : objects for alignment - column or manway. Defaults to 'column'.

        Returns:
            (boolean): if alignment finishes succesfully
        """
        # st0: ref
        # rotate st1
        if object == 'column':
            centers_0 = np.concatenate(
                [c.get_center(loc='ground') for c in self.station_0.refined_cylinders]
            )
            # 2D projection
            centers_0[:,2] = 0
            centers_1 = np.concatenate(
                [c.get_center(loc='ground') for c in self.station_1.refined_cylinders]
            )
            # 2D projection
            centers_1[:,2] = 0
        else:
            # manway
            centers_0 = np.r_[
                [way_data['fit_circle'][:3] for _, way_data in self.station_0.manway.items()]
            ]
            centers_1 = np.r_[
                [way_data['fit_circle'][:3] for _, way_data in self.station_1.manway.items()]
            ]

        self.station_0.num_columns = centers_0.shape[0]
        self.station_1.num_columns = centers_1.shape[0]

        angle = 0
        best_num_pairs = 0
        best_angle = None
        best_mae = 1000
        best_pairs = None
        while(angle < 360):
            # rotate st1
            rotated_centers_1 = rotate_around_z(centers_1, angle)
            # match
            errors, pairs = match_centers(centers_0, rotated_centers_1, max_distance)
            num_pairs = len(pairs.keys())
            mae = np.mean(np.array(errors))
            if best_num_pairs < num_pairs:
                best_num_pairs = num_pairs
                best_angle = angle
                best_pairs = pairs
            elif best_num_pairs == num_pairs:
                if mae < best_mae:
                    best_angle = angle
                    best_mae = mae
                    best_pairs = pairs
            # increase angle
            angle += alpha
        
        for k in range(1,5):
            centers_0 = np.r_[
                centers_0,
                np.array([0,0,2*k]).reshape((1,3))
            ]
            centers_1 = np.r_[
                centers_1,
                np.array([0,0,2*k]).reshape((1,3))
            ]
            best_pairs[centers_0.shape[0]-1] = centers_1.shape[0]-1

        # now that we have identified the best pairing
        i = list(best_pairs.keys())
        j = list(best_pairs.values())

        # transformation matrix
        best_r_01, best_t_01 = rigid_transform_3D(
            centers_0[i].transpose(),
            centers_1[j].transpose()
        )
        best_r_10, best_t_10 = rigid_transform_3D(
            centers_1[j].transpose(),
            centers_0[i].transpose()
        )

        self.transformation = {
            '01': {
                'rotation': best_r_01,
                'translation': best_t_01,
                'RMS': best_mae,
                'uncertain_flag': False,
                'num_matching_cylinders': best_num_pairs
            },
            '10': {
                'rotation': best_r_10,
                'translation': best_t_10,
                'RMS': best_mae,
                'uncertain_flag': False,
                'num_matching_cylinders': best_num_pairs
            }
        }
        self.station_0.transformation.append({
            'station': self.station_1,
            'rotation': best_r_10,
            'translation': best_t_10,
            'RMS': best_mae,
            'uncertain_flag': False,
            'num_matching_cylinders': best_num_pairs
        })
        self.station_1.transformation.append({
            'station': self.station_0,
            'rotation': best_r_01,
            'translation': best_t_01,
            'RMS': best_mae,
            'uncertain_flag': False,
            'num_matching_cylinders': best_num_pairs
        })
        return True

def match_centers(centers_0, centers_1, max_distance=0.5):
    """calculate overlap between two sets of point

    Args:
        centers_0 (numpy array)         : first set of (center) points
        centers_1 (numpy array)         : second set of (center) points
        max_distance (float, optional)  : only match points within this distance . Defaults to 0.5.

    Returns:
        errors (float)  : lowest/best matching error
        pairs (dict)    : best matches
    """
    pairs = {}
    errors = []
    used_j = []
    for i, c1 in enumerate(centers_1):
        d = np.linalg.norm(
            c1 - centers_0, axis=1
        )
        d[used_j] = 2*max_distance
        j = np.argmin(d)
        d_min = d[j]
        if d_min < max_distance:
            pairs[j] = i
            errors.append(d_min)

    return errors, pairs

def find_best_match(centers_0, centers_1, max_distance, alpha):
    """find best match between 2 stations

    Args:
        centers_0 (numpy array)         : center points of various (points) clusters from the first station
        centers_1 (numpy array)         : center points of various (points) clusters from the first station
        max_distance (float, optional)  : only match centers within this distance . Defaults to 0.5.
        alpha (int, optional)           : angle step size for rotating a cylinder. Defaults to 1.

    Returns:
    
        max_num_pairs (int) : maximum possible matching pairs
        best_pairs (dict)   : best matches
        min_mae (float)     : best matching error
    """
    angles = np.arange(0, 360, alpha).tolist()
    num_angle = len(angles)
    with Pool(10) as p:
        match_info = p.starmap(
            _rotate_and_match,
            zip(
                [centers_0] * num_angle,
                [centers_1] * num_angle,
                angles,
                [max_distance] * num_angle
            )
        )
    # pairs, num_pairs, mae
    match_info = np.array(match_info)
    # find angles give best num_pairs
    max_num_pairs = match_info[:, 1].max()
    # keep angles with this number of pairs
    match_info = match_info[
        match_info[:, 1] == max_num_pairs
    ]
    # among these, find the one with the least mae
    ind = match_info[:, 2].argmin()
    best_pairs, _, min_mae = match_info[ind]
    return max_num_pairs, best_pairs, min_mae

def _rotate_and_match(centers_0, centers_1, angle, max_distance):
    """rotate a station to find the best match with another station

    Args:
        centers_0 (numpy array)         : center points of various (points) clusters from the first station
        centers_1 (numpy array)         : center points of various (points) clusters from the first station
        angle(int)                      : the angle to rotate the second station
        max_distance (float, optional)  : only match centers within this distance . Defaults to 0.5.

    Returns:
        pairs (dict)    : best matches
        num_pairs (int) : number of maching clusters/centers
        mae (float)     : best matching error
    """
    # rotate st1
    rotated_centers_1 = rotate_around_z(centers_1, angle)
    # match
    errors, pairs = match_centers(centers_0, rotated_centers_1, max_distance)
    num_pairs = len(pairs.keys())
    mae = np.mean(np.array(errors))
    return pairs, num_pairs, mae
    


        