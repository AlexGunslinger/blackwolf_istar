# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:27:37 2021

@author: Vinh Ho
@contributors: Tin Nguyen
"""
from re import match
import traceback
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from cylinder_extraction.station import Station
from utils.math_methods import perms, rms, rigid_transform_3D, pair_matching, trans_2D_to_3D

class StationAlignment():
    def __init__(self, station_0, station_1):
        self.station_0 = station_0
        self.station_1 = station_1
    def align(self, max_delta_radius=1, max_radius=100, refined=True, top_surface=True):
        print(f'Align station {self.station_0.name} and {self.station_1.name}:')
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
        num_match_cyls = min(num_match_cyls, 3)
        if num_match_cyls < 3:
            s0 = s0[valid_cyls_0, :]
            s1 = s1[valid_cyls_1, :]
            match_radius = True
            perm = False
        else:
            match_radius = False
            perm = True

        print(num_match_cyls)
        # print('s0')
        # print(s0)
        # print('s1')
        # print(s1)

        s0_cb = list(itertools.combinations(s0, num_match_cyls))
        s1_cb = list(itertools.combinations(s1, num_match_cyls))
        index_pair = []
        al_RMS = []
        for ia,A in enumerate(s0_cb):
            try:
                A = np.array(A)
                if match_radius:
                    A = A[np.argsort(A[:,1]),:]
                    # to keep z un-rotated
                    A = np.r_[A, A[-1,:].reshape([1,-1])]
                    A[-1, -1] = A[-1, -1] + 1
                # print('A')
                # print(A)
                A = np.transpose(A[:,-3:])
                # A = np.concatenate([A,np.zeros((1,A.shape[1]))]) 
                
                for ib,B in enumerate(s1_cb):
                    B = np.array(B)
                    if match_radius:
                        B = B[np.argsort(B[:,1]),:]
                        # to keep z un-rotated
                        B = np.r_[B, B[-1,:].reshape([1,-1])]
                        B[-1, -1] = B[-1, -1] + 1
                    # print('B')
                    # print(B)
                    B = np.transpose(B[:,-3:])
                    # B = np.concatenate([B,np.zeros((1,B.shape[1]))])
                    RMS, perm_id = pair_matching(A, B, perm)
                    
                    # print(RMS)
                    al_RMS.append(RMS)
                    index_pair.append([ia,ib])
            except Exception as exc:
                print(A)
                print(B)
                # traceback.print_exc()

        ibest = np.argmin(np.array([min(e) for e in al_RMS]))
        cb    = index_pair[ibest]
        cbest0 = np.array(s0_cb[cb[0]])
        cbest1 = np.array(s1_cb[cb[1]])
        if match_radius:
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
        else:
            RMS, perm_id = pair_matching(
                cbest0[:,-3:].transpose(),
                cbest1[:,-3:].transpose(),
                perm
            )
            # match the best pair
            iRMS_best = np.argmin(np.array(RMS))
            perm_best = perm_id[iRMS_best]
            cbest0 = cbest0[perm_best,:]
        # check radius
        # print(cbest0)
        # print(cbest1)
        for i in range(num_match_cyls):
            delta_radius = abs(cbest0[i][1] - cbest1[i][1])
            if delta_radius > max_delta_radius:
                print(cbest0[i][1], cbest1[i][1], delta_radius)
                return

        # get the transformation matrices
        best_r_01, best_t_01 = rigid_transform_3D(
            cbest0[:,-3:].transpose(),
            cbest1[:,-3:].transpose()
        )
        best_r_10, best_t_10 = rigid_transform_3D(
            cbest1[:,-3:].transpose(),
            cbest0[:,-3:].transpose()
        )
        print(best_r_10)
        print(best_t_10)
        print(RMS[iRMS_best])
        #
        
        # store information
        # station level
        self.transformation = {
            '01': {
                'rotation': best_r_01,
                'translation': best_t_01,
                'RMS': RMS[iRMS_best]
            },
            '10': {
                'rotation': best_r_10,
                'translation': best_t_10,
                'RMS': RMS[iRMS_best]
            }
        }
        self.station_0.transformation.append({
            'station': self.station_1,
            'rotation': best_r_10,
            'translation': best_t_10,
            'RMS': RMS[iRMS_best]
        })
        self.station_1.transformation.append({
            'station': self.station_0,
            'rotation': best_r_01,
            'translation': best_t_01,
            'RMS': RMS[iRMS_best]
        })
        # cylinder level
        if refined:
            cyl_type = 'cylinders'
        else:
            cyl_type = 'refined_cylinders'
        cylinders_0 = getattr(self.station_0, cyl_type)
        cylinders_1 = getattr(self.station_1, cyl_type)
        for t0, t1 in zip(cbest0, cbest1):
            i0 = [i for i,c in enumerate(cylinders_0) if c.id == t0[0]][0]
            i1 = [i for i,c in enumerate(cylinders_1) if c.id == t1[0]][0]
            print(cylinders_0[i0].id, cylinders_1[i1].id)

            # self.station_1.cylinders[i1].match.append({
            cylinders_1[i1].match.append({
                'cylinder': cylinders_0[i0],
                'rotation': best_r_01,
                'translation': best_t_01,
                'RMS': RMS[iRMS_best]
            })
            # self.station_0.cylinders[i0].match.append({
            cylinders_0[i0].match.append({
                'cylinder': cylinders_1[i1],
                'rotation': best_r_10,
                'translation': best_t_10,
                'RMS': RMS[iRMS_best]
            })
        print(cbest0)
        print(cbest1)
        