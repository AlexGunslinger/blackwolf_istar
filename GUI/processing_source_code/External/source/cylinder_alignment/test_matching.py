# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:27:37 2021

@author: Vinh Ho
@contributors: Tin Nguyen
"""
#%% 
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from ..utils.math_methods import perms, rms, rigid_transform_3D, pair_matching, trans_2D_to_3D

#%% raw data
setup_004 = np.array( [[1,	9.132025,	-28.519823, 23.851562],
      [3,	10.234577,	-1.0320892, 25.561563],
      [4,	7.262211,	-1.1865463, -0.5457942],
      [5, 10.47777,	13.80102, -20.920395],
      [8,	10.470699,	34.272377, -40.157074],
      [10,	2.666495, 56.662113, -30.964066]])

setup_005 = np.array( [[4,	10.597694,	-27.393173, -16.234924],
      [5,	7.43364,	    -6.514775, -1.6161577],
      [7,	10.218439,	19.506702, -2.3712635]])
# cl2 = cl1

setup_006 = np.array([[6,	9.201336,-11.038518, 33.249813],
                [7,	10.263556,-8.002243, -28.115576],
                [8,	7.324513,-6.460612, -3.1290224]])

#%%
group_data = []
for sid, setup in zip([4,5,6],[setup_004, setup_005, setup_006]):
    for i, tank in enumerate(setup):
        group_data.append({
            'setup': sid,
            'tank_id': int(tank[0]),
            'group': i
        })
group_df = pd.DataFrame(group_data)

#%% find potential matching sets

cl1_cb = list(itertools.combinations(setup_006, 3))
cl2_cb = list(itertools.combinations(setup_004, 3))
index_pair = []; al_RMS = []
for ia,A in enumerate(cl1_cb):
    A = np.array(A)
    A = np.transpose(A[:,-2:])
    A = np.concatenate([A,np.zeros((1,A.shape[1]))]) 
    
    for ib,B in enumerate(cl2_cb):
        B = np.array(B)
        B = np.transpose(B[:,-2:])
        B = np.concatenate([B,np.zeros((1,B.shape[1]))])
        
        RMS, perm_id = pair_matching(A,B)
        
        al_RMS.append(RMS)
        index_pair.append([ia,ib])
        print(min(RMS))

ibest = np.argmin(np.array([min(e) for e in al_RMS]))
cb    = index_pair[ibest]
cbest1 = np.array(cl1_cb[cb[0]])
cbest2 = np.array(cl2_cb[cb[1]])

#%% find the best order of 2 sets
RMS, perm_id = pair_matching(
    trans_2D_to_3D(cbest1[:,-2:]),
    trans_2D_to_3D(cbest2[:,-2:])
)

#%%
iRMS_best = np.argmin(np.array(RMS))
perm_best = perm_id[iRMS_best]
cbest1 = cbest1[perm_best,:]

#%%

best_R, best_t = rigid_transform_3D(
    trans_2D_to_3D(cbest1[:,-2:]),
    trans_2D_to_3D(cbest2[:,-2:])
)
print(best_R)
print(best_t)
#%%
setup_1 = 6
setup_2 = 4
for tank_1, tank_2 in zip(cbest1,cbest2):
    i1 = (group_df.setup == setup_1) & (group_df.tank_id == tank_1[0])
    i2 = (group_df.setup == setup_2) & (group_df.tank_id == tank_2[0])
    group_df.loc[i1, 'group'] = group_df.loc[i2, 'group'].values[0]
#%%
# perm_id = perms(range(A.shape[0]))
# RMS  = [];
# for perm in perm_id:
#     aa = (A[:,perm])
# # Recover R and t
#     ret_R, ret_t = rigid_transform_3D(aa, B)
    
#     # Compare the recovered R and t with the original
    
#     B2 = (ret_R@aa) + ret_t
    
#     # Find the root mean squared error
#     RMS.append(rms(B2,B)) 
    
#     print(rms(B2,B))
    
#     plt.scatter(B2[0,:],B2[1,:]);
#     plt.scatter(B[0,:],B[1,:])
#     plt.show()
# vec = range(0,3);
# print(perms(vec))