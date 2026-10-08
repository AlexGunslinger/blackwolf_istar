# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 23:36:06 2017

@author: LinhTruongHong
"""

import numpy as np
import numpy.matlib 
from numpy import linalg as LA
#from scipy import spatial
from sklearn.neighbors import KDTree
# Computing a normal vector

def compute3DCentroid(points):
	centroid = np.mean(points,axis=0)
	return centroid

def computeCovarianceMatrix(centroid,points):
	pp_mean = np.asmatrix(points - centroid)
	CovarianceMatrix = (pp_mean.T*pp_mean)/len(points)
	return CovarianceMatrix

def normal_vector(points,option):
	
	points = np.array(points, dtype = np.float64, copy = True)
	
	if points.shape[0] <= 3:
		outVector =  np.array([np.inf, np.inf, np.inf]).T
	else:
		centroid = np.mean(points,axis=0)
		pp_mean = np.asmatrix(points - centroid)
		covariance_matrix = (pp_mean.T*pp_mean)/len(points)

		w, v = LA.eig(covariance_matrix)
		ids = np.argsort(w)
		w = w[ids]
		v = v[:,ids]
		w = np.array(w, dtype = np.float64, copy = True)
		v = np.array(v, dtype = np.float64, copy = True)  
		if option == 1:
			# The largest tangent vector
			outVector = v[:,2]
		else:
			# Normal vector
			outVector = v[:,0]
			# To orient a normal towards the viewpoint, which is opposite to the original
			viewVector = centroid - [0,0,0]
			if np.sum(outVector.T*viewVector) < 0:
				outVector = - outVector
	return np.array(outVector.T)
#%%
def computing_normal(data, pt_ids, searching_option = 'kNN', value = 50):
# =============================================================================
# 	data = ]x, y, z]
# 	pts_ids = point id to be computed the normal
#	data = src_pts.copy()
#	pt_ids = src_match_pt_ids.copy()
#	searching_option = 'rangesearch'
#	value = 0.2
#	data = src_all_pts.copy()
#	pt_ids = src_pt_ids.copy()
#	searching_option = searching_method
#	value = search_value
# =============================================================================
	data = np.array(data, dtype = np.float64, copy = True)
	pt_ids = np.array(pt_ids, dtype = np.int64, copy = True)
	
	if searching_option == 'kNN':
		searching_value = np.int(value)
	elif searching_option == 'rangesearch':
		searching_value = value
		del value
	else:
		searching_option = 'kNN'
		searching_value = 20

	# Find neighbour points
	node_leaf_no_pts = 50
	Tree = KDTree(data, leaf_size = node_leaf_no_pts, metric='euclidean')
	if searching_option == 'kNN':
		nn_pt_ids = Tree.query(data[pt_ids], k = searching_value + 1, return_distance= False, sort_results = True)
	else:
		nn_pt_ids = Tree.query_radius(data[pt_ids], r = searching_value, count_only= False)
	
	# Computing normal vector
	pt_normal = np.zeros([len(pt_ids),3])
	for i in range(len(nn_pt_ids)):
		nn_pt_id = nn_pt_ids[i]

		if len(nn_pt_id) < 3: #not enough to compute the normal vector
			pt_normal[i] = np.array([np.inf, np.inf, np.inf])
		else:
			# Compute a normal vector
			nn_pts_coord = data[nn_pt_id]
			pt_normal[i] = normal_vector(nn_pts_coord, 0)
			
	# Remove sigularity normal vector
	mask = np.isfinite(pt_normal[:,0])
	pt_normal = pt_normal[mask]
	pt_ids = pt_ids[mask]
	
	# Return value
	return (pt_ids, pt_normal)
#%%
def moving_points(affine_mat, points):
	# Demo
#	affine_mat = pose_matrix.copy()
#	points = orig_src_pts.copy()

	affine_mat = np.array(affine_mat, dtype = np.float64, copy= True)
	points = np.array(points, dtype = np.float64, copy= True)
	
	transform_pts = points.copy()
	transform_pts[:,0] = points[:,0]*affine_mat[0,0] + points[:,1]*affine_mat[0,1] + points[:,2]*affine_mat[0,2] + affine_mat[0,3]
	transform_pts[:,1] = points[:,0]*affine_mat[1,0] + points[:,1]*affine_mat[1,1] + points[:,2]*affine_mat[1,2] + affine_mat[1,3]
	transform_pts[:,2] = points[:,0]*affine_mat[2,0] + points[:,1]*affine_mat[2,1] + points[:,2]*affine_mat[2,2] + affine_mat[2,3]
	
	if transform_pts.shape[1] > 4:
		transform_pts[:,3] = points[:,3]*affine_mat[0,0] + points[:,4]*affine_mat[0,1] + points[:,5]*affine_mat[0,2]
		transform_pts[:,4] = points[:,3]*affine_mat[1,0] + points[:,4]*affine_mat[1,1] + points[:,5]*affine_mat[1,2]
		transform_pts[:,5] = points[:,3]*affine_mat[2,0] + points[:,4]*affine_mat[2,1] + points[:,5]*affine_mat[2,2]
	
	return transform_pts

#%% Estimate covariance
def estimate_cov(data, normal):
	#% Demo
#	data = rand_pts_coord.copy()
#	normal = rand_pts_normal.copy()
	#%
	data = np.array(data, dtype = np.float64, copy = True)
	normal = np.array(normal, dtype = np.float64, copy = True)
	
	A1 = np.cross(data, normal)
	A2 = normal
	A = np.column_stack((A1,A2))
	A = np.asmatrix(A)
	C = A.T*A
	
	return np.array(C)

#%% Sampling stable ptc
def stable_sampling_pt(pts_coord, pts_normal, num_sampling_pts):
	#% Demo
#	pts_coord = src_pts[src_match_pt_ids]
#	pts_normal = src_pts_normal.copy()
#	num_sampling_pts = num_samples
	#% Initialize data
	pts_coord = np.array(pts_coord, dtype = np.float64, copy = True)
	pts_normal = np.array(pts_normal, dtype = np.float64, copy = True)
	num_sampling_pts = np.array(num_sampling_pts, dtype = np.int64, copy = True)

	#
	num_pts = pts_coord.shape[0]
	# Pre-processing as in Hartley Zissermann (map to unit sphere):
	# shift to origin
	mean_pts_coord = np.mean(pts_coord, axis = 0)
	pts_coord = pts_coord - mean_pts_coord
	
	# compute average dist from origin
	avg_dist_pts = np.sum(np.sqrt(np.sum(pts_coord*pts_coord, axis = 1)))
	scale = num_pts/avg_dist_pts
	pts_coord = pts_coord*scale
	
	# Figure out how bad the situation would be if we just used random sampling. To do this, we estimate what C would be using just a small uniform sampling of points.
	num_random_sample = np.int16(np.max((num_sampling_pts/10, 10)))
	
	ids = np.uint64(np.floor(np.random.uniform(0,1,num_pts)*num_pts))
	ids = ids[range(num_random_sample)]
	rand_pts_coord = pts_coord[ids]
	rand_pts_normal = pts_normal[ids]
	
	covariance_matrix = estimate_cov(rand_pts_coord, rand_pts_normal)

	# Find the eigenvector and eigenvalues cov matrix
	w, v = LA.eig(covariance_matrix) 
	sort_id = np.argsort(w)
	v_sort = v[sort_id]

	# Step B1: form v
	v = np.column_stack((np.cross(pts_coord, pts_normal), pts_normal))
	
	# Step B2: form Lk
	L = dict()
	link_id = dict()
	
	for k in range(6):
		xk = numpy.matlib.repmat(v_sort[:,k], num_pts, 1)
		magnitude_xk_v = np.sum(v*xk, axis = 1)
		magnitude_xk_v = magnitude_xk_v**2
		sort_id_xk_v = np.argsort(magnitude_xk_v)
		L[k] = v[sort_id_xk_v]
		link_id[k] = sort_id_xk_v

	# Step B3 and B4 : Constrain eigenvectors and choose the points 
	total_vi = np.zeros([6,1])
	count = np.ones([6,1])
	sample_ids = []
	for i in range(num_pts):
		
		# Choose the next point from sorted list that has the smallest total
		ind_min = np.argmin(total_vi)
		cL = L[ind_min]
		vi = cL[np.int64(count[ind_min])]
		sample_ids = np.append(sample_ids, link_id[ind_min][np.int64(count[ind_min])])
		count[ind_min] = count[ind_min] + 1
		
		# Update total
		
		vi = numpy.matlib.repmat(vi.T, 1, 6)
		err = np.sum(vi*v_sort, axis = 0)
		err_mag = err*err
		total_vi = total_vi + np.reshape(err_mag,(6,1))
		kappa = total_vi[0]/total_vi[5]
		
		# if we sampled enough points and the condition number is satisfactory (around 1)
		# Note that kappa constraint is usually satisfied and we end up with desired number of points
		
		if (i >= num_sampling_pts) & ((kappa > 0.95) & (kappa < 1.15)):
			num_sampling_pts = 1
			break
		
	# Really sample them.
	sample_ids = sample_ids[range(num_sampling_pts)]
	sample_ids = np.reshape(np.int64(sample_ids),(len(sample_ids)))
	sampling_pts_coord = pts_coord[sample_ids]
	sampling_pts_normal = pts_normal[sample_ids]
	
	return (sampling_pts_coord, sampling_pts_normal)
#%% Uniform sampling data
def uniform_sampling_pt(pts_coord, pts_normal, num_sampling_pts):
	#% Demo
#	pts_coord = src_pts[src_match_pt_ids]
#	pts_normal = src_pts_normal.copy()
#	num_sampling_pts = num_samples
	#% Initialize data
	pts_coord = np.array(pts_coord, dtype = np.float64, copy = True)
	pts_normal = np.array(pts_normal, dtype = np.float64, copy = True)
	num_sampling_pts = np.array(num_sampling_pts, dtype = np.int64, copy = True)
	#
	num_pts = pts_coord.shape[0]
	# Really sample them.
	sampled_ind = range(0, num_pts, np.int(np.floor(num_pts/num_sampling_pts)))
	sampled_ind = np.reshape(sampled_ind,(len(sampled_ind)))
	
	sampling_pts_coord = pts_coord[sampled_ind]
	sampling_pts_normal = pts_normal[sampled_ind]
	
	return (sampling_pts_coord, sampling_pts_normal)
#%% Get a duplicate
def get_duplicate(data):
	import collections
	#% Demo
#	data = new_j.copy()
	#% Initialize
	data = np.array(data, dtype = np.int64, copy = True)
	data = np.reshape(data,(-1,))
	#%
	duplicates = [item for item, count in collections.Counter(data).items() if count > 1]
	return np.reshape(duplicates,(len(duplicates)))
#%% 
def minimize_point_to_plane(src_pts, dst_pts, dst_pts_normal):
	# Minimize the point to plane metric according to
	# Kok Lim Low : Linear Least Squares Optimization for Point-to-Plane ICP Surface Registration
	# Also check Effcient Variants of the ICP Algorithm by Szymon Rusinkiewicz
	#% Demo
#	src_pts = match_src_pts_coord.copy()
#	dst_pts = match_dst_pts_coord.copy()
#	dst_pts_normal = match_dst_pts_normal.copy()
	#% Initialize
	src_pts = np.array(src_pts, dtype = np.float64, copy = True)
	dst_pts = np.array(dst_pts, dtype = np.float64, copy = True)
	dst_pts_normal = np.array(dst_pts_normal, dtype = np.float64, copy = True)
	
	b = np.sum((dst_pts-src_pts)*dst_pts_normal, axis = 1)
	mat_01 = np.cross(src_pts, dst_pts_normal)
	a = np.column_stack((mat_01, dst_pts_normal))
	x = LA.lstsq(np.asmatrix(a), np.asmatrix(b).T)
	
	return np.array(x[0])
#%% get transform matrix
def get_transform_mat(data):
#	data = x.copy()
	#% Initialise
	data = np.array(data, dtype = np.float64, copy = True)
	r = data[0:3]
	r_cos = np.cos(r)
	r_sin = np.sin(r)
	t = data[3:6]
	
	r_x = np.zeros([3,3], dtype = np.float64)
	r_x[0,0] = 1
	r_x[1,1] = r_cos[0]
	r_x[1,2] = -r_sin[0]
	r_x[2,1] = r_sin[0]
	r_x[2,2] = r_cos[0]
	r_x = np.asmatrix(r_x)
	
	r_y = np.zeros([3,3], dtype = np.float64)
	r_y[0,0] = r_cos[1]
	r_y[0,2] = r_sin[1]
	r_y[1,1] = 1
	r_y[2,0] = -r_sin[1]
	r_y[2,2] = r_cos[1]
	r_y = np.asmatrix(r_y)
	#
	r_z = np.zeros([3,3], dtype = np.float64)
	r_z[0,0] = r_cos[2]
	r_z[0,1] = -r_sin[2]
	r_z[1,0] = r_sin[2]
	r_z[1,1] = r_cos[2]
	r_z[2,2] = 1
	r_z = np.asmatrix(r_z)
	#
	rot_xyz = r_x*r_y*r_z
	rot_xyz = np.array(rot_xyz)
	transform_matrix = np.column_stack((rot_xyz, t))
	
	return transform_matrix
	
#%%
def iterative_closest_point (src_all_pts, src_pt_ids, des_all_pts, des_pt_ids, icp_tol = 0.0001, icp_max_iterations = 100, searching_method = 'rangesearch', search_value = 0.2, use_robust_rejection = 0, outlier_scale = 3, num_pyr = 8, sampling_type = 0):
	#%% Demo
#	src_all_pts = PRE_PTS.copy()
#	src_pt_ids = PRE_PTS_Tunel_face_ids.copy()
#	des_all_pts = POST_PTS.copy()
#	des_pt_ids = POST_PTS_Tunel_face_ids.copy()
#	#% Assigne variables. Leave as default if not provided
#	icp_tol = 0.0001 #Tolerence
#	icp_max_iterations = 100 #MaxIterations
#	searching_method = 'rangesearch'
#	search_value = 0.2
#	use_robust_rejection = 1 #RejectionScale
#	outlier_scale = 3 #RejectionScale
#	num_pyr = 8 # NumLevels
#	sampling_type = 0 # A method for sampling
	#%% Initialize data
	src_all_pts = np.array(src_all_pts, dtype = np.float64, copy = True)
	src_pt_ids = np.array(src_pt_ids, dtype = np.int64, copy = True)
	des_all_pts = np.array(des_all_pts, dtype = np.float64, copy = True)
	des_pt_ids = np.array(des_pt_ids, dtype = np.int64, copy = True)

	# Constants
	num_kNN = 1 # numNeighbors  = 1: Allow for N multiple assignments = NumNeighborsCorr
	node_leaf_no_pts = 50 # number of points on leaf node of kd-tree
	#%% 
	# Computing normal vectors for the candidate matching points
	# Source data -> pre
	src_pt_ids, src_pt_normal = computing_normal(src_all_pts, src_pt_ids, searching_option = searching_method, value = search_value)
	src_pts = src_all_pts[src_pt_ids]
	# Destination data -> Post
	des_pt_ids, des_pt_normal = computing_normal(des_all_pts, des_pt_ids, searching_option = searching_method, value = search_value)
	des_pts = des_all_pts[des_pt_ids]
	#%%
	num_src_pts = src_pts.shape[0]
	# Hartley-Zissermann Scaling:
	mean_src_pts_coord = np.mean(src_pts, axis = 0)
	mean_des_pts_coord = np.mean(des_pts, axis = 0)
	dist_src_des_centroid = (mean_src_pts_coord + mean_des_pts_coord)*0.5
	src_pts = src_pts - dist_src_des_centroid
	des_pts = des_pts - dist_src_des_centroid

	# Compute average distance from an origin
	src_pts_dist = np.sum(np.sqrt(np.sum(src_pts*src_pts, axis = 1)))
	des_pts_dist = np.sum(np.sqrt(np.sum(des_pts*des_pts, axis = 1)))
	avg_dist_src_des = (src_pts_dist + des_pts_dist)/2
	
	# scale to unit sphere
	scale_src_des = num_src_pts / avg_dist_src_des
	src_pts = src_pts*scale_src_des
	des_pts = des_pts*scale_src_des
	
	# Store original source (src) and destination (des) data
	orig_src_pts = src_pts.copy()
	orig_src_pt_normal = src_pt_normal.copy()
	orig_des_pts = des_pts.copy()
	orig_des_pt_normal = des_pt_normal.copy()
	
	# Construct the KD-tree out of destination points (scene)
	des_pts = orig_des_pts.copy()
	des_pt_normal = orig_des_pt_normal.copy()
#	des_kdtree = spatial.KDTree(des_pts, node_leaf_no_pts) # Spatial library
	des_kdtree = KDTree(des_pts, leaf_size = node_leaf_no_pts, metric='euclidean')
	
	# Initial pose matrix
	affine_matrix = np.column_stack((np.eye(3, dtype = np.float64), np.zeros([3,1], dtype = np.float64)))
#%%	
	# Start the registration
	for level in range(num_pyr-1,-1,-1):
		print('level:', level)
		# Obtain the parameters in this pyramid level
		pry_num_sampling_pts = np.uint32(num_src_pts/np.power(2,level))
		pry_icp_tol = icp_tol*np.power(level+1,2)
		pry_icp_max_iterations = np.uint32(icp_max_iterations/np.power(2,level))
		
		# Obtain the sampled point clouds for this level
		affine_matrix_rot = affine_matrix
		affine_matrix_rot[:,3] = 0
		src_pts = moving_points(affine_matrix, orig_src_pts)
		src_pt_normal = moving_points(affine_matrix_rot, orig_src_pt_normal)
		
		# Define a sampling data 
		if sampling_type == 1:
			src_pts, src_pt_normal = stable_sampling_pt(src_pts, src_pt_normal, pry_num_sampling_pts)
		else:
			src_pts, src_pt_normal = uniform_sampling_pt(src_pts, src_pt_normal, pry_num_sampling_pts)
		
		# Initialise parameters for a main loop of ICP
		# Array whith contains the transformed points
		moved_src_pts = np.array(src_pts, dtype = np.float64, copy = True)
		# Distance error in the last iteration
		dist_err_iter = np.inf
		# Difference of distance errors between two consecutive iteration
		diff_dist_err_2_iter = 0
		# The minimum residual with a affine transform
		fval_min = np.inf
		# Current an affine transform
		cur_affine_transform = np.column_stack((np.eye(3, dtype = np.float64), np.zeros([3,1], dtype = np.float64)))
		# Iterations
		i = 0
		
		while (not((diff_dist_err_2_iter < (1 + pry_icp_tol)) & (diff_dist_err_2_iter>(1 - pry_icp_tol)))) & (i < pry_icp_max_iterations):
			print('Iteration:', i)
			# Calculate the closest point for all points
#			match_pt = des_kdtree.query(moved_src_pts, num_kNN) # Spatial library
#			match_pt_dist = match_pt[0]
#			match_pt_id = match_pt[1]
			
			match_pt_dist, match_pt_id = des_kdtree.query(moved_src_pts, k = num_kNN, return_distance= True, sort_results = True)
			
			# Implement Picky ICP or BC-ICP
			new_i = np.array(range(len(match_pt_id)), dtype = np.int64)
			new_i = np.reshape(new_i, (-1,1))
			new_j = np.array(match_pt_id, dtype = np.int64, copy = True)
		
			# Step 1 of ICP : Robustly reject outliers
			
			if use_robust_rejection > 0:
				median_match_pt_dist = np.median(match_pt_dist) # Check distribution -> mean or median
				sigma_match_pt_dist = 1.4826*median_match_pt_dist # Robust estimate of stddev
				threshold_match_pt_dist = outlier_scale*sigma_match_pt_dist + median_match_pt_dist
				mask = match_pt_dist <= threshold_match_pt_dist
				new_j = new_j[mask]
				new_i = new_i[mask]
				match_pt_dist = match_pt_dist[mask]

			# Step 2 of Picky ICP:
			# Among the resulting corresponding pairs, if more than one scene point p_i is assigned to the same model point m_j, then select p_i that corresponds to the minimum distance.
			duplicate_ids = get_duplicate(new_j)
		
			for j in range(len(duplicate_ids)):
#				print(j)
				dup_id = duplicate_ids[j]
				if not dup_id:
					continue
				# Searching inds in new_j
				mask = new_j == dup_id
				mask_id = np.where(mask==True)[0]
				dup_dist = match_pt_dist[mask]
				if dup_dist.size:
					continue
				# Update new_i and new_j
				dup_dist_min_id = np.argmin(dup_dist)
				temp_new_i = new_i[mask_id[dup_dist_min_id]]
				new_j[mask_id] = np.nan
				new_i[mask_id] = np.nan
				new_j[mask_id[0]] = dup_id
				new_i[mask_id[0]] = temp_new_i
		
			# Determine a pair of points
			map_src_pt_ids = new_i
			map_des_pt_ids = new_j
			map_src_pt_ids = map_src_pt_ids[~ np.isnan(map_src_pt_ids)]
			map_des_pt_ids = map_des_pt_ids[~ np.isnan(map_des_pt_ids)]
			match_src_pts = src_pts[map_src_pt_ids]
			match_des_pts = des_pts[map_des_pt_ids]
			match_des_pt_normal = des_pt_normal[map_des_pt_ids]
			
			# Map match_src_pts onto des_pts as points to surface
			mapping_vector = minimize_point_to_plane(match_src_pts, match_des_pts, match_des_pt_normal)
			
			# Get a tranform matrix
			cur_affine_transform = get_transform_mat(mapping_vector)
			# Map src_pts onto des_pts
			moved_src_pts = moving_points(cur_affine_transform, src_pts)
			
			# Compute errors after transform
			fval = np.sum((moved_src_pts - src_pts)**2, axis = 1)
			fval = np.sqrt(np.sum(fval)/len(fval))
			# Calculate change in error between two iterations
			diff_dist_err_2_iter = fval/dist_err_iter
			
			print('Error:', fval)
			print('Error in 2:', diff_dist_err_2_iter)
			# Update current errors
			dist_err_iter = fval
			if fval < fval_min:
				fval_min = fval #Min pose
			
			del match_src_pts 
			del match_des_pts
			del match_des_pt_normal
			
			# Update iteration number
			i += 1
			# End of ICP for the pyramid level
		
		# Compute an update affine matrix for next pyramid level
		affine_matrix = np.asmatrix(np.concatenate((cur_affine_transform, np.array([[0, 0, 0, 1]])), axis = 0))*np.asmatrix(np.concatenate((affine_matrix, np.array([[0, 0, 0, 1]])), axis = 0))
		affine_matrix = affine_matrix[:-1,:]
		
		# End of the pyramid
	
	# Compute a final affine transform
	translate_para = affine_matrix[0:3, 3]/scale_src_des + np.matrix(dist_src_des_centroid).T -affine_matrix[0:3, 0:3]*np.asmatrix(dist_src_des_centroid).T
	affine_matrix[0:3, 3] = translate_para
	
	return affine_matrix