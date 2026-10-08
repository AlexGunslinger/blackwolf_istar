# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:00:30 2021

@author: LinhTruongHong

Voxel grid
"""
import itertools
from itertools import chain
import numpy as np
class voxel_grid:
    def __init__(self, points, voxel_size, min_ptc = 10):
        """This Class is to create voxel grid
        Parameters:
        ------------------------
            points              : [Nx3] x-, y-, z-
            voxel_size          : a voxel size
            min_ptc             : a minimum number of points to classify the voxel as full or empty

        Return:
        ------------------------
            self.points         : Points
            self.size:          : voxel size
            self.min_ptc        : min_ptc
            self.ids            : [Nx, Ny, Nz]
            self.ptc_ids        : Point indices within each voxel
#            self.ptc_link       : Voxel indice for each points
            self.bounds         : [Min, Max] -
#            self.shrink_bounds  : Tight bound
            self.prop           : True = Full; False = Empty

        Demo:

        ------------------------
        """
        # Initial
        self.points = points
        self.size = voxel_size
        self.min_ptc = min_ptc
        self.ids = None               #[id_x, id_y, id_z]
        self.ptc_ids = None           #
        self.active_ptc = None
#        self.link_ptc = None
        self.bounds = None
#        self.shrink_bounds = None
        self.prop = None

    def subdivision(self):
        eps = 0.001
        # Call libs
#        import time
        # Compute a bounding box of a point cloud
        ptc_min_xyz, ptc_max_xyz = np.min(self.points, axis = 0), np.max(self.points, axis = 0)
        bounding_box_length = ptc_max_xyz - ptc_min_xyz
        bounding_box_center = (ptc_max_xyz + ptc_min_xyz)/2.

        # Computing a number of cells and adjust the new bounds
        num_voxel_xyz = np.ceil(bounding_box_length/self.size).astype(int)
        bounding_box_min = bounding_box_center - (num_voxel_xyz*self.size)/2.0 - eps
        bounding_box_max = bounding_box_center + (num_voxel_xyz*self.size)/2.0 + eps

        # Create the grid
        grids_range = []
        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(bounding_box_min[i], bounding_box_max[i], num = (num_voxel_xyz[i] + 1), retstep=True)
            grids_range.append(s)

        # Define voxel grid id
        voxel_grid_ids = list(itertools.product(np.arange(num_voxel_xyz[0]), np.arange(num_voxel_xyz[1]), np.arange(num_voxel_xyz[2])))
        voxel_grid_ids = np.asarray(voxel_grid_ids)
        self.ids = voxel_grid_ids

        # Define the link list between points and voxels: ids is voxel_ids
#        s_time = time.time()
        ptc_voxel_x = np.clip(np.searchsorted(grids_range[0], self.points[:, 0]) - 1, 0, num_voxel_xyz[0])
        ptc_voxel_y = np.clip(np.searchsorted(grids_range[1], self.points[:, 1]) - 1, 0, num_voxel_xyz[1])
        ptc_voxel_z = np.clip(np.searchsorted(grids_range[2], self.points[:, 2]) - 1, 0, num_voxel_xyz[2])
        linked_list_ptc_voxel = np.ravel_multi_index([ptc_voxel_x, ptc_voxel_y, ptc_voxel_z], num_voxel_xyz)
#        self.link_ptc = linked_list_ptc_voxel
#        print('Mapping voxels to points {:.3} seconds'.format(time.time() - s_time))

        # Define voxel bound
        voxel_grid_bound_min = list(itertools.product(grids_range[0][:-1], grids_range[1][:-1], grids_range[2][:-1]))
        voxel_grid_bound_min = np.asarray(voxel_grid_bound_min)
        voxel_grid_bound_max = list(itertools.product(grids_range[0][1:], grids_range[1][1:], grids_range[2][1:]))
        voxel_grid_bound_max = np.asarray(voxel_grid_bound_max)
        self.bounds = np.hstack((voxel_grid_bound_min, voxel_grid_bound_max))

        # Define the properties
        voxel_grid_property = np.full(voxel_grid_ids.shape[0], False, dtype = bool) #Flase = empty; True = Full
        count_points = np.bincount(linked_list_ptc_voxel)
        mask = self.min_ptc <= count_points
        full_voxel_linear_ids = np.where(mask == True)[0]
#        empty_voxel_linear_ids = np.where(mask == False)[0]
        voxel_grid_property[full_voxel_linear_ids] = True
        self.prop = voxel_grid_property

        # Get points of the voxels
        #% create a linear indices points within voxels
        ptc_ids = np.arange(self.points.shape[0])
        linked_list_ptc_voxel = linked_list_ptc_voxel
        sort_ids = np.argsort(linked_list_ptc_voxel)
        linked_list_ptc_voxel = linked_list_ptc_voxel[sort_ids]
        ptc_ids = ptc_ids[sort_ids]
        voxel_linear_ids, count_ptc_in_voxel = np.unique(linked_list_ptc_voxel, return_counts=True)
        cum_sum = np.cumsum(count_ptc_in_voxel)
        start_end_ptc_ids = np.append([0], cum_sum) #0 a linear indicies of points within the voxels

        # Assign the points
        voxel_ptc_ids = dict()

        # Preallocation
        for count in range(self.ids.shape[0]):  voxel_ptc_ids[count] = None

        # Full voxel
        for count, voxel_linear_id in enumerate(voxel_linear_ids):
            voxel_ptc_ids[voxel_linear_id] = ptc_ids[start_end_ptc_ids[count]:start_end_ptc_ids[count+1]]

        # Empty voxel
#        for empty_voxel_linear_id in empty_voxel_linear_ids: voxel_ptc_ids[empty_voxel_linear_id] = None

        # Update points in voxels
        voxel_grid_ptc_ids =  [np.asarray(ptc_ids) for __, ptc_ids in voxel_ptc_ids.items()]
        self.ptc_ids = voxel_grid_ptc_ids

        # Update active points
        self.active_ptc = np.full(self.points.shape[0], True, dtype = bool)

    # Get points within the voxel
    def get_voxelptc(self, voxel_ids, index_type='grid'):
        """This function is to retrieve the points within the given voxel
        Parameters:
        ------------------------
            self                : voxel grid
            voxel_ids            : grid - [Vx, Vy, Vz] = [1x3]; linear: [1-N]
            option              : grid or linear ide

        Return:
        ------------------------
            voxel_ptc_ids       : Linear point indices
            voxel_ptc           : [x,y,z]

        Demo:
            voxel_ids = np.array([1,2,3])
        ------------------------
        """
        # call libs
        
        if index_type.casefold() == 'grid':
            if (voxel_ids.ndim == 1) & (voxel_ids.shape[0] != 3):
                raise ValueError('Input voxel_id must be an array [1x3]')
            elif voxel_ids.ndim == 2:
                 if voxel_ids.shape[1] != 3:
                     raise ValueError('Input voxel_id must be an 2d array [Nx3]')
            elif voxel_ids.ndim > 2:
                raise ValueError('Input voxel_id must be an 2d array')
        # Get linear indices
        if index_type.casefold() == 'linear':
            local_ids = np.array(voxel_ids, dtype=int, copy=True)
        else:
            # Get linear indices
            if voxel_ids.ndim == 1:
                local_ids = np.where(np.all(self.ids == voxel_ids, axis = 1) == True)[0]
            else:
                local_ids = [np.where(np.all(self.ids == voxel_id, axis = 1) == True)[0] for voxel_id in voxel_ids]
                local_ids = np.array(local_ids).flatten()
        # Get points
        if not isinstance(local_ids, (list, tuple, np.ndarray)):
            voxel_ptc_ids = self.ptc_ids[np.asscalar(local_ids)]
        else:
            voxel_ptc_ids = list(chain.from_iterable(self.ptc_ids[ind] for ind in local_ids))
            voxel_ptc_ids = np.asarray(voxel_ptc_ids, dtype = np.int32)
        voxel_ptc = self.points[voxel_ptc_ids]
        # Return
        return voxel_ptc_ids, voxel_ptc

    # Search connect voxels
    def neighbour_voxels_window_search(self, voxel_id, target_voxel_ids, direction, window=1, voxel_prop="Full"):
        """
        Parameters:
        ------------------------
            self                        : voxel grid
            voxel_id                    : [Vx, Vy, Vz] = [1x3]
            target_voxel_ids            : [Vx, Vy, Vz] = [Nx3]
            direction                   : a searching direction
            window                      : window size, window = 1: adjoined voxels
            option                      : voxel type: full or empty or both

        Return:
        ------------------------
            neighbour_voxel_ids         : indices of the neighbour voxel [Vx, Vy, Vz] = [Nx3] where N <= 26 or None
            neighbour_voxel_linear_ids  : global linear indices of the neighbour voxels regarding to self
            local_ids                   : Linear indices of the neighbour voxels: local_ids <= self.ids.shape[0]
        Demo:
            voxel_id = voxel_ids
            np.array([1,2,3])
        """
        # Check input variables
        # voxel_ids
#        if not isinstance(self, type):
#            raise ValueError("this function is applied for a object class")
        # voxel_ids
        if isinstance(voxel_id, (list, np.ndarray)):
            voxel_id = np.array(voxel_id, copy = True)
        else:
            raise ValueError("Voxel id must be [Nx3] array")

        # target_voxel_ids
        if target_voxel_ids is None:
            target_voxel_ids = self.ids
            global_ids = np.arange(target_voxel_ids.shape[0])
        elif len(target_voxel_ids) == 0:
            target_voxel_ids = self.ids
            global_ids = np.arange(target_voxel_ids.shape[0])
        else:
            if target_voxel_ids.ndim == 1: target_voxel_ids = target_voxel_ids.reshape(-1, 3)
            global_ids = [np.where(np.all(self.ids == target_voxel_id, axis = 1))[0] for target_voxel_id in target_voxel_ids]
            global_ids = np.asarray(global_ids).flatten()

        # Check valid voxel_property
        voxel_req_prop = self.check_voxel_prop(voxel_prop)

        # Get the neighbour
        if direction.casefold() == 'x'.casefold():
            mask = np.all((voxel_id - window <= target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.all((target_voxel_ids[:,[1,2]] == voxel_id[[1,2]]), axis = 1)
        elif direction.casefold() == 'y'.casefold():
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.all((target_voxel_ids[:,[0,2]] == voxel_id[[0,2]]), axis = 1)
        elif direction.casefold() == 'z'.casefold():
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.all((target_voxel_ids[:,[0,1]] == voxel_id[[0,1]]), axis = 1)
        elif (direction.casefold() == 'xy'.casefold())|(direction.casefold() == 'yx'.casefold()):
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.any((target_voxel_ids[:,[0,1]] == voxel_id[[0,1]]), axis = 1) & (target_voxel_ids[:,2] == voxel_id[2])
        elif (direction.casefold() == 'yz'.casefold())|(direction.casefold() == 'zy'.casefold()):
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.any((target_voxel_ids[:,[1,2]] == voxel_id[[1,2]]), axis = 1) & (target_voxel_ids[:,0] == voxel_id[0])
        elif (direction.casefold() == 'xz'.casefold())|(direction.casefold() == 'zx'.casefold()):
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1) & np.any((target_voxel_ids[:,[0,2]] == voxel_id[[0,2]]), axis = 1) & (target_voxel_ids[:,1] == voxel_id[1])
        else:
            mask = np.all((voxel_id - window <=  target_voxel_ids)&(target_voxel_ids <= voxel_id + window), axis = 1) & np.any((target_voxel_ids != voxel_id), axis = 1)

        # Get voxel types
        neighbour_voxel_local_ids = np.where(mask == True)[0]
        if neighbour_voxel_local_ids.shape[0] > 0:
            if voxel_req_prop is not None:
                # get neighbour voxel properties
                neighbour_voxel_linear_ids = global_ids[neighbour_voxel_local_ids]
                voxel_prop = self.prop[neighbour_voxel_linear_ids]
                mask = voxel_prop == voxel_req_prop
                neighbour_voxel_local_ids = neighbour_voxel_local_ids[mask]
                neighbour_voxel_linear_ids = neighbour_voxel_linear_ids[mask]
                neighbour_voxel_ids = self.ids[neighbour_voxel_linear_ids]
            else:
                neighbour_voxel_linear_ids = None
                neighbour_voxel_ids = self.ids[neighbour_voxel_linear_ids]
        else:
            neighbour_voxel_linear_ids = None
            neighbour_voxel_ids = None

        return  neighbour_voxel_ids, neighbour_voxel_linear_ids, neighbour_voxel_local_ids
    # In[]
    def searching_adjoin_voxels(self, src_voxel_id, tg_voxel_ids, index_type="linear", searching_opt='xyz', voxel_prop="full"):
        """This function is to search neighbour voxels
        """
        # Call libs
        from .utils_geometry import inNd
        # Check source voxel ids
        src_voxel_id = self.check_src_voxels(src_voxel_id, index_type=index_type)
        # Check target voxel_ids
        tg_voxel_ids, tg_voxel_linear_ids = self.check_tg_voxels(tg_voxel_ids, index_type=index_type)
        # Check valid option
        voxel_req_prop = self.check_voxel_prop(voxel_prop)
        # Get neighbour ides
        neighbour_voxel_ids = src_voxel_id + self.base_voxel_ids(searching_opt)
        # Remove negative indices
        mask = np.any(neighbour_voxel_ids < 0, axis=1)
        neighbour_voxel_ids = neighbour_voxel_ids[~mask]
        # Find neighbour voxel_ids
        mask = inNd(tg_voxel_ids, neighbour_voxel_ids, assume_unique=False)
        # Neighbour voxels
        neighbour_voxel_linear_local_ids = np.where(mask == True)[0]
        neighbour_voxel_linear_ids, neighbour_voxel_ids = tg_voxel_linear_ids[mask], tg_voxel_ids[mask]
        # Filter voxel_type
        if (voxel_req_prop == 0) | (voxel_req_prop == 1):
            # Get the voxel type
            neighbour_voxel_type = self.prop[neighbour_voxel_linear_ids]
            mask = neighbour_voxel_type == voxel_req_prop
            neighbour_voxel_linear_local_ids = neighbour_voxel_linear_local_ids[mask]
            neighbour_voxel_linear_ids, neighbour_voxel_ids = neighbour_voxel_linear_ids[mask], neighbour_voxel_ids[mask]
        # return
        return neighbour_voxel_linear_local_ids, neighbour_voxel_linear_ids, neighbour_voxel_ids
# In[]
    def base_voxel_ids(self, searching_option):
        """This function is to build the base adjoin voxel ids
        
        """
        if not isinstance(searching_option, str):
            raise ValueError("Searching option must be a string: xyz, xyzgrids_surfaces, xy_surfaces, yz_surfaces, xz_surfaces, xgrid_surfaces, ygrid_surfaces, zgrid_surfaces")
        else:
            if searching_option.lower() in ["all"]:
                #27 voxels: all egdes and surfaces
                base_voxels = [[-1, -1, -1], [+0, -1, -1], [+1, -1, -1], [-1, +0, -1], [+0, +0, -1], [+1, +0, -1], [-1, +1, -1], [+0, +1, -1], [+1, +1, -1],
                               [-1, -1, +0], [+0, -1, +0], [+1, -1, +0], [-1, +0, +0], [+0, +0, +0], [+1, +0, +0], [-1, +1, +0], [+0, +1, +0], [+1, +1, +0],
                               [-1, -1, +1], [+0, -1, +1], [+1, -1, +1], [-1, +0, +1], [+0, +0, +1], [+1, +0, +1], [-1, +1, +1], [+0, +1, +1], [+1, +1, +1]]  
            elif searching_option.lower() in ["xyz", "yzx", "zxy"]: 
                #26 voxels: all egdes and surfaces
                base_voxels = [[-1, -1, -1], [+0, -1, -1], [+1, -1, -1], [-1, +0, -1], [+0, +0, -1], [+1, +0, -1], [-1, +1, -1], [+0, +1, -1], [+1, +1, -1],
                               [-1, -1, +0], [+0, -1, +0], [+1, -1, +0], [-1, +0, +0], [+1, +0, +0], [-1, +1, +0], [+0, +1, +0], [+1, +1, +0],
                               [-1, -1, +1], [+0, -1, +1], [+1, -1, +1], [-1, +0, +1], [+0, +0, +1], [+1, +0, +1], [-1, +1, +1], [+0, +1, +1], [+1, +1, +1]]
            elif searching_option.lower() in ["xyzgrids_surfaces", "yzxgrids_surfaces", "zxygrids_surfaces"]: 
                # 6 surfaces along the gird
                base_voxels = [[+0, +0, -1], 
                               [+0, -1, +0], [-1, +0, +0], [+1, +0, +0], [+0, +1, +0], 
                               [+0, +0, +1]]
            elif searching_option.lower() in ["xy_surfaces", "yx_surfaces"]:
                # 9 surfaces parallel to xy plane
                base_voxels = [[-1, -1, -1], [+0, -1, -1], [+1, -1, -1], [-1, +0, -1], [+0, +0, -1], [+1, +0, -1], [-1, +1, -1], [+0, +1, -1], [+1, +1, -1],
                               [-1, -1, +1], [+0, -1, +1], [+1, -1, +1], [-1, +0, +1], [+0, +0, +1], [+1, +0, +1], [-1, +1, +1], [+0, +1, +1], [+1, +1, +1]]
            elif searching_option.lower() in ["yz_surfaces", "zy_surfaces"]:
                # 9 surfaces parallel to yz plane
                base_voxels = [[-1, -1, -1], [+1, -1, -1], [-1, +0, -1], [+1, +0, -1], [-1, +1, -1], [+1, +1, -1],
                               [-1, -1, +0], [+1, -1, +0], [-1, +0, +0], [+1, +0, +0], [-1, +1, +0], [+1, +1, +0],
                               [-1, -1, +1], [+1, -1, +1], [-1, +0, +1], [+1, +0, +1], [-1, +1, +1], [+1, +1, +1]]
            elif searching_option.lower() in ["zx_surfaces", "xz_surfaces"]:
                # 9 surfaces parallel to yz plane
                base_voxels = [[-1, -1, -1], [+0, -1, -1], [+1, -1, -1], [-1, +1, -1], [+0, +1, -1], [+1, +1, -1],
                               [-1, -1, +0], [+0, -1, +0], [+1, -1, +0], [-1, +1, +0], [+0, +1, +0], [+1, +1, +0],
                               [-1, -1, +1], [+0, -1, +1], [+1, -1, +1], [-1, +1, +1], [+0, +1, +1], [+1, +1, +1]]
            elif searching_option.lower() == "xgrid_surfaces":
                # 2 surfaces parallel to x grid (normal vectors along an ox)
                base_voxels = [[-1, +0, +0], [+1, +0, +0]]
            elif searching_option.lower() == "ygrid_surfaces":
                # 2 surfaces parallel to y grid (normal vectors along an oy)
                base_voxels = [[+0, -1, +0], [+0, +1, +0]]
            elif searching_option.lower() == "zgrid_surfaces":
                # 2 surfaces parallel to z grid (normal vectors along an oz)
                base_voxels = [[+0, +0, -1], [+0, +0, +1]]
            else:
                raise ValueError("Searching type {:s} invalid.".format(searching_option) + "\n"
                                 + "Searching option must be a string: xyz, xyzgrids_surfaces, xy_surfaces, yz_surfaces, xz_surfaces, xgrid_surfaces, ygrid_surfaces, zgrid_surfaces")
        # return
        return np.asarray(base_voxels, dtype=np.int32)
                               
        
    def check_voxel_prop(self, voxel_prop):
        """This fuction is to check voxel property
        """
        if isinstance(voxel_prop, str):
            if voxel_prop.casefold() == "Full".casefold():
                voxel_prop = True
            elif voxel_prop.casefold() == "Empty".casefold():
                voxel_prop = False
            elif voxel_prop.casefold() == "Both".casefold():
                voxel_prop = None #both full and empty
            else:
                raise ValueError("Voxel type {:s} invalid".format(str(voxel_prop)))
        elif isinstance(voxel_prop, (int, float)):
            if int(voxel_prop) == 1:
                voxel_prop = True
            elif int(voxel_prop) == 0:
                voxel_prop = False
            elif int(voxel_prop) == -1:
                voxel_prop = None #both full and empty
            else:
                raise ValueError("Voxel type {:d} invalid".format(int(voxel_prop)))
        elif isinstance(voxel_prop, bool):
            voxel_prop = voxel_prop
        else:
            raise ValueError("Voxel type {} invalid".format(voxel_prop))  
        # return
        return voxel_prop
    
    def check_src_voxels(self, src_voxel_id, index_type = 'linear'):
        # Check input voxel_id
        # voxel_ids
        if np.isscalar(src_voxel_id):
            src_voxel_id = self.ids[src_voxel_id]
        elif isinstance(src_voxel_id, (list, np.ndarray)):
            src_voxel_id = np.array(src_voxel_id, copy = True)
            if index_type.casefold() == 'linear'.casefold():
                src_voxel_id = self.ids[src_voxel_id]
            else:
                if src_voxel_id.shape[0] % 3 != 0:
                    raise ValueError("Voxel id must be [Nx3] array")
        else:
            raise ValueError("Voxel id must be [Nx1] or [Nx3] array")
        # return 
        return src_voxel_id
    # In[]
    def check_tg_voxels(self, tg_voxel_ids, index_type='linear'):
        """The function is to check target voxel"
        
        """
        # target_voxel_ids
        if tg_voxel_ids is None:
            tg_voxel_ids = self.ids
            tg_voxel_linear_ids = np.arange(tg_voxel_ids.shape[0])
        elif len(tg_voxel_ids) == 0:
            tg_voxel_ids = self.ids
            tg_voxel_linear_ids = np.arange(tg_voxel_ids.shape[0])
        else:
            if index_type.casefold() == 'linear'.casefold():
                tg_voxel_linear_ids = tg_voxel_ids.flatten()
            else:
                if tg_voxel_ids.shape[0] % 3 == 0:
                    tg_voxel_linear_ids = [np.where(np.all(self.ids == tg_voxel_id, axis=1))[0] for tg_voxel_id in tg_voxel_ids]
                    tg_voxel_linear_ids = np.asarray(tg_voxel_linear_ids).flatten()
                else:
                    raise ValueError("Voxel id must be [Nx3] array")
            # Voxel grid ids
            tg_voxel_ids = self.ids[tg_voxel_linear_ids]
        # return
        return tg_voxel_ids, tg_voxel_linear_ids
#%%
class voxelgridmodify(voxel_grid):
#    def __init__(self):
#        super().__init__(None, None, None)
    def update_data_structure(self, voxel_grid, voxel_ids, index_type='linear'):
        """This function is to generate the new voxel grid with predefine voxel ids
        Parameters:
        ------------------------
            self                : new object likes voxel_grid
            voxel_grid          : a voxel grid
            voxel_ids           : [Vx, Vy, Vz] = [Nx3] or linear index [1, N]
            option              : 'Linear index' or voxel ids

        Return:
        ------------------------
            self                : new objects

        Demo:
            voxel_id = np.array([1,2,3])
        ------------------------
        """
        # Check input parameters
        if not isinstance(voxel_ids, np.ndarray):
            raise ValueError('Input voxel ids must be an array')
        else:
            voxel_ids = np.array(voxel_ids, copy = True)

        if index_type.casefold() == 'Linear_index'.casefold():
            if voxel_ids.ndim != 1:
                raise ValueError('Input voxel ids must be a linear array')
            else:
                update_type = 0

        elif index_type.casefold() == 'Voxel_index'.casefold():
            if (voxel_ids.ndim != 2)&(voxel_ids.shape[1] != 3):
                raise ValueError('Input voxel ids must be an array')
            else:
                update_type = 1
        else:
            raise ValueError('Type of input voxel ids is invalid')

        # Initiali instant
        super().__init__(None, None, None)
        self.points = voxel_grid.points
        # Extract voxel indices to be used to create a new voxel grid
        if update_type == 0:
            local_ids = voxel_ids
        else:

            mask = np.all(voxel_grid.ids == voxel_ids, axis = 1)
            local_ids = np.where(mask == True)[0]

        # Update
        self.ids = voxel_grid.ids[local_ids]
        self.size = voxel_grid.size
        self.prop = voxel_grid.prop[local_ids]
        self.ptc_ids = [np.asarray(voxel_grid.ptc_ids[local_id]) for local_id in local_ids]
        self.bounds = voxel_grid.bounds[local_ids]
        self.prop = voxel_grid.prop[local_ids]
        self.active_ptc = voxel_grid.active_ptc
#        self.link_ptc = voxel_grid.link_ptc
        return self

    def voxelgrid_extraction(self, voxel_grid, voxel_prop="full"):
        """This function is to generate the new voxel grid with predefine voxel ids
        Parameters:
        ------------------------
            self                : new object likes voxel_grid
            voxel_grid          : a voxel grid
            voxel_type          : True ror false

        Return:
        ------------------------
            self                : new objects

        Demo:
            VoxelGridModify(None, None, None).voxelgrid_extraction(voxel_grid, True)
            voxel_type = "Full"
        ------------------------
        """
        # Check input parameters
        if isinstance(voxel_prop, str):
            if not(voxel_prop.casefold() in [x.lower() for x in ["Full", "Empty"]]):
                raise ValueError("Voxel type {:s} invalid".format(str(voxel_prop)))
            elif voxel_prop.casefold() == "Empty".casefold():
                voxel_prop = False
            elif voxel_prop.casefold() == "Full".casefold():
                voxel_prop = True
        elif isinstance(voxel_prop, (int, float)):
            if int(voxel_prop) == 1:
                voxel_prop = True
            elif int(voxel_prop) == 0:
                voxel_prop = False
            else:
                raise ValueError("Voxel property {:d} invalid".format(int(voxel_prop)))
        elif isinstance(voxel_prop, bool):
            pass
        else:
            raise ValueError("Voxel property does not support")

        # Initiali instant
        super().__init__(None, None, None)
        self.points = voxel_grid.points

        # Extract voxel indices to be used to create a new voxel grid
        mask = voxel_grid.prop == voxel_prop
        local_ids = np.where(mask == True)[0]

        # Update
        self.ids = voxel_grid.ids[local_ids]
        self.size = voxel_grid.size
        self.prop = voxel_grid.prop[local_ids]
        self.ptc_ids = [np.asarray(voxel_grid.ptc_ids[local_id]) for local_id in local_ids]
        self.bounds = voxel_grid.bounds[local_ids]
        self.prop = voxel_grid.prop[local_ids]
        self.active_ptc = voxel_grid.active_ptc
#        self.link_ptc = voxel_grid.link_ptc
        return self
    # Shrink the voxel
    def voxelgrid_shrink(self, option = 'bounding_box'):
        """This function is to shrink the voxel based on either bounding box or minimum bounding box

        """
        from Geometry import cal_bounding_box_3D
        if isinstance(option, str):
            if option.casefold() == 'bounding_box'.casefold():
                # Call cal_bounding_box
                for count, ptc_ids in enumerate(self.ptc_ids):
                    if ptc_ids is not None:
                        # Get the points within the cell
                        ptc_xyz = self.points[ptc_ids]
                        new_bbox = cal_bounding_box_3D(ptc_xyz)
                        # Update the new bounds
                        self.bounds[count] = new_bbox
#            elif option.casefold() == 'min_bounding_box'.casefold():
#                # Call cal_min_bounding_box
#                a = 1
            else:
                raise ValueError('{:s} is invalid'.format(option))
        else:
            raise ValueError("Only string data type is supported")



#%%
