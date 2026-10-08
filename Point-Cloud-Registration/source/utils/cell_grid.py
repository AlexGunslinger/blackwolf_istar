# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:55:45 2021

@author: LinhTruongHong
"""

import numpy as np
import itertools


class cell_grid:

    def __init__(self, points, cell_size, min_ptc=10):
        """This Class is to create cell grid
        Parameters:
        ------------------------
            points              : [Nx3] x-, y-, z-
            cell_size           : a cell size
            min_ptc             : a minimum number of points to classify the cell as full or empty
        
        Return:
        ------------------------
            self.points         : Points
            self.size:          : cell size
            self.min_ptc        : min_ptc
            self.ids            : [Nx, Ny]
            self.ptc_ids        : Point indices within each cell 
            self.bounds         : [Min, Max] - 
            self.prop           : True = Full; False = Empty
            
        Demo:
            
        ------------------------
        """
        # Initial
        self.points = points
        self.size = cell_size
        self.min_ptc = min_ptc
        self.ids = None               #[id_x, id_y]
        self.ptc_ids = None           #
        self.active_ptc = None
        self.bounds = None
        self.prop = None
        
    def subdivision(self):
        
        # Constants
        eps = 0.0001
        # Compute a bounding box of a point cloud
        ptc_min_xy, ptc_max_xy = np.min(self.points[:,:2], axis = 0), np.max(self.points[:,:2], axis = 0)
        bounding_box_length = ptc_max_xy - ptc_min_xy
        bounding_box_center = (ptc_max_xy + ptc_min_xy)/2.
        
        # Computing a number of cells and adjust the new bounds
        num_cell_xy = np.ceil(bounding_box_length/self.size).astype(int)
        bounding_box_min = bounding_box_center - (num_cell_xy*self.size)/2.0 - eps
        bounding_box_max = bounding_box_center + (num_cell_xy*self.size)/2.0 + eps
        # Create the grid
        grids_range = []
        for i in range(2):
            # note the +1 in num
            s, step = np.linspace(bounding_box_min[i], bounding_box_max[i], num = (num_cell_xy[i] + 1), retstep=True)
            grids_range.append(s)
        # Define cell grid id
        cell_grid_ids = list(itertools.product(np.arange(num_cell_xy[0]), np.arange(num_cell_xy[1])))
        cell_grid_ids = np.asarray(cell_grid_ids)
        self.ids = cell_grid_ids
        
        # Define the link list between points and cells: ids is cell_ids
#        s_time = time.time()
        ptc_cell_x = np.clip(np.searchsorted(grids_range[0], self.points[:, 0]) - 1, 0, num_cell_xy[0])
        ptc_cell_y = np.clip(np.searchsorted(grids_range[1], self.points[:, 1]) - 1, 0, num_cell_xy[1])
        linked_list_ptc_cell = np.ravel_multi_index([ptc_cell_x, ptc_cell_y], num_cell_xy)
#        self.link_ptc = linked_list_ptc_cell
#        print('Mapping cells to points {:.3} seconds'.format(time.time() - s_time))
        
        # Define cell bound
        cell_grid_bound_min = list(itertools.product(grids_range[0][:-1], grids_range[1][:-1]))
        cell_grid_bound_min = np.asarray(cell_grid_bound_min)
        cell_grid_bound_max = list(itertools.product(grids_range[0][1:], grids_range[1][1:]))
        cell_grid_bound_max = np.asarray(cell_grid_bound_max)
        self.bounds = np.hstack((cell_grid_bound_min, cell_grid_bound_max)) 
        
        # Define the properties
        cell_grid_property = np.full(cell_grid_ids.shape[0], False, dtype = bool) #Flase = empty; True = Full
        count_points = np.bincount(linked_list_ptc_cell)
        mask = self.min_ptc <= count_points
        full_cell_linear_ids = np.where(mask == True)[0]
#        empty_cell_linear_ids = np.where(mask == False)[0]
        cell_grid_property[full_cell_linear_ids] = True
        self.prop = cell_grid_property

        # Get points of the cells
        #% create a linear indices points within cells
        ptc_ids = np.arange(self.points.shape[0])
        sort_ids = np.argsort(linked_list_ptc_cell)
        linked_list_ptc_cell = linked_list_ptc_cell[sort_ids]
        ptc_ids = ptc_ids[sort_ids]
        cell_linear_ids, count_ptc_in_cell = np.unique(linked_list_ptc_cell, return_counts=True)
        cum_sum = np.cumsum(count_ptc_in_cell)
        start_end_ptc_ids = np.append([0], cum_sum) #0 a linear indicies of points within the cells
        
        # Assign the points
        cell_ptc_ids = dict()
        
        # Preallocation
        for count in range(self.ids.shape[0]):  cell_ptc_ids[count] = None 
        
        # Full cell
        for count, cell_linear_id in enumerate(cell_linear_ids):
            cell_ptc_ids[cell_linear_id] = ptc_ids[start_end_ptc_ids[count]:start_end_ptc_ids[count+1]]
        
        # Empty cell
#        for empty_cell_linear_id in empty_cell_linear_ids: cell_ptc_ids[empty_cell_linear_id] = None
        
        # Update points in cells
        cell_grid_ptc_ids =  [np.asarray(ptc_ids) for __, ptc_ids in cell_ptc_ids.items()] 
        self.ptc_ids = cell_grid_ptc_ids
        # Update active points
        self.active_ptc = np.full_like(self.prop, True, dtype = bool)


    # Get points within the cell
    def get_cellptc(self, cell_ids, index_type='grid'):
        """This function is to retrieve the points within the given cell
        Parameters:
        ------------------------
            self                : cell grid
            cell_ids            : [Vx, Vy] = [1x2]
            index_type              : 'grid' or linear
            
        Return:
        ------------------------
            cell_ptc_ids         : Linear point indices 
            cell_ptc             : [x,y,z]
            
        Demo:
            cell_ids = np.array([1,2,3])
        ------------------------
        """
        
        # Check valid cell_id
        cell_ids = np.array(cell_ids, copy = True)
        
        if index_type.casefold() == 'linear':
            cell_ids = cell_ids.flatten()
        else:
            if (cell_ids.ndim == 1) & (cell_ids.shape[0] != 3):
                raise ValueError('Input cell_id must be an array [1x3]')
            elif cell_ids.ndim == 2:
                 if cell_ids.shape[1] != 3:
                     raise ValueError('Input cell_id must be an 2d array [Nx3]')
            elif cell_ids.ndim > 2:
                raise ValueError('Input cell_id must be an 2d array')
            
        # Get linear indices
        if index_type.casefold() == 'linear':
            local_ids = cell_ids
        else:    
            if cell_ids.ndim == 1:
                local_ids = np.where(np.all(self.ids == cell_ids, axis = 1) == True)[0]
            else:
                local_ids = [np.where(np.all(self.ids == cell_ids, axis = 1) == True)[0] for cell_id in cell_ids]
            local_ids = np.array(local_ids).flatten()
        if len(local_ids) == 1:
            cell_ptc_ids = self.ptc_ids[local_ids.item()]
        else:
            cell_ptc_ids = list(itertools.chain.from_iterable(self.ptc_ids[ind] for ind in local_ids))
            cell_ptc_ids = np.asarray(cell_ptc_ids, dtype = np.int32)
        cell_ptc = self.points[cell_ptc_ids]
        # Return
        return cell_ptc_ids, cell_ptc
    
    # Search connect cells
    def neighbour_cells_window_search(self, cell_id, target_cell_ids, direction, window=1, cell_type="Full"):
        """
        Parameters:
        ------------------------
            self                        : cell grid
            cell_id                     : [Vx, Vy] = [1x2]
            target_cell_ids             : [Vx, Vy] = [Nx2]
            direction                   : a searching direction
            window                      : window size, window = 1: adjoined cells
            cell_type                   : cell type: full or empty or both
        
        Return:
        ------------------------
            neighbour_cell_ids         : indices of the neighbour cell [Vx, Vy, Vz] = [Nx3] where N <= 26 or None
            neighbour_cell_linear_ids  : global linear indices of the neighbour cells regarding to self
            local_ids                   : Linear indices of the neighbour cells: local_ids <= self.ids.shape[0] 
            
            
        
        Demo:
            cell_id = cell_ids
            
            
            np.array([1,2,3])
        """
        
        # Check input variables
        # cell_id
        if isinstance(cell_id, (list, np.ndarray)):
            cell_id = np.array(cell_id, copy=True)
        else:
            raise ValueError("cell id must be [Nx2] array")

        # target_cell_ids
        if target_cell_ids is None:
            target_cell_ids = self.ids
            global_ids = np.arange(target_cell_ids.shape[0])
        elif len(target_cell_ids) == 0:
            target_cell_ids = self.ids
            global_ids = np.arange(target_cell_ids.shape[0])
        else:
            if target_cell_ids.ndim == 1: target_cell_ids = target_cell_ids.reshape(-1, 3)
            global_ids = [np.where(np.all(self.ids == target_cell_id, axis=1))[0] for target_cell_id in target_cell_ids]
            global_ids = np.asarray(global_ids).flatten()
            
        # Check valid option
        if isinstance(cell_type, str):
            if cell_type.casefold() == "Full".casefold(): 
                cell_type = True
            elif cell_type.casefold() == "Empty".casefold(): 
                cell_type = False
            elif cell_type.casefold() == "Both".casefold(): 
                cell_type = None #both full and empty
            else:
                raise ValueError("Cell type {:s} invalid".format(str(cell_type)))
        elif isinstance(cell_type, (int, float)):
            if int(cell_type) == 1:
                cell_type = True
            elif int(cell_type) == 0: 
                cell_type = False
            elif int(cell_type) == -1: 
                cell_type = None #both full and empty
            else:
                raise ValueError("Cell type {:d} invalid".format(int(cell_type)))
        elif isinstance(cell_type, bool):
            cell_type = cell_type
        else:
            raise ValueError("Cell type {} invalid".format(cell_type))
            
        # Get the neighbour 
        if direction.casefold() == 'x'.casefold():
            mask = np.all((cell_id - window <= target_cell_ids)&(target_cell_ids <= cell_id + window), axis=1) & np.any((target_cell_ids != cell_id), axis=1) & np.all((target_cell_ids[:,1] == cell_id[1]), axis=1)
        elif direction.casefold() == 'y'.casefold():
            mask = np.all((cell_id - window <=  target_cell_ids)&(target_cell_ids <= cell_id + window), axis=1) & np.any((target_cell_ids != cell_id), axis=1) & np.all((target_cell_ids[:,0] == cell_id[0]), axis=1) 
        else:
            mask = np.all((cell_id - window <=  target_cell_ids)&(target_cell_ids <= cell_id + window), axis=1) & np.any((target_cell_ids != target_cell_ids), axis=1)
        
        # Get cell types
        neighbour_cell_local_ids = np.where(mask == True)[0]
        
        if neighbour_cell_local_ids.shape[0] > 0:
            if cell_type is not None:
                # get neighbour cell properties
                neighbour_cell_linear_ids = global_ids[neighbour_cell_local_ids]
                cell_prop = self.prop[neighbour_cell_linear_ids]
                mask = cell_prop == cell_type
                neighbour_cell_local_ids = neighbour_cell_local_ids[mask]
                neighbour_cell_linear_ids = neighbour_cell_linear_ids[mask]
                neighbour_cell_ids = self.ids[neighbour_cell_linear_ids]
            else:
                neighbour_cell_local_ids = None
                neighbour_cell_linear_ids = global_ids[neighbour_cell_local_ids]
                neighbour_cell_ids = self.ids[neighbour_cell_linear_ids]
        else:
            neighbour_cell_local_ids = None
            neighbour_cell_linear_ids = None
            neighbour_cell_ids = None        
              
        return  neighbour_cell_ids, neighbour_cell_linear_ids, neighbour_cell_local_ids  
    # In[]
    def searching_adjoin_cells(self, src_cell_id, tg_cell_ids, index_type="linear", searching_opt='xy', cell_prop="full"):
        """This function is to search neighbour cells
        Parameters:
        ------------------------
            src_cell_id            : scalar - cell_id to search its neighbour cells
            tg_cell_ids            : [1xN] - cell_ids where neighbour cell retrieved 
            index_type             : "linear" or "grid" type of cell indices
            searching_option       : what type of neighbour cells to be retrived: xyz, xyzgrids_surfaces, xy_surfaces, yz_surfaces, xz_surfaces, xgrid_surfaces, ygrid_surfaces, zgrid_surfaces"
            cell_prop              : "full" or "empty" or "both" - type of cells          
        Return:
        ------------------------
            neighbour_cell_linear_local_ids    : [1 X M] - local indices of cells respect to tg_cell_ids
            neighbour_cell_linear_ids          : [1 x M] - linear indices of neighbour cells respect to self.ids.shape[0]
            neighbour_cell_ids                 : [M x3] - grid indicies of neighbour cells respect to self.ids
        Demo:
            src_cell_id, tg_cell_ids, index_type = "linear", searching_opt=26, option = "full"
        """
        # Call libs
        from utils_geometry import inNd
        # Check source cell ids
        src_cell_id = self.check_src_cells(src_cell_id, index_type=index_type)
        # Check target cell_ids
        tg_cell_ids, tg_cell_linear_ids = self.check_tg_cells(tg_cell_ids, index_type=index_type)
        # Check valid option
        cell_prop = self.check_cell_prop(cell_prop)
        # Get neighbour ides
        neighbour_cell_ids = src_cell_id + self.base_cell_ids(searching_opt)
        # Remove negative indices
        mask = np.any(neighbour_cell_ids < 0, axis=1)
        neighbour_cell_ids = neighbour_cell_ids[~mask]
        # Find neighbour cell_ids
        mask = inNd(tg_cell_ids, neighbour_cell_ids, assume_unique=False)
        # Neighbour cells
        neighbour_cell_linear_local_ids = np.where(mask == True)[0]
        neighbour_cell_linear_ids, neighbour_cell_ids = tg_cell_linear_ids[mask], tg_cell_ids[mask]
        # Filter cell_prop
        if (cell_prop == 0) | (cell_prop == 1):
            # Get the cell type
            neighbour_cell_type = self.prop[neighbour_cell_linear_ids]
            mask = neighbour_cell_type == cell_prop
            neighbour_cell_linear_local_ids = neighbour_cell_linear_local_ids[mask]
            neighbour_cell_linear_ids, neighbour_cell_ids = neighbour_cell_linear_ids[mask], neighbour_cell_ids[mask]
        # return
        return neighbour_cell_linear_local_ids, neighbour_cell_linear_ids, neighbour_cell_ids
# In[]
    def base_cell_ids(self, searching_option):
        """This function is to build the base adjoin cell ids
        Parameters:
        ------------------------
            self                    : scalar - cell_id to search its neighbour cellss
            searching_option        : what type of neighbour cells to be retrived: xyz, xyzgrids_surfaces, xy_surfaces, yz_surfaces, xz_surfaces, xgrid_surfaces, ygrid_surfaces, zgrid_surfaces"
        Return:
        ------------------------
            base_cells             : [M x 2] - grid indicies of neighbour cells respect to self.ids
        Demo:
        """
        if not isinstance(searching_option, str):
            raise ValueError("Searching option must be a string: xyz, xyzgrids_surfaces, xy_surfaces, yz_surfaces, xz_surfaces, xgrid_surfaces, ygrid_surfaces, zgrid_surfaces")
        else:
            if searching_option.lower() in ["xy", "yx", "all"]: 
                #26 cells: all egdes and surfaces
                base_cells = [[-1, -1], [+0, -1], [+1, -1], 
                              [-1, +0], [+1, +0], 
                              [-1, +1], [+0, +1], [+1, +1]]
            elif searching_option.lower() in ["xgrid"]: 
                # 6 surfaces along the gird
                base_cells = [[+0, -1], [+0, +1]]
            elif searching_option.lower() in ["y_grid"]:
                # 9 surfaces parallel to xy plane
                base_cells = [[-1, +0], [+1, +0]]
            elif searching_option.lower() in ["xy_grid", "yx_grid"]:
                # 9 surfaces parallel to yz plane
                base_cells = [[+0, -1], [+0, +1],
                              [-1, +0], [+1, +0]]
            elif searching_option.lower() in ["yx_corner", "yx_corner"]:
                # 9 surfaces parallel to yz plane
                base_cells = [[-1, -1], [+1, -1], [-1, +1], [+1, +1]]
            else:
                raise ValueError("Searching type {:s} invalid.".format(searching_option) + "\n"
                                 + "Searching option must be a string: xy, all, xgrid, ygrid, xygrid, xy_corner")
        # return
        return np.asarray(base_cells, dtype=np.int32)
                               
    # Check cell type   
    def check_cell_prop(self, cell_prop):
        """This fuction is to check cell type
        Parameters:
        ------------------------
            self                       : cell grid (See cellgrid class)
            cell_prop                  : str or numeric: type of cell
        Return:
        ------------------------
            cell_prop                  : scalar - type of cell       
        Demo:
        """
        if isinstance(cell_prop, str):
            if cell_prop.casefold() == "Full".casefold():
                cell_prop = True
            elif cell_prop.casefold() == "Empty".casefold():
                cell_prop = False
            elif cell_prop.casefold() == "Both".casefold():
                cell_prop = None #both full and empty
            else:
                raise ValueError("cell type {:s} invalid".format(str(cell_prop)))
        elif isinstance(cell_prop, (int, float)):
            if int(cell_prop) == 1:
                cell_prop = True
            elif int(cell_prop) == 0:
                cell_prop = False
            elif int(cell_prop) == -1:
                cell_prop = None #both full and empty
            else:
                raise ValueError("cell type {:d} invalid".format(int(cell_prop)))
        elif isinstance(cell_prop, bool):
            cell_prop = cell_prop
        else:
            raise ValueError("cell type {} invalid".format(cell_prop))  
        # return
        return cell_prop
    # Check cell
    def check_src_cells(self, src_cell_id, index_type='linear'):
        """The function is to check src cell"
        Parameters:
        ------------------------
            self                        : cell grid (See cellgrid class)
            src_cell_id                : scalar - cell_ids 
            index_type                  : "linear" or "grid" type of cell indices      
        Return:
        ------------------------
            src_cell_id                : scalar - cell_ids       
        Demo:
            cellgrid, cell_linear_ids
        """
        # Check input cell_id
        # cell_ids
        if np.isscalar(src_cell_id):
            src_cell_id = self.ids[src_cell_id]
        elif isinstance(src_cell_id, (list, np.ndarray)):
            src_cell_id = np.array(src_cell_id, copy = True)
            if index_type.casefold() == 'linear'.casefold():
                src_cell_id = self.ids[src_cell_id]
            else:
                if src_cell_id.shape[0] % 3 != 0:
                    raise ValueError("cell id must be [Nx3] array")
        else:
            raise ValueError("cell id must be [Nx1] or [Nx3] array")
        # return 
        return src_cell_id
    # In[]
    def check_tg_cells(self, tg_cell_ids, index_type='linear'):
        """The function is to check target cell"
        Parameters:
        ------------------------
            self                        : cell grid (See cellgrid class)
            tg_cell_ids                : [1xN] - cell_ids where neighbour cell retrieved 
            index_type                  : "linear" or "grid" type of cell indices      
        Return:
        ------------------------
            tg_cell_ids                : [1 x M] - linear indices of neighbour cells respect to self.ids.shape[0]
            tg_cell_linear_ids         : [M x3] - grid indicies of neighbour cells respect to self.ids        
        Demo:
            cellgrid, cell_linear_ids
        """
        # target_cell_ids
        if tg_cell_ids is None:
            tg_cell_ids = self.ids
            tg_cell_linear_ids = np.arange(tg_cell_ids.shape[0])
        elif len(tg_cell_ids) == 0:
            tg_cell_ids = self.ids
            tg_cell_linear_ids = np.arange(tg_cell_ids.shape[0])
        else:
            if index_type.casefold() == 'linear'.casefold():
                tg_cell_linear_ids = tg_cell_ids.flatten()
            else:
                if tg_cell_ids.shape[0] % 3 == 0:
                    tg_cell_linear_ids = [np.where(np.all(self.ids == tg_cell_id, axis=1))[0] for tg_cell_id in tg_cell_ids]
                    tg_cell_linear_ids = np.asarray(tg_cell_linear_ids).flatten()
                else:
                    raise ValueError("cell id must be [Nx2] array")
            # cell grid ids
            tg_cell_ids = self.ids[tg_cell_linear_ids]
        # return
        return tg_cell_ids, tg_cell_linear_ids
    # In[]
    def link_list_pts_cell(self, cell_linear_ids):
        """This function is to create a link list between the points and cell
        Parameters:
        ------------------------
            cellgrid                   : cell grid (See cellgrid class)
            cell_linear_ids            : linear indices of cell       
        Return:
        ------------------------
            link_list_pts_cell         : [Nx2] [pts_ids, cell_linear_ids]         
        Demo:
            cellgrid, cell_linear_ids
        """
        link_list_pts_cell = []
        for cell_linear_id in cell_linear_ids:
            cell_pts_ids = self.ptc_ids[cell_linear_id]
            cell_pts_ids = np.vstack((cell_pts_ids, np.full(cell_pts_ids.shape[0], cell_linear_id))).T
            link_list_pts_cell = np.append(link_list_pts_cell, cell_pts_ids)
        # return
        link_list_pts_cell = np.asarray(link_list_pts_cell, dtype=np.uint32).reshape(-1, 2)
        return link_list_pts_cell
    
    def cal_cell_height(self, cell_ids, gap_threshold=0.005):
        """This function is to calculate the cell height based on the points within the cell
        Parameters:
        ------------------------
            self                : new object likes cellgrid
            cell_ids            : a voxel grid

        Return:
        ------------------------
            cell_height                : [height, max_gap]
        Demo:

        """
        if cell_ids is None:
            cells_height = np.full((self.ids.shape[0], 2), np.inf, dtype = np.float32)
            cell_ids = np.arange(self.ids.shape[0])
        else:
            cells_height = np.zeros_like(cell_ids)

        # Calculate the height
        for count, cell_id in enumerate(cell_ids):
            # Get ptc_ids
            cell_ptc_ids = self.ptc_ids[cell_id]
            cell_ptc_z = self.points[cell_ptc_ids, 2]
            block_length = cal_block_length(cell_ptc_z, gap_threshold)
            # Cal height
            cells_height[count] = [np.max(cell_ptc_z) - np.min(cell_ptc_z), np.max(block_length)]
        # Return
        return cells_height

    def cal_cell_bottom(self):
        """This function calculates the bottom of points in cells
        Parameters:
        ------------------------
            self                : new object likes cellgrid
            cell_ids            : a voxel grid

        Return:
        ------------------------
            cell_bottoms                : 
        Demo:

        """
        cell_bottoms = np.full((self.ids.shape[0], 2), np.inf, dtype = np.float32)
        cell_ids = np.arange(self.ids.shape[0])

        # Calculate the height
        for count, cell_id in enumerate(cell_ids):
            # Get ptc_ids
            cell_ptc_ids = self.ptc_ids[cell_id]
            cell_ptc_z = self.points[cell_ptc_ids, 2]
            # Cel bottom
            cell_bottoms[count] = cell_ptc_z.min()
        # Return
        return cell_bottoms

    #%%
class cell_grid_modify(cell_grid):
    #    def __init__(self):
    #        super().__init__(None, None, None)
    def update_data_structure(self, inp_cell_grid, cell_ids, index_type='linear'):
        """This function is to generate the new cell grid with predefine cell ids
        Parameters:
        ------------------------
            self                : new object likes cell_grid
            inp_cell_grid      : a cell grid
            cell_ids           : [Vx, Vy, Vz] = [Nx3] or linear index [1, N]
            index_type          : 'Linear' or 'grid'
        Return:
        ------------------------
            self                : new objects
        Demo:
            cell_id = np.array([1,2,3])
        ------------------------
        """
        # Check input parameters
        if not isinstance(cell_ids, np.ndarray):
            raise ValueError('Input cell ids must be an array')
        else:
            cell_ids = np.array(cell_ids, copy=True)

        if index_type.casefold() == 'linear'.casefold():
            if cell_ids.ndim != 1:
                raise ValueError('Input cell ids must be a linear array')
            else:
                update_type = 0
        elif index_type.casefold() == 'grid'.casefold():
            if (cell_ids.ndim != 2)&(cell_ids.shape[1] != 2):
                raise ValueError('Input cell ids must be an array')
            else:
                update_type = 1
        else:
            raise ValueError('Type of input cell ids is invalid')

        # Initiali instant
        super().__init__(None, None, None)
        self.points = inp_cell_grid.points
        # Extract cell indices to be used to create a new cell grid
        if update_type == 0:
            local_ids = cell_ids
        else:

            mask = np.all(inp_cell_grid.ids == cell_ids, axis = 1)
            local_ids = np.where(mask == True)[0]

        # Update
        self.ids = inp_cell_grid.ids[local_ids]
        self.size = inp_cell_grid.size
        self.min_ptc = inp_cell_grid.min_ptc
        self.prop = inp_cell_grid.prop[local_ids]
        self.ptc_ids = [np.asarray(inp_cell_grid.ptc_ids[local_id]) for local_id in local_ids]
        self.bounds = inp_cell_grid.bounds[local_ids]
        self.prop = inp_cell_grid.prop[local_ids]
        self.active_ptc = inp_cell_grid.active_ptc
#        self.link_ptc = cell_grid.link_ptc
        return self

    def cellgrid_extraction(self, inp_cell_grid, cell_prop="Full"):
        """This function is to generate the new cell grid with predefine cell ids
        Parameters:
        ------------------------
            self                : new object likes cell_grid
            cell_grid          : a cell grid
            cell_prop          : True ror false
        
        Return:
        ------------------------
            self                : new objects
            
        Demo:
            cellGridModify(None, None, None).cellgrid_extraction(cell_grid, True)
            cell_prop = "Full"
        ------------------------
        """
        # Check input parameters
        if isinstance(cell_prop, str):
            if not (cell_prop.casefold() in [x.lower() for x in ["Full", "Empty"]]):
                raise ValueError("cell type {:s} invalid".format(str(cell_prop)))
            elif cell_prop.casefold() == "Empty".casefold():
                cell_prop = False
            elif cell_prop.casefold() == "Full".casefold():
                cell_prop = True
        elif isinstance(cell_prop, (int, float)):
            if int(cell_prop) == 1:
                cell_prop = True
            elif int(cell_prop) == 0:
                cell_prop = False
            else:
                raise ValueError("cell type {:d} invalid".format(int(cell_prop)))
        elif isinstance(cell_prop, bool):
            pass
        else:
            raise ValueError("cell type does not support")

        # Initiali instant
        super().__init__(None, None, None)
        self.points = inp_cell_grid.points

        # Extract cell indices to be used to create a new cell grid
        mask = inp_cell_grid.prop == cell_prop
        local_ids = np.where(mask == True)[0]

        # Update
        self.ids = inp_cell_grid.ids[local_ids]
        self.size = inp_cell_grid.size
        self.min_ptc = inp_cell_grid.min_ptc
        self.prop = inp_cell_grid.prop[local_ids]
        self.ptc_ids = [np.asarray(inp_cell_grid.ptc_ids[local_id]) for local_id in local_ids]
        self.bounds = inp_cell_grid.bounds[local_ids]
        self.prop = inp_cell_grid.prop[local_ids]
        self.active_ptc = inp_cell_grid.active_ptc
        #        self.link_ptc = cell_grid.link_ptc
        return self

        
def cal_block_length(pts_val, gap_threshold):
    """This function is to calculate length of the block
    Parameters:
    ------------------------
        pts_val                    : values of points
        gap_threshold              : gap_threshold

    Return:
    ------------------------
        block_length               : [1xN]
    Demo:
    """
    
    # Find a gap
    pts_val = np.sort(pts_val)
    mask = np.diff(pts_val) > gap_threshold
    gap_ids = np.where(mask == True)[0]

    # Calculate block length
    if gap_ids.shape[0] == 0:
        block_length = pts_val[-1] - pts_val[0]
    else:
        # Set the ids
        s_ids = np.append(0, gap_ids + 1)
        s_ids = np.unique(s_ids)
        e_ids = np.append(gap_ids, pts_val.shape[0] - 1)
        e_ids = np.unique(e_ids)
        # Cal block length
        block_length = pts_val[e_ids] - pts_val[s_ids]
    # return
    return block_length        

