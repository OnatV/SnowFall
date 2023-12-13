import taichi as ti
import numpy as np
from snow_config import SnowConfig
vec3 = ti.types.vector(3, float)

@ti.data_oriented
class FluidGrid:
    def __init__(self,
        grid_start, 
        grid_end,
        grid_spacing
    ):
        self.grid_start = grid_start
        self.grid_end = grid_end
        self.grid_spacing = grid_spacing
        self.grid_size_x = int((grid_end[0] - grid_start[0]) / self.grid_spacing) + 2
        self.grid_size_y = int((grid_end[1] - grid_start[1]) / self.grid_spacing) + 2
        self.grid_size_z = int((grid_end[2] - grid_start[2]) / self.grid_spacing) + 2
        self.num_cells = self.grid_size_x * self.grid_size_y * self.grid_size_z
        ## create the grid
        print(f"Creating a grid with dimension {self.grid_size_x}, {self.grid_size_y}, {self.grid_size_z}")
        # self.grid_cells = ti.field(ti.root.dynamic(ti.i, 1024, chunk_size=32), shape=(self.grid_size_x * self.grid_size_y * self.grid_size_z))
        self.handle = ti.root.dense(ti.i, (self.num_cells) ).dynamic(ti.j, 4000, chunk_size=8)
        self.grid = ti.field(int)
        self.handle.place(self.grid)

    @ti.func
    def to_grid_idx(self, p:ti.template()):
        '''
        Takes a position in 3D space and maps it to 3D grid indices
        '''
        x = int(ti.math.floor(p.x / self.grid_spacing)) + 1
        y = int(ti.math.floor(p.y / self.grid_spacing)) + 1
        z = int(ti.math.floor(p.z / self.grid_spacing)) + 1

        return (x,y,z)

    @ti.func
    def grid_to_array_index(self, x, y, z):
        '''
        This takes a 3D grid index and maps it to a 1D array index 
        '''
        indx = z * self.grid_size_x * self.grid_size_y + y * self.grid_size_x + x

        return indx

    @ti.kernel
    def update_grid(self, particles:ti.template()):
        for i in range(self.num_cells):
            self.grid[i].deactivate()
        for i in ti.grouped(particles):
            (x, y, z) = self.to_grid_idx(particles[i])
            indx = self.grid_to_array_index(x, y, z)
            indx = ti.math.min(self.num_cells - 1, indx) # guard against particles that are outside the grid
            indx = ti.math.max(0, indx)
            self.grid[indx].append(i)


    @ti.func
    def for_all_neighbors_vec3(self, i, positions: ti.template(), func : ti.template(), ret : vec3, h):
        '''
            param: i index of particle
            param: pos position of particle i
            param: func function to be evaluated
            param: ret return value
        '''
        grid_idx = self.to_grid_idx(positions[i])
        ###Iterate over all neighbours of grid cell i
        ti.loop_config(serialize=True)
        for g in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            current_grid = (
                ti.math.clamp(grid_idx[0] + g[0], 0, self.grid_size_x),
                ti.math.clamp(grid_idx[1] + g[1], 0, self.grid_size_y),
                ti.math.clamp(grid_idx[2] + g[2], 0, self.grid_size_z)
            )
            
            current_arr = self.grid_to_array_index(current_grid[0], current_grid[1], current_grid[2])
            current_arr = ti.math.min(self.num_cells - 1, current_arr) # guard against particles that are outside the grid
            current_arr = ti.math.max(0, current_arr)
            # print("h")
            ti.loop_config(serialize=True)
            for j in range(self.grid[current_arr].length()):
                p_j = self.grid[current_arr, j] # Get point idx
                if (positions[i] - positions[p_j]).norm() < h:
                    func(i, p_j, ret)

    @ti.func
    def for_all_neighbors(self, i, positions: ti.template(), func : ti.template(), ret : ti.template(), h):
        '''
            param: i index of particle
            param: pos position of particle i
            param: func function to be evaluated
            param: ret return value
        '''
        grid_idx = self.to_grid_idx(positions[i])
        ###Iterate over all neighbours of grid cell i
        ti.loop_config(serialize=True)
        for g in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            current_grid = (
                ti.math.clamp(grid_idx[0] + g[0], 0, self.grid_size_x),
                ti.math.clamp(grid_idx[1] + g[1], 0, self.grid_size_y),
                ti.math.clamp(grid_idx[2] + g[2], 0, self.grid_size_z)
            )
            
            current_arr = self.grid_to_array_index(current_grid[0], current_grid[1], current_grid[2])
            current_arr = ti.math.min(self.num_cells - 1, current_arr) # guard against particles that are outside the grid
            current_arr = ti.math.max(0, current_arr)
            # print("h")
            ti.loop_config(serialize=True)
            for j in range(self.grid[current_arr].length()):
                p_j = self.grid[current_arr, j] # Get point idx
                if (positions[i] - positions[p_j]).norm() < h:
                    func(i, p_j, ret)

    @ti.func
    def for_all_b_neighbors(self, i, position: ti.template(), b_positions: ti.template(), func : ti.template(), ret : ti.template(), h):
        '''
            to be used for computing boundary particles contributing to fluid particless
        '''
        grid_idx = self.to_grid_idx(position)
        
        ###Iterate over all neighbours of grid cell i
        for g in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            current_grid = (
                ti.math.clamp(grid_idx[0] + g[0], 0, self.grid_size_x),
                ti.math.clamp(grid_idx[1] + g[1], 0, self.grid_size_y),
                ti.math.clamp(grid_idx[2] + g[2], 0, self.grid_size_z)
            )
            current_arr = self.grid_to_array_index(current_grid[0], current_grid[1], current_grid[2])
            current_arr = ti.math.min(self.num_cells - 1, current_arr) # guard against particles that are outside the grid
            current_arr = ti.math.max(0, current_arr)
            for j in range(self.grid[current_arr].length()):
                p_j = self.grid[current_arr, j] # Get point idx
                if (position - b_positions[p_j]).norm() < h:
                    func(i, p_j, ret)
    
    @ti.func
    def for_all_b_neighbors_vec3(self, i, position: ti.template(), b_positions: ti.template(), func : ti.template(), ret : vec3, h):
        '''
            to be used for computing boundary particles contributing to fluid particless
        '''
        grid_idx = self.to_grid_idx(position)
        
        ###Iterate over all neighbours of grid cell i
        for g in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            current_grid = (
                ti.math.clamp(grid_idx[0] + g[0], 0, self.grid_size_x),
                ti.math.clamp(grid_idx[1] + g[1], 0, self.grid_size_y),
                ti.math.clamp(grid_idx[2] + g[2], 0, self.grid_size_z)
            )
            current_arr = self.grid_to_array_index(current_grid[0], current_grid[1], current_grid[2])
            current_arr = ti.math.min(self.num_cells - 1, current_arr) # guard against particles that are outside the grid
            current_arr = ti.math.max(0, current_arr)
            for j in range(self.grid[current_arr].length()):
                p_j = self.grid[current_arr, j] # Get point idx
                if (position - b_positions[p_j]).norm() < h:
                    func(i, p_j, ret)