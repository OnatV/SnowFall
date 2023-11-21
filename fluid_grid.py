# import taichi as ti
# import numpy as np

# from snow_config import SnowConfig

# @ti.data_oriented
# class FluidGrid:
#     def __init__(self,
#         domain_start, 
#         domain_end,
#         grid_spacing, 
#         max_particles_per_cell,
#         num_particles
#     ):
#         self.domain_start = domain_start
#         self.domain_end = domain_end
#         self.grid_spacing = grid_spacing
#         self.max_particles_per_cell = max_particles_per_cell
#         self.grid_size_x = int((domain_end[0] - domain_start[0]) / self.grid_spacing)
#         self.grid_size_y = int((domain_end[1] - domain_start[1]) / self.grid_spacing)
#         self.grid_size_z = int((domain_end[2] - domain_start[2]) / self.grid_spacing)
#         self.num_particles = num_particles
#         ## allocate memory for the grid
#         self.grid_id = ti.Vector.field(n=self.max_particles_per_cell, shape=(self.grid_size_x * self.grid_size_y * self.grid_size_z), dtype=int)
#         self.grid_particles_num = ti.field(int, shape=int(self.grid_size_x * self.grid_size_y * self.grid_size_z))

#     @ti.func
#     def pos_to_index(self, pos):
#         return (pos / self.grid_spacing).cast(int)

#     @ti.func
#     def flatten_grid_index(self, grid_index):
#         return grid_index[0] * self.grid_size_y * self.grid_size_z + grid_index[1] * self.grid_size_z + grid_index[2]
    
#     @ti.func
#     def get_flattened_grid_index(self, pos):
#         return self.flatten_grid_index(self.pos_to_index(pos))

#     @ti.kernel
#     def update_grid(self, positions: ti.template()):
#         for i in ti.grouped(self.grid):
#             self.grid[i] = 0
#         for i in ti.grouped(positions):
#             grid_index = self.get_flattened_grid_index(positions[i])
#             self.grid[i] = grid_index
#             ti.atomic_add(self.grid_particles_num[grid_index], 1)
#         # for i in ti.grouped(self.grid_particles_num):
#         #     self.grid_particles_num


