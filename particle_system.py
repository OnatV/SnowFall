import taichi as ti
import numpy as np

from snow_config import SnowConfig
# from fluid_grid import FluidGrid
@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SnowConfig, GGUI = True):
        self.cfg = config
        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_size = self.cfg.domain_size
        self.num_particles = self.cfg.num_particles
        self.domain_end = self.domain_start + self.domain_size
        self.particle_radius = 0.01 # move to config
        self.dim = 3 # 3D simulation
        self.gravity = ti.Vector(self.cfg.gravity)
        self.temperature = -10.0 # degrees Celsuis
        self.m_k = 0.0001 * (self.particle_radius ** self.dim) # particle mass
        self.smoothing_radius = self.particle_radius * 4.0
        self.wind_direction = ti.Vector(self.cfg.wind_direction)

        self.enable_wind = config.enable_wind

        self.grid_spacing = self.smoothing_radius * 2
        # self.grid_size= self.cfg.grid_size
        # self.num_grid_cells = int(self.cfg.grid_size ** 3)
        self.max_particles_per_cell = self.cfg.grid_max_particles_per_cell

        # allocate memory
        self.allocate_fields()
        self.initialize_fields()
        print("Creating Grid")
        self.update_grid()
        print("Built grid")
        print(self.position[0])
        # for visualization
        self.window = ti.ui.Window("Snowfall", (800,800), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(-7.5, 2.5, 2.5)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.lookat(2.5, 2.5, 2.5)
        self.initalize_domain_viz()

    def allocate_fields(self):
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.velocity = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.density = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.p_star = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.rest_density = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.position = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.position_0 = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure = ti.field(float, shape=self.num_particles)
        self.pressure_old = ti.field(float, shape=self.num_particles)
        self.pressure_gradient = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure_laplacian = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.deformation_gradient = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.pseudo_correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.is_pseudo_L_i = ti.field(dtype=bool, shape=self.num_particles)
        self.lambda_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters
        self.G_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters
        self.jacobian_diagonal = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.density_error = ti.Vector.field(1, dtype=float, shape=self.num_particles)

        self.grid_size_x = int((self.domain_end[0] - self.domain_start[0]) / self.grid_spacing)
        self.grid_size_y = int((self.domain_end[1] - self.domain_start[1]) / self.grid_spacing)
        self.grid_size_z = int((self.domain_end[2] - self.domain_start[2]) / self.grid_spacing)
        self.grid_size = self.grid_size_x * self.grid_size_y * self.grid_size_z
        self.grid = ti.field(dtype=int, shape=(self.grid_size, self.max_particles_per_cell))   ##Holds the indices of partices at grid points
        self.grid_new = ti.field(dtype=int, shape=(self.grid_size, self.max_particles_per_cell))   ##Holds the indices of partices at grid points
        self.grid_num_particles = ti.field(dtype=int, shape=(self.grid_size))  ##Holds the number of particles at each grid point
        self.particle_to_grid = ti.field(dtype=int, shape=self.num_particles)        ##Holds the grid point index of each particle, currently not needed because we
        self.padding = self.grid_spacing

        # boundary particles
        self.boundary_particles = ti.Vector.field(self.dim, shape=self.num_particles)

        self.velocity_star = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles) #@@
        # #2nd grid:
        # ## allocate memory for the grid
        # self.grid_particles_num = ti.field(int, shape=(self.grid_size_x * self.grid_size_y * self.grid_size_z))
        # self.grid_particles_num_swap = ti.field(int, shape=(self.grid_size_x * self.grid_size_y * self.grid_size_z))
        # self.grid_ids = ti.field(int, shape=self.num_particles)
        # self.grid_ids_swap = ti.field(int, shape=self.num_particles)
        # self.grid_ids_new = ti.field(int, shape=self.num_particles)
        # # allocate swaps for sorting
        # self.position_swap = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.position_0_swap = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.velocity_swap = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.acceleration_swap  = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.density_swap = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        # self.pressure_swap = ti.field(float, shape=self.num_particles)

        # cumsum for particle grid
        # self.cumsum = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.theta_clamp_c = ti.Matrix([
            [1 - self.cfg.theta_c, 0, 0],
            [0, 1 - self.cfg.theta_c, 0],
            [0, 0, 1 - self.cfg.theta_c]
        ])
        self.theta_clamp_s =ti.Matrix([
            [1 + self.cfg.theta_s, 0, 0],
            [0, 1 + self.cfg.theta_s, 0],
            [0, 0, 1 + self.cfg.theta_s]
        ])

    @ti.kernel
    def initialize_fields(self):
        print("initilizing particle positions...")
        for i in range(self.num_particles):
            # set it so that it fills up most of the default domain size, should be done more cleverly in the future
            #x = (self.domain_size[0] - 1) * ti.random(dtype=float) + 0.5
            #y = (self.domain_size[1] - 1)  * ti.random(dtype=float) + 0.5
            z = (self.domain_size[2] - 1)  * ti.random(dtype=float) + 0.5
            # put all on a line -> to test 1d case
            x = 2.5
            y = 2.5
            self.position[i] = ti.Vector([x, y, z])
        print("Intialized!")

    # def cumsum_indx(self):
    #     np_arr = np.cumsum(self.grid_particles_num.to_numpy())
    #     self.grid_particles_num.from_numpy(np_arr)

    # @ti.func
    # def pos_to_index(self, pos):
    #     return (pos / self.grid_spacing).cast(int)

    # @ti.func
    # def flatten_grid_index(self, grid_index):
    #     return grid_index[0] * self.grid_size_y * self.grid_size_z + grid_index[1] * self.grid_size_z + grid_index[2]
    
    # @ti.func
    # def get_flattened_grid_index(self, pos):
    #     return self.flatten_grid_index(self.pos_to_index(pos))

    # @ti.kernel
    # def sort_particles(self):
    #     for i in range(self.num_particles):
    #         idx = self.num_particles - 1 - i
    #         offset = 0
    #         if self.grid_ids[idx] - 1 >= 0:
    #             offset = self.grid_particles_num[self.grid_ids[idx] - 1]
    #         self.grid_ids_new[idx] = ti.atomic_sub(self.grid_particles_num_swap[self.grid_ids[idx]], 1) - 1 + offset
    #     # copy data into swaps, with reordering
    #     for i in ti.grouped(self.grid_ids):
    #         idx = self.grid_ids_new[i]
    #         self.grid_ids_swap[idx] = self.grid_ids[i]
    #         self.position_0_swap[idx] = self.position_0[i]
    #         self.position_swap[idx] = self.position[i]
    #         self.velocity_swap[idx] = self.velocity[i]
    #         self.acceleration_swap[idx] = self.acceleration[i]
    #         self.density_swap[idx] = self.density[i]
    #         self.pressure_swap[idx] = self.pressure[i]
    #     # repopulate original fields
    #     for i in ti.grouped(self.grid_ids):
    #         self.grid_ids[i] = self.grid_ids_swap[i]
    #         self.position_0[i] = self.position_0_swap[i]
    #         self.position[i] = self.position_swap[i]
    #         self.velocity[i] = self.velocity_swap[i]
    #         self.acceleration[i] = self.acceleration_swap[i]
    #         self.density[i] = self.density_swap[i]
    #         self.pressure[i] = self.pressure_swap[i]
    # @ti.kernel
    # def update_grid(self):
    #     for i in ti.grouped(self.grid_particles_num):
    #         self.grid_particles_num[i] = 0
    #     for i in ti.grouped(self.position):
    #         grid_index = self.get_flattened_grid_index(self.position[i])
    #         self.grid_ids[i] = grid_index
    #         ti.atomic_add(self.grid_particles_num[grid_index], 1)
    #     for i in ti.grouped(self.grid_particles_num):
    #         self.grid_particles_num_swap[i] = self.grid_particles_num[i]


    def update_grid(self):
        ##First remove all particles from grid
        self.update_grid1()
        self.update_grid2()
        print("Done with update")
    
    @ti.kernel
    def update_grid1(self):
        ##First remove all particles from grid
        for i in range(self.grid_size):
            self.grid_num_particles[i] = 0
        # print("Done with update")
    
    @ti.kernel
    def update_grid2(self):
        ##First remove all particles from grid
        for i in range(self.num_particles):   
            grid_idx = self.to_grid_idx(i)
            if self.grid_num_particles[grid_idx] >= self.max_particles_per_cell:
                continue
            self.grid[grid_idx, self.grid_num_particles[grid_idx]] = i
            self.grid_num_particles[grid_idx] += 1
            self.particle_to_grid[i] = grid_idx


    # this function converts a particle to a grid index
    @ti.func
    def to_grid_idx(self, i):
        '''
            @TODO: This function assumes grid_size is larger 
        '''
        p = self.position[i]
        x = ti.math.floor(p.x / self.grid_spacing)
        y = ti.math.floor(p.y / self.grid_spacing)
        z = ti.math.floor(p.z / self.grid_spacing)

        return self.convert_grid_ix(x,y,z)
    
    @ti.func
    def convert_grid_ix(self, x,y,z):

        ##This ensures the indices are always correct, 
        # but out of bounds indices overflow
        x_ = ti.math.clamp(ti.math.mod(x , self.grid_size_x), 0 , self.grid_size_x - 1)
        y_ = ti.math.clamp(ti.math.mod(y , self.grid_size_y), 0 , self.grid_size_y - 1)
        z_ = ti.math.clamp(ti.math.mod(z , self.grid_size_z), 0 , self.grid_size_z - 1)
        idx = ti.cast((x_ + y_ * self.grid_size_y + z_ * self.grid_size_y * self.grid_size_z), int)
        return ti.math.clamp(idx, 0, self.grid_size - 1)

    # takes a grid index and converts it to a 3D index
    # this probably can and should be precomputed
    # @ti.func
    # def grid_ind_to_ijk(self, n):
    #     i = n % self.grid_size_x
    #     j = (i / self.grid_size_x) % self.grid_size_x
    #     k = i / (self.grid_size_x * self.grid_size_x)
    #     return (i, j, k)
    
    @ti.func
    def for_all_neighbors(self, i, func : ti.template(), ret : ti.template()):
        '''
            Only iterates over 1 neighbours of grid cell i to find the points in the neighbourhood..
            A slow function because:, 
                -can't be parallelized.
                -if checks are not static.
        '''
        grid_idx = self.to_grid_idx(i)
        ###Iterate over all neighbours of grid cell i
        for x,y,z in ti.ndrange((-1,2),(-1,2),(-1,2)):
            current_grid = grid_idx + self.convert_grid_ix(x,y,z)
            if max(0, current_grid) == 0 : continue
            if self.grid_size - 1 == min(self.grid_size - 1, current_grid) : continue
            for j in range(self.grid_num_particles[current_grid]):
                p_j = self.grid[current_grid, j] # Get point idx
                if i[0] != p_j and (self.position[i] - self.position[p_j]).norm() < self.smoothing_radius:
                    func(i, p_j, ret)
    
    # def sum_all_negihbours(self, i, func : ti.template(), ret : ti.template()):
    #     '''
    #         Only iterates over 1 neighbours of grid cell i to find the points in the neighbourhood..
    #         A slow function because:, 
    #             -can't be parallelized.
    #             -if checks are not static.
    #     '''
    #     grid_idx = self.to_grid_idx(i)
    #     ###Iterate over all neighbours of grid cell i
    #     for x,y,z in ti.ndrange((-1,2),(-1,2),(-1,2)):
    #         current_grid = grid_idx + self.convert_grid_ix(x,y,z)
    #         if max(0, current_grid) == 0 : continue
    #         # if self.num_grid_cells - 1 == min(self.num_grid_cells - 1, current_grid) : continue
    #         if current_grid < 0: continue
    #         if current_grid >= self.num_grid_cells: continue
    #         for j in range(self.grid_num_particles[current_grid]):
    #             p_j = self.grid[current_grid, j] # Get point idx
    #             if p_j > self.num_particles or p_j < 0: continue
    #             if i != j and (self.position[i] - self.position[p_j]).norm() < self.smoothing_radius:
    #                 func(i, p_j, ret)

    # 2nd grid:
    # @ti.func
    # def for_all_neighbors(self, i, func: ti.template(), retval: ti.template()):
    #     center = self.pos_to_index(self.position[i])
    #     for cell_offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
    #         grid_index = self.flatten_grid_index(center + cell_offset)
    #         for j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
    #             if i[0] != j and (self.position[i] - self.position[j]).norm() < self.smoothing_radius:
    #                 func(i, j, retval)

    # @ti.func
    # def find_neighbors_as_list(self, i):
    #     current_particle_grid_idx = self.particle_to_grid[i]
    #     ret = ti.Vector.field(1, shape=self.max_particles_per_cell)



    def visualize(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.draw_domain()
        self.draw_particles()
        self.canvas.scene(self.scene)
        self.window.show()

    def initalize_domain_viz(self):
        self.vertices = ti.Vector.field(self.dim, dtype=float, shape=8)
        self.vertices[0].xyz = self.domain_start[0], self.domain_start[1], self.domain_start[2]
        self.vertices[1].xyz = self.domain_start[0], self.domain_start[1], self.domain_end[2]
        self.vertices[2].xyz = self.domain_end[0], self.domain_start[1], self.domain_end[2]
        self.vertices[3].xyz = self.domain_end[0], self.domain_start[1], self.domain_start[2]

        self.vertices[4].xyz = self.domain_start[0], self.domain_end[1], self.domain_start[2]
        self.vertices[5].xyz = self.domain_start[0], self.domain_end[1], self.domain_end[2]
        self.vertices[6].xyz = self.domain_end[0], self.domain_end[1], self.domain_end[2]
        self.vertices[7].xyz = self.domain_end[0], self.domain_end[1], self.domain_start[2]
        
        self.indices = ti.Vector.field(2, dtype=int, shape=12)
        self.indices[0].xy = 0, 1
        self.indices[1].xy = 1, 2
        self.indices[2].xy = 2, 3
        self.indices[3].xy = 3, 0        

        self.indices[4].xy = 4, 5
        self.indices[5].xy = 5, 6
        self.indices[6].xy = 6, 7
        self.indices[7].xy = 7, 4

        self.indices[8].xy = 0, 4
        self.indices[9].xy = 1, 5
        self.indices[10].xy = 2, 6
        self.indices[11].xy = 3, 7

    def draw_domain(self):
        self.scene.lines(vertices=self.vertices, indices=self.indices, width=1.0)

    def draw_particles(self):
        self.scene.particles(self.position, color = (0.99, 0.0, 0.99), radius = self.smoothing_radius)


