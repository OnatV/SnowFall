import taichi as ti
import numpy as np

from snow_config import SnowConfig
from fluid_grid import FluidGrid
@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SnowConfig, GGUI = True):
        self.cfg = config
        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_size = self.cfg.domain_size
        self.num_particles = self.cfg.num_particles
        self.domain_end = self.domain_start + self.domain_size
        self.particle_radius = self.cfg.particle_radius 
        self.boundary_particle_radius = 0.1 # move to config
        self.dim = 3 # 3D simulation
        self.gravity = ti.Vector(self.cfg.gravity)
        self.temperature = -10.0 # degrees Celsuis
        self.m_k = 0.0001 * (self.particle_radius ** self.dim) # particle mass
        self.smoothing_radius = self.cfg.smoothing_radius_ratio * self.particle_radius
        self.boundary_smoothing_radius = self.boundary_particle_radius * 4.0
        self.wind_direction = ti.Vector(self.cfg.wind_direction)
        self.enable_wind = True
        self.initialize_type = self.cfg.initialize_type
        self.grid_spacing = self.smoothing_radius * 2
        # self.grid_size= self.cfg.grid_size
        # self.num_grid_cells = int(self.cfg.grid_size ** 3)
        self.max_particles_per_cell = self.cfg.grid_max_particles_per_cell

        # allocate memory
        self.allocate_fields()
        self.initialize_fields()
        print("Creating Grid")
        self.update_grid()
        # init once as long as boundaries are static
        self.update_boundary_grid()
        print("Built grid")
        print(self.position[0])
        # for visualization
        self.window = ti.ui.Window("Snowfall", (800,800), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(-self.domain_size[0] * 2, self.domain_size[1] / 2.0, self.domain_size[2] / 2.0)
        self.camera.up(0.0, 1.0, 0.0)
        self.camera.lookat(self.domain_size[0] / 2.0, self.domain_size[1] / 2.0, self.domain_size[2] / 2.0)
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
        # self.pressure_old = ti.field(float, shape=self.num_particles)
        
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

        self.fluid_grid = FluidGrid(self.domain_start, self.domain_end, self.smoothing_radius)

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
        self.bgrid_x = int(self.domain_size[0] / self.boundary_particle_radius)
        self.bgrid_z = int(self.domain_size[2] / self.boundary_particle_radius)
        self.nBoundaryParticles = self.bgrid_x * self.bgrid_z
        self.boundary_particles = ti.Vector.field(self.dim, float,  shape=self.nBoundaryParticles)
        self.boundary_particles_volume = ti.Vector.field(self.dim, float,  shape=self.nBoundaryParticles)
        # will be updated only when boundary moves
        self.boundary_grid = ti.field(dtype=int, shape=(self.grid_size, self.max_particles_per_cell))   ##Holds the indices of partices at grid points
        self.boundary_grid_num_particles = ti.field(dtype=int, shape=(self.grid_size))  ##Holds the number of particles at each grid point
        
        
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
        self.colors = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)

    def initialize_particle_block(self):
        block_length = int(np.floor(np.cbrt(self.num_particles)))
        print("Block length", block_length)
        grid_origin = ((self.domain_end - self.domain_start) / 2.0) - ((self.particle_radius * block_length) / 2.0)
        for i in range(self.num_particles):
            self.position[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(block_length):
            for j in range(block_length):
                for k in range(block_length):
                    x = i * (self.particle_radius - 3e-3) + grid_origin[0]
                    y = j * (self.particle_radius - 3e-3) + grid_origin[1]
                    z = k * (self.particle_radius - 3e-3) + grid_origin[2]
                    self.position[i * block_length * block_length + j * block_length + k] = ti.Vector([x, y, z])


    def initialize_random_particles(self):
        for i in range(self.num_particles):
                x = (self.domain_size[0] - 1) * ti.random(dtype=float) + 0.5
                y = (self.domain_size[1] - 1)  * ti.random(dtype=float) + 0.5
                z = (self.domain_size[2] - 1)  * ti.random(dtype=float) + 0.5
                self.position[i] = ti.Vector([x, y, z])

    def initialize_fields(self):
        print("initializing particle positions...")
        if self.initialize_type == 'block':
            self.initialize_particle_block()
        else:
            self.initialize_random_particles
        self.fluid_grid.update_grid(self.position)
        for i in range(self.num_particles):
            self.pressure[i] = 0.0
            self.colors[i] = ti.Vector([1.0, 1.0, 1.0])
        boundary_plane_num_z_dir = self.bgrid_z
        boundary_plane_num_x_dir = self.bgrid_x        
        for i in range(self.nBoundaryParticles):
            x = float(i // boundary_plane_num_z_dir) / (boundary_plane_num_x_dir - 1) * self.domain_size[0]
            y = 0
            z = float(i % boundary_plane_num_z_dir) / (boundary_plane_num_z_dir - 1) * self.domain_size[2]
            self.boundary_particles[i] = ti.Vector([x, y, z])
        print("Intialized!")


    def update_grid(self):
        # self.update_grid1()
        # self.update_grid2()
        self.fluid_grid.update_grid(self.position)
        print("Done with update")
    
    # @ti.kernel
    # def update_grid1(self):
    #     ##First remove all particles from grid
    #     for i in range(self.grid_size):
    #         self.grid_num_particles[i] = 0
    #     # print("Done with update")
    
    # @ti.kernel
    # def update_grid2(self):
    #     for i in range(self.num_particles):   
    #         grid_idx = self.to_grid_idx(i)
    #         if self.grid_num_particles[grid_idx] >= self.max_particles_per_cell:
    #             continue
    #         self.grid[grid_idx, self.grid_num_particles[grid_idx]] = i
    #         self.grid_num_particles[grid_idx] += 1
    #         self.particle_to_grid[i] = grid_idx

    @ti.func
    def set_neighbor_color(self, i, j, color):
        self.colors[i] = color
        self.colors[j] = color

    @ti.kernel
    def color_neighbors(self, i:int, color:ti.template()):
        self.for_all_neighbors(ti.Vector([i]), self.set_neighbor_color, color)
    
    @ti.kernel
    def update_boundary_grid(self):
        for b in range(self.nBoundaryParticles):
            self.boundary_grid_num_particles[b] = 0
        for b in range(self.nBoundaryParticles):
            grid_idx = self.to_grid_b_idx(b)
            if self.boundary_grid_num_particles[grid_idx] >= self.max_particles_per_cell:
                continue
            self.boundary_grid[grid_idx, self.boundary_grid_num_particles[grid_idx]] = b
            self.boundary_grid_num_particles[grid_idx] += 1
            #self.particle_to_grid[b] = grid_idx

    # this function converts a particle to a grid index
    @ti.func
    def to_grid_b_idx(self, i):
        '''
            @TODO: This function assumes grid_size is larger 
        '''
        p = self.boundary_particles[i]
        x = ti.math.floor(p.x / self.grid_spacing)
        y = ti.math.floor(p.y / self.grid_spacing)
        z = ti.math.floor(p.z / self.grid_spacing)

        return self.convert_grid_ix(x,y,z)

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

    @ti.func
    def for_all_neighbors(self, i, func : ti.template(), ret : ti.template()):
        # pos = self.position[i]
        self.fluid_grid.for_all_neighbors(i, self.position, func, ret, self.smoothing_radius)
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
    #         if self.grid_size - 1 == min(self.grid_size - 1, current_grid) : continue
    #         for j in range(self.grid_num_particles[current_grid]):
    #             p_j = self.grid[current_grid, j] # Get point idx
    #             if i[0] != p_j and (self.position[i] - self.position[p_j]).norm() < self.smoothing_radius:
    #                 func(i, p_j, ret)
    @ti.func
    def for_all_b_neighbors(self, i, func : ti.template(), ret : ti.template()):
        '''
            boundary version of function above
            Only iterates over 1 neighbours of grid cell i to find the points in the neighbourhood..
            A slow function because:, 
                -can't be parallelized.
                -if checks are not static.
        '''
        grid_idx = self.to_grid_b_idx(i)
        ###Iterate over all neighbours of grid cell i
        for x,y,z in ti.ndrange((-1,2),(-1,2),(-1,2)):
            current_grid = grid_idx + self.convert_grid_ix(x,y,z)
            if max(0, current_grid) == 0 : continue
            if self.grid_size - 1 == min(self.grid_size - 1, current_grid) : continue
            for j in range(self.boundary_grid_num_particles[current_grid]):
                p_j = self.boundary_grid[current_grid, j] # Get point idx
                if i[0] != p_j and (self.position[i] - self.boundary_particles[p_j]).norm() < self.smoothing_radius:
                    func(i, p_j, ret)

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
        self.scene.particles(self.position, color = (0.99, 0.99, 0.99), radius = 0.5 * self.particle_radius, per_vertex_color=self.colors)
        self.scene.particles(self.boundary_particles, color = (0.4, 0.4, 0.4),
                              radius = 0.5 * self.boundary_particle_radius)


