import taichi as ti
import numpy as np
import configparser

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
        self.boundary_particle_radius = 0.1 * self.particle_radius # move to config
        self.dim = 3 # 3D simulation
        self.init_density = self.cfg.init_density
        self.gravity = ti.Vector(self.cfg.gravity)
        self.temperature = -10.0 # degrees Celsuis
        self.m_k = np.pi * float(4/3) * self.particle_radius ** self.dim * self.init_density # particle mass
        # self.m_k = 0.008
        self.smoothing_radius = self.cfg.smoothing_radius_ratio * self.particle_radius
        # self.boundary_smoothing_radius = self.boundary_particle_radius * 4.0
        self.wind_direction = ti.Vector(self.cfg.wind_direction)
        self.enable_wind = False
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
        self.update_boundary_grid()
        # init once as long as boundaries are static
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

        self.density = ti.field(float, shape=self.num_particles)
        self.p_star = ti.field(float, shape=self.num_particles)
        self.rest_density = ti.field(float, shape=self.num_particles)
        self.avg_rest_density = ti.field(float, shape=1)
        self.position = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.position_0 = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure = ti.field(float, shape=self.num_particles)
        
        self.pressure_gradient = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure_laplacian = ti.field(float, shape=self.num_particles)
        self.deformation_gradient = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.pseudo_correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.is_pseudo_L_i = ti.field(dtype=bool, shape=self.num_particles)
        self.lambda_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters
        self.G_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters
        self.jacobian_diagonal = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.density_error = ti.field(float, shape=self.num_particles)

        self.friction_diagonal = ti.Vector.field(1, dtype=float, shape=self.num_particles)

        self.fluid_grid = FluidGrid(self.domain_start, self.domain_end, self.smoothing_radius)
        self.b_grid = FluidGrid(self.domain_start, self.domain_end, self.smoothing_radius)


        self.padding = 0.1 * self.grid_spacing
        # self.boundary_particle_spacing = self.boundary_particle_radius # important quantity for figuring out boundary volume
        # boundary particles

        self.num_b_particles = 2000
        self.boundary_particles = ti.Vector.field(self.dim, dtype=float,  shape=self.num_b_particles)
        self.boundary_velocity = ti.Vector.field(self.dim, dtype=float, shape=self.num_b_particles)

        self.boundary_particles_volume = ti.field(float,  shape=self.num_b_particles)
        self.boundary_colors = ti.Vector.field(self.dim, dtype=float, shape=self.num_b_particles)
        
        
        self.velocity_star = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles) #@@

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

    # @ti.kernel
    def initialize_particle_block(self, len_x:float, len_y:float, len_z:float, origin:ti.template(), positions:ti.template(), particle_radius:ti.template()):
        # print("Block length", block_length)
        # block_position = origin
        # positions = ti.Vector.field(3, dtype=float, shape=int(len_x / self.particle_radius) * int(len_z / self.particle_radius) * int(len_y / self.particle_radius))
        for i in range(self.num_particles):
            positions[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(int(len_x / particle_radius)):
            for j in range(int(len_z / particle_radius)):
                for k in range(int(len_y / particle_radius)):
                    x = i * (particle_radius) + origin[0]
                    y = j * (particle_radius) + origin[2]
                    z = k * (particle_radius) + origin[1]
                    positions[int(k * (len_x / particle_radius) * (len_z / particle_radius) + j * (len_x / particle_radius) + i)] = ti.Vector([x, z, y])

    

    @ti.kernel
    def initialize_random_particles(self):
        print("making random particles")
        for i in range(self.num_particles):
            x = (self.domain_size[0] - self.domain_size[0] * 0.1) * ti.random(dtype=float) + self.domain_size[0] * 0.1
            y = (self.domain_size[1] - self.domain_size[1] * 0.1)  * ti.random(dtype=float) + self.domain_size[1] * 0.1
            z = (self.domain_size[2] - self.domain_size[2] * 0.1)  * ti.random(dtype=float) + self.domain_size[2] * 0.1
            self.position[i] = ti.Vector([x, y, z])

    # simple helper to initialize deformation gradient as the identity
    @ti.kernel
    def gradient_initialize(self):
        # mat = ti.Matrix(m=3, n=3, dtype=float)
        for i in range(self.num_particles):
            self.deformation_gradient[i] = ti.Matrix.identity(float, 3)

    @ti.kernel
    def boundary_initialize(self):
        '''
            NOT USED
        '''
        boundary_plane_num_z_dir = self.bgrid_z
        boundary_plane_num_x_dir = self.bgrid_x        
        for i in range(int(self.num_b_particles)):
            x = float(i // boundary_plane_num_z_dir) / (boundary_plane_num_x_dir - 1) * (self.domain_size[0] /2) + self.domain_size[0] * 0.25
            y = float(0.35)
            z = float(i % boundary_plane_num_z_dir) / (boundary_plane_num_z_dir - 1) * (self.domain_size[2]/2) + self.domain_size[2] * 0.25
            self.boundary_particles[i] = ti.Vector([x, y, z])

    @ti.kernel
    def boundary_velocity_initialize(self):
        for i in range(self.num_b_particles):
            self.boundary_velocity[i] = ti.Vector([0.0, 0.0, 0.0])

    def initialize_fields(self):
        print("initializing particle positions...")
        if self.initialize_type == 'block':
            block_origin = ti.field(float, 3)
            block_origin.from_numpy(self.cfg.block_origin)
            self.initialize_particle_block(self.cfg.block_length, self.cfg.block_height, self.cfg.block_width, block_origin, self.position, self.particle_radius)
        else:
            self.initialize_random_particles()
        self.fluid_grid.update_grid(self.position)
        # initialize starting quantities
        for i in range(self.num_particles):
            self.pressure[i] = 0.0
            self.colors[i] = ti.Vector([1.0, 1.0, 1.0])
            
        # init boundary colors
        for i in range(self.num_b_particles):
            self.boundary_colors[i] = ti.Vector([1.0, 1.0, 1.0])


        self.initialize_particle_block(0.5, 0.06, 0.5, ti.Vector([0.25, 0.25, 0.25]), self.boundary_particles, self.boundary_particle_radius)
        self.b_grid.update_grid(self.boundary_particles)
        self.boundary_velocity_initialize()
        self.gradient_initialize()
        print("Intialized!")


    def update_grid(self):
        for i in range(self.num_particles):
            self.colors[i] = ti.Vector([1.0, 1.0, 1.0])
        for i in range(self.num_b_particles):
            self.boundary_colors[i] = ti.Vector([0.4, 0.4, 0.4])
        self.fluid_grid.update_grid(self.position)

    @ti.func
    def set_neighbor_color(self, i, j, color):
        self.colors[j] = color

    @ti.func
    def set_b_neighbor_color(self, i, j, color):
        self.boundary_colors[j] = color

    @ti.kernel
    def color_neighbors(self, i:int, color:ti.template()):
        self.colors[i] = ti.Vector([0.0, 1.0, 0.0])
        self.for_all_neighbors(ti.Vector([i]), self.set_neighbor_color, color)

    @ti.kernel
    def color_b_neighbors(self, i:int, color:ti.template()):
        self.for_all_b_neighbors(ti.Vector([i]), self.set_b_neighbor_color, color)
    

    def update_boundary_grid(self):
        self.b_grid.update_grid(self.boundary_particles)


    @ti.func
    def for_all_neighbors(self, i, func : ti.template(), ret : ti.template()):
        # pos = self.position[i]
        self.fluid_grid.for_all_neighbors(i, self.position, func, ret, self.smoothing_radius)

    @ti.func
    def for_all_neighbors_b_grid(self, i, func : ti.template(), ret : ti.template()):
        '''
            Boundary neighbors of boundary particle i.
        '''
        pos = self.boundary_particles[i]
        self.b_grid.for_all_b_neighbors(i, pos, self.boundary_particles, func, ret, self.smoothing_radius)

    @ti.func
    def for_all_b_neighbors(self, i, func : ti.template(), ret : ti.template()):
        '''
            Boundary neighbors of fluid particle i.
        '''
        position = self.position[i]
        self.b_grid.for_all_b_neighbors(i, position, self.boundary_particles, func, ret, self.smoothing_radius)

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
        self.scene.particles(self.boundary_particles, per_vertex_color=self.boundary_colors,
                              radius = self.boundary_particle_radius)


