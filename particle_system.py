import taichi as ti
import numpy as np

from snow_config import SnowConfig

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SnowConfig, GGUI = True):
        self.cfg = config
        self.domain_origin = np.array([0.0, 0.0, 0.0])
        self.domain_size = self.cfg.domain_size
        self.num_particles = self.cfg.num_particles
        self.domain_end = self.domain_origin + self.domain_size
        self.particle_radius = 0.01 # move to config
        self.dim = 3 # 3D simulation
        self.gravity = ti.Vector(self.cfg.gravity)
        self.temperature = -10.0 # degrees Celsuis
        self.m_k = 0.08 * (self.particle_radius ** self.dim) # particle mass
        self.smoothing_radius = self.particle_radius * 4.0
        self.wind_direction = ti.Vector(self.cfg.wind_direction)

        self.grid_spacing = self.cfg.grid_spacing
        self.grid_size= self.cfg.grid_size
        self.num_grid_cells = int(self.cfg.grid_size ** 3)
        self.max_particles_per_cell = self.cfg.grid_max_particles_per_cell

        # allocate memory
        self.allocate_fields()
        self.initialize_fields()

        # for visualization
        self.window = ti.ui.Window("Snowfall", (800,800))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(-7.5, 2.5, 2.5)
        self.initalize_domain_viz()

    def allocate_fields(self):
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.velocity = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.density = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.rest_density = ti.Vector.field(1, dtype=float, shape=self.num_particles)
        self.position = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.position_0 = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.deformation_gradient = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.pseudo_correction_matrix = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_particles) # an num_particles length array of 3x3 matrices
        self.is_pseudo_L_i = ti.field(dtype=bool, shape=self.num_particles)
        self.lambda_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters
        self.G_t_i = ti.field(dtype=float, shape=self.num_particles) # Lame' parameters

        self.grid = ti.field(dtype=int, shape=(self.grid_size, self.max_particles_per_cell))   ##Holds the indices of partices at grid points
        self.grid_new = ti.field(dtype=int, shape=(self.grid_size, self.max_particles_per_cell))   ##Holds the indices of partices at grid points
        self.grid_num_particles = ti.field(dtype=int, shape=(self.grid_size))  ##Holds the number of particles at each grid point
        self.particle_to_grid = ti.field(dtype=int, shape=self.num_particles)        ##Holds the grid point index of each particle, currently not needed because we

    @ti.kernel
    def initialize_fields(self):
        print("initilizing particle positions...")
        for i in range(self.num_particles):
            # set it so that it fills up most of the default domain size, should be done more cleverly in the future
            x = 4 * ti.random(dtype=float) + 0.5
            y = 4 * ti.random(dtype=float) + 0.5
            z = 4 * ti.random(dtype=float) + 0.5
            self.position[i] = ti.Vector([x, y, z]) 

    @ti.kernel
    def update_grid(self):
        ##First remove all particles from grid
        for i in range(self.num_grid_cells):
            self.grid_num_particles[i] = 0

        for i in range(self.num_particles):   
            grid_idx = self.to_grid_idx(i)
            self.grid[grid_idx, self.grid_num_particles[grid_idx]] = i
            self.grid_num_particles[grid_idx] += 1
            self.particle_to_grid[i] = grid_idx

    @ti.func
    def to_grid_idx(self, i):
        '''
            @TODO: This function assumes grid_size is larger 
        '''

        p = self.position[i]
        x = p.x // self.grid_spacing
        y = p.y // self.grid_spacing
        z = p.z // self.grid_spacing

        return self.convert_grid_ix(x,y,z)
    
    @ti.func
    def convert_grid_ix(self, x,y,z):

        ##This ensures the indices are always correct, 
        # but out of bounds indices overflow
        x_ = x % self.grid_size
        y_ = y % self.grid_size
        z_ = z % self.grid_size

        return int(x_ + y_ * self.grid_size + z_ * self.grid_size * self.grid_size)
    
    @ti.func
    def for_all_negihbours(self, i, func):
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
            if self.num_grid_cells - 1 == min(self.num_grid_cells - 1, current_grid) : continue
            for j in range(self.grid_num_particles[current_grid]):
                func(i, self.grid[current_grid, j])


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
        self.vertices[0].xyz = self.domain_origin[0], self.domain_origin[1], self.domain_origin[2]
        self.vertices[1].xyz = self.domain_origin[0], self.domain_origin[1], self.domain_end[2]
        self.vertices[2].xyz = self.domain_end[0], self.domain_origin[1], self.domain_end[2]
        self.vertices[3].xyz = self.domain_end[0], self.domain_origin[1], self.domain_origin[2]

        self.vertices[4].xyz = self.domain_origin[0], self.domain_end[1], self.domain_origin[2]
        self.vertices[5].xyz = self.domain_origin[0], self.domain_end[1], self.domain_end[2]
        self.vertices[6].xyz = self.domain_end[0], self.domain_end[1], self.domain_end[2]
        self.vertices[7].xyz = self.domain_end[0], self.domain_end[1], self.domain_origin[2]
        
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
        self.scene.particles(self.position, color = (0.99, 0.99, 0.99), radius = self.particle_radius)


