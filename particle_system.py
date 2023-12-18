import taichi as ti
import numpy as np
import configparser

from time import time_ns
from snow_config import SnowConfig
from fluid_grid import FluidGrid, FakeGrid
vec3 = ti.types.vector(3, float)

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SnowConfig, GGUI = True):
        self.cfg = config
        self.grid_type = self.cfg.grid_type
        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_size = self.cfg.domain_size
        self.num_particles = self.cfg.num_particles
        self.domain_end = self.domain_start + self.domain_size
        self.particle_radius = self.cfg.particle_radius 
        self.boundary_particle_radius = self.cfg.boundary_particle_radius # move to config
        self.num_b_particles = self.cfg.num_boundary_particles
        self.dim = 3 # 3D simulation
        self.init_density = self.cfg.init_density
        self.gravity = ti.Vector(self.cfg.gravity)
        self.temperature = -10.0 # degrees Celsuis
        # self.m_k = ti.math.pi * float(4/3) * (self.particle_radius ** self.dim) * self.init_density # particle mass
        self.m_k = self.cfg.mass
        self.particle_spacing = self.cfg.particle_spacing
        self.boundary_particle_spacing = self.cfg.boundary_particle_spacing
        # self.m_k = 0.008
        self.smoothing_radius = self.cfg.smoothing_radius

        self.wind_direction = ti.Vector(self.cfg.wind_direction)
        self.enable_wind = False
        self.initialize_type = self.cfg.initialize_type
        self.grid_spacing = self.smoothing_radius * 2

        self.max_particles_per_cell = self.cfg.grid_max_particles_per_cell

        self.friction_coef = self.cfg.friction
        self.m_psi = self.cfg.m_psi

        self.object_paths = self.cfg.object_paths
        self.object_scales = self.cfg.object_scales
        self.object_pos = self.cfg.object_pos

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

        if self.grid_type == 'fake':
            self.fluid_grid = FakeGrid(self.domain_start, self.domain_end, self.smoothing_radius)
            self.b_grid = FakeGrid(self.domain_start, self.domain_end, self.smoothing_radius)
        elif self.grid_type == 'fluid':
            self.fluid_grid = FluidGrid(self.domain_start, self.domain_end, self.smoothing_radius)
            self.b_grid = FluidGrid(self.domain_start, self.domain_end, self.smoothing_radius)

        self.padding = 0.1 * self.grid_spacing

        # if self.object_paths is not None:
        self.import_boundary_objects()
        
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

    def import_boundary_objects(self):
        pos_tmp = []
        self.boundary_objects = []
        # offsets to later retrieve the particles of object i
        object_offsets = [] 
        total_num_b_particles = self.num_b_particles
        for path, scale, pos in zip(self.object_paths, self.object_scales, self.object_pos):
            print("loading boundary object from", path)
            if not path.endswith(".npy"): continue
            with open(path, "rb") as rf:
                particle_pos = np.load(rf)
                particle_pos *= scale
                particle_pos += pos
                pos_tmp.append(particle_pos)
                object_offsets.append(total_num_b_particles)
                total_num_b_particles += particle_pos.shape[0]
        self.boundary_particles = ti.Vector.field(self.dim, dtype=float,  shape=total_num_b_particles)
        self.boundary_velocity = ti.Vector.field(self.dim, dtype=float, shape=total_num_b_particles)

        self.boundary_particles_volume = ti.field(float,  shape=total_num_b_particles)
        self.boundary_colors = ti.Vector.field(self.dim, dtype=float, shape=total_num_b_particles)
        # next, load the positions into the ti field
        @ti.kernel
        def copy_slice(dest:ti.template(), src:ti.types.ndarray(), offset:int):
            for i in range(src.shape[0]):
                v = ti.Vector([src[i, 0], src[i, 1], src[i, 2]])
                dest[offset + i] = v
        for i in range(len(object_offsets)):
            offset = object_offsets[i]
            obj_pos = pos_tmp[i]
            copy_slice(self.boundary_particles, obj_pos, offset)
            self.boundary_objects.append(BoundaryObject(offset, obj_pos.shape[0]))
        self.num_b_particles = total_num_b_particles

    def object_control_gui(self):
        object_idx = 0
        bobj = self.boundary_objects[object_idx]
        gui = self.window.get_gui()
        with gui.sub_window("Boundary Object controls", x=0, y=0, width=0.5, height=1/5):
            any_update = False
            gui.text(f"Object {object_idx}")
            new_scale = gui.slider_float("Scale", bobj.scale, 0.2, 5.0)
            if new_scale != bobj.scale:
                bobj.rescale(new_scale, self)
                any_update = True
            old_x = bobj.pos[0]
            old_y = bobj.pos[1]
            old_z = bobj.pos[2]
            new_x = gui.slider_float("Position X", old_x, -3.0, 3.0)
            new_y = gui.slider_float("Position Y", old_y, -3.0, 3.0)
            new_z = gui.slider_float("Position Z", old_z, -3.0, 3.0)
            new_pos = np.array([new_x, new_y, new_z], float)
            if (new_pos != bobj.pos).any():
                bobj.translate(new_pos, self)
                any_update = True
            if any_update:
                self.b_grid.update_grid(self.boundary_particles)

    def initialize_particle_block(self, len_x:float, len_y:float, len_z:float, origin:ti.template()):
        # print("Block length", block_length)
        print((int(len_x / self.particle_spacing)), int(len_y / self.particle_spacing), int(len_z / self.particle_spacing))
        requested_num_particles = int((len_x / self.particle_spacing) * (len_y / self.particle_spacing) * (len_z / self.particle_spacing))
        print("requested num particles", requested_num_particles, "max num particles", self.cfg.block_max_num_particles)

        @ti.kernel
        def _fill(posiitons: ti.template(), len_x:float, len_y:float, len_z:float, particle_spacing: float, num_particles:ti.types.i64, origin:ti.template()):

            for i in range(num_particles):
                posiitons[i] = ti.Vector([0.0, 0.0, 0.0])

            cnt = 0
            for i in range(int((len_x / particle_spacing))):
                for j in range(int((len_z / particle_spacing))):
                    for k in range(int((len_y / particle_spacing))):
                        if cnt > num_particles:
                            continue
                        x = i * (particle_spacing) + origin[0]
                        y = j * (particle_spacing) + origin[2]
                        z = k * (particle_spacing) + origin[1]
                        posiitons[int(k * (len_x / particle_spacing) * (len_z / particle_spacing) + j * (len_x / particle_spacing) + i)] = ti.Vector([x, z, y])
                        cnt += 1

        _fill(self.position, len_x, len_y, len_z, self.particle_spacing, self.num_particles, origin)

    @ti.kernel
    def initialize_boundary_particle_block(self, len_x:float, len_y:float, len_z:float, origin:ti.template()):
        print("Boundary Block length", len_x)
        # block_position = origin
        print((int(len_x / self.boundary_particle_spacing)), int(len_y / self.boundary_particle_spacing), int(len_z / self.boundary_particle_spacing))

        for i in range(int(len_x / self.boundary_particle_spacing)):
            for j in range(int(len_z / self.boundary_particle_spacing)):
                for k in range(int(len_y / self.boundary_particle_spacing)):
                    x = i * (self.boundary_particle_spacing) + origin[0]
                    y = j * (self.boundary_particle_spacing) + origin[2]
                    z = k * (self.boundary_particle_spacing) + origin[1]
                    self.boundary_particles[int(k * (len_x / self.boundary_particle_spacing) * (len_z / self.boundary_particle_spacing) + j * (len_x / self.boundary_particle_spacing) + i)] = ti.Vector([x, z, y])


    @ti.kernel
    def initialize_random_particles(self):
        print("making random particles")
        for i in range(self.num_particles):
            x = (self.domain_size[0] - self.domain_size[0] * 0.1) * ti.random(dtype=float) + self.domain_size[0] * 0.1
            y = (self.domain_size[1] - self.domain_size[1] * 0.1)  * ti.random(dtype=float) + self.domain_size[1] * 0.1
            z = (self.domain_size[2] - self.domain_size[2] * 0.1)  * ti.random(dtype=float) + self.domain_size[2] * 0.1
            self.position[i] = ti.Vector([x, y, z])

    # simple helper to initialize deformation gradient as the identity

    def physics_initialize(self):
        self.gradient_initialize()
        self.velocity_initialize()
        self.acceleration_initialize()

    @ti.kernel
    def gradient_initialize(self):
        for i in range(self.num_particles):
            self.deformation_gradient[i] = ti.Matrix.identity(float, 3)

    @ti.kernel
    def velocity_initialize(self):
        for i in range(self.num_particles):
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            
    @ti.kernel
    def acceleration_initialize(self):
        for i in range(self.num_particles):
            self.acceleration[i] = ti.Vector([0.0, 0.0, 0.0])


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
            self.initialize_particle_block(self.cfg.block_length, self.cfg.block_height, self.cfg.block_width, block_origin)
        elif self.initialize_type == 'box_test':
            pass
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

        # self.boundary_initialize()
        boundary_origin = ti.field(float, 3)
        boundary_origin.from_numpy(self.cfg.boundary_origin)
        self.initialize_boundary_particle_block(self.cfg.boundary_length, self.cfg.boundary_height, self.cfg.boundary_width, boundary_origin)
        self.b_grid.update_grid(self.boundary_particles)
        self.boundary_velocity_initialize()
        self.physics_initialize()
        print("Intialized!")

    @ti.kernel
    def color_reset(self):
        for i in range(self.num_particles):
            self.colors[i] = ti.Vector([1.0, 1.0, 1.0])
        for i in range(self.num_b_particles):
            self.boundary_colors[i] = ti.Vector([0.4, 0.4, 0.4])
    def update_grid(self):
        self.color_reset()
        self.fluid_grid.update_grid(self.position)

    @ti.func
    def set_neighbor_color(self, i, j, color):
        self.colors[j] = color

    @ti.func
    def set_b_neighbor_color(self, i, j, color):
        self.boundary_colors[j] = color

    @ti.kernel
    def color_neighbors(self, i:int, color:vec3):
        self.colors[i] = ti.Vector([0.0, 1.0, 0.0])
        self.for_all_neighbors_vec3(ti.Vector([i]), self.set_neighbor_color, color)

    @ti.kernel
    def color_b_neighbors(self, i:int, color:vec3):
        self.for_all_b_neighbors_vec3(ti.Vector([i]), self.set_b_neighbor_color, color)
    

    def update_boundary_grid(self):
        self.b_grid.update_grid(self.boundary_particles)

    @ti.func
    def for_all_neighbors_vec3(self, i, func : ti.template(), ret : vec3):
        # pos = self.position[i]
        self.fluid_grid.for_all_neighbors_vec3(i, self.position, func, ret, self.smoothing_radius)

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
        
    @ti.func
    def for_all_b_neighbors_vec3(self, i, func : ti.template(), ret : vec3):
        '''
            Boundary neighbors of fluid particle i.
        '''
        position = self.position[i]
        self.b_grid.for_all_b_neighbors_vec3(i, position, self.boundary_particles, func, ret, self.smoothing_radius)


    def visualize(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.draw_domain()
        self.draw_particles()
        self.canvas.scene(self.scene)
        if self.boundary_objects != []:
            self.object_control_gui()
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
        self.scene.particles(self.position, color = (0.99, 0.99, 0.99), radius = self.particle_radius, per_vertex_color=self.colors)
        self.scene.particles(self.boundary_particles, per_vertex_color=self.boundary_colors,
                              radius = self.boundary_particle_radius)
        
class BoundaryObject:
    def __init__(self, offset, num_particles) -> None:
        self.scale = 1.0
        self.pos = np.zeros(3)
        self.data_offset = offset
        self.num_particles = num_particles

    def rescale(self, new_scale, ps:ParticleSystem):
        @ti.kernel
        def multiply_slice(dest:ti.template(), offset:int, length:int, factor:float):
            for i in range(length):
                dest[offset + i] *= factor
        ratio = new_scale / self.scale
        multiply_slice(ps.boundary_particles, self.data_offset,
            self.num_particles, ratio)
        self.scale = new_scale
    
    def translate(self, new_pos, ps:ParticleSystem):
        @ti.kernel
        def add_slice(dest:ti.template(), offset:int, length:int, value:ti.types.vector(3, float)):
            for i in range(length):
                dest[offset + i] += value
        diff = ti.Vector(new_pos - self.pos)
        add_slice(ps.boundary_particles, self.data_offset, self.num_particles, diff)
        self.pos = new_pos