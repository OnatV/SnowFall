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
        self.density = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.position = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.position_0 = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.pressure = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        
    @ti.kernel
    def initialize_fields(self):
        print("initilizing particle positions...")
        for i in range(self.num_particles):
            # set it so that it fills up most of the default domain size, should be done more cleverly in the future
            x = 4 * ti.random(dtype=float) + 0.5
            y = 4 * ti.random(dtype=float) + 0.5
            z = 4 * ti.random(dtype=float) + 0.5
            self.position[i] = ti.Vector([x, y, z]) 

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


