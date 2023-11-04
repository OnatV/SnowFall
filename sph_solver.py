import taichi as ti
import numpy as np

from particle_system import ParticleSystem

@ti.data_oriented
class SnowSolver:
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        self.time = 0

    @ti.kernel
    def enforce_boundary_3D(self):
        for i in range(self.ps.num_particles):
            if self.ps.position[i].y < self.ps.domain_origin[0]:
                self.ps.position[i].y = self.ps.domain_origin[0]
            if self.ps.position[i].z < self.ps.domain_origin[1]:
                self.ps.position[i].z = self.ps.domain_origin[1]
            if self.ps.position[i].x < self.ps.domain_origin[2]:
                self.ps.position[i].x = self.ps.domain_origin[2]
            if self.ps.position[i].y > self.ps.domain_end[0]:
                self.ps.position[i].y = self.ps.domain_end[0]
            if self.ps.position[i].z > self.ps.domain_end[1]:
                self.ps.position[i].z = self.ps.domain_end[1]
            if self.ps.position[i].x > self.ps.domain_end[2]:
                self.ps.position[i].x = self.ps.domain_end[2]

    @ti.kernel
    def cubic_kernel(self, r_norm):
        # use ps.smoothing_radius to calculate the kernel weight of particles
        # for now, sum over nearby particles
        pass
    
    @ti.kernel
    def cubic_kernel_derivative(self, r_norm)
        # use ps.smoothing_radius to calculate the derivative of kernel weight of particles
        # for now, sum over nearby particles
        pass

    @ti.kernel
    def calculate_acceleration(self, deltaTime: float):
        # f = ma
        # a = f / m
        # aggregate acceleration caused by gravity, pressure from nearby particles, and external forces
        # right now, only supports gravity
        for i in range(self.ps.num_particles):
            self.ps.acceleration[i] = self.ps.gravity

    @ti.kernel
    def calculate_velocity(self, deltaTime: float):
        for i in range(self.ps.num_particles):
            self.ps.velocity[i] = self.ps.velocity[i] + (deltaTime * self.ps.acceleration[i])

    @ti.kernel
    def update_position(self, deltaTime: float):
        for i in range(self.ps.num_particles):
            self.ps.position[i] = self.ps.position[i] + deltaTime * self.ps.velocity[i]

    def substep(self, deltaTime):
        self.calculate_acceleration(deltaTime)
        self.calculate_velocity(deltaTime)
        self.update_position(deltaTime)

    def step(self, deltaTime):
        # step physics
        self.substep(deltaTime)
        # enforce the boundary of the domain (and later rigid bodies)
        self.enforce_boundary_3D()
        # update time
        self.time += deltaTime
