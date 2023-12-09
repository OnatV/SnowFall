import taichi as ti
import numpy as np
from particle_system import ParticleSystem


class ElasticSolver:
    def __init__(ps:ParticleSystem):
        self.ps = ps
        self.velocity_pred = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.F_E_pred = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.rhs = ti.field(dtype=float, shape=self.ps.num_particles)

    def solve():


    