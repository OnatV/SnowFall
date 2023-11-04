import taichi as ti
import numpy as np

from particle_system import ParticleSystem

class SnowSolver:
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        