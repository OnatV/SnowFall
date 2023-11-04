import numpy as np
import taichi as ti

from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver

if __name__ =='__main__':
    ti.init()
    print_about()
    cfg = SnowConfig()
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)
    sim_is_running = False
    while ps.window.running:
        # press SPACE to start the simulation
        if ps.window.is_pressed(ti.ui.SPACE, ' '): sim_is_running = ~sim_is_running
        if sim_is_running:
            snow_solver.step(cfg.deltaTime)
        ps.visualize()