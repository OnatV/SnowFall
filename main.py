import numpy as np
import taichi as ti

from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver

import argparse



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--config', action='store', required=True)
    args = parser.parse_args()
    ti.init(arch=ti.cpu, debug=args.debug)
    # print_about()
    cfg = SnowConfig(args.config)
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)
    # print(ps.fluid_grid.to_grid_idx(ti.Vector([0.0, 0.0, 0.0])))
    # print(ps.fluid_grid.to_grid_idx(ti.Vector([1.0, 1.0, 1.0])))
    # print(ps.fluid_grid.to_grid_idx(ti.Vector([0.1, 0.1, 0.1])))
    # print(ps.fluid_grid.to_grid_idx(ti.Vector([0.1, 0.2, 0.1])))
    sim_is_running = False
    while ps.window.running:
        # press SPACE to start the simulation
        if ps.window.is_pressed(ti.ui.SPACE, ' '): sim_is_running = True
        if ps.window.is_pressed(ti.ui.ALT): sim_is_running = False
        if sim_is_running:
            snow_solver.step(cfg.deltaTime)
            # sim_is_running = False # press space for one step at a time
        ps.visualize()