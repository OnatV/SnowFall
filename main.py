import numpy as np
import taichi as ti

from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver

import argparse



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ti.init(arch=ti.cpu, debug=args.debug)
    print_about()
    cfg = SnowConfig()
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)
    sim_is_running = False
    while ps.window.running:
        # press SPACE to start the simulation
        if ps.window.is_pressed(ti.ui.SPACE, ' '): sim_is_running = True
        if ps.window.is_pressed(ti.ui.ALT): sim_is_running = False
        # if sim_is_running:
        #     snow_solver.step(cfg.deltaTime)
        #     sim_is_running = False # press space for one step at a time
        ps.visualize()