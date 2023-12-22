import numpy as np
import taichi as ti
import argparse
import time
import os

from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver
from logger import Logger


def run_simulation(config):
    pass


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--config', action='store', required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    ti.init(arch=ti.cpu, debug=args.debug)
    # print_about()
    cfg = SnowConfig(args.config)
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)
    sim_is_running = False
    sim_time = 0.0
    last_log_time = 0.0
    
    logger = Logger(ps, cfg, args.replay)
    if not args.replay:
        print("Creating a new simulation!")
        while ps.window.running and sim_time < cfg.max_time:
            # press SPACE to start the simulation
            if ps.window.is_pressed(ti.ui.SPACE, ' '): sim_is_running = True
            if ps.window.is_pressed(ti.ui.ALT): sim_is_running = False
            if ps.window.is_pressed(ti.ui.BACKSPACE):
                ps.window.running = False
            if sim_is_running:
                print("Time:", sim_time)
                snow_solver.step(cfg.deltaTime, sim_time)
                # sim_is_running = False # press space for one step at a time
                if sim_time - last_log_time > logger.log_time_step or sim_time == 0.0:
                    logger.log_step(sim_time)
                    last_log_time = sim_time
                sim_time += cfg.deltaTime            
            ps.visualize()
    else:
        print("Replaying simulation")
        start_time = time.time()
        current_time = time.time()
        logger.replay_step(current_time)
        while ps.window.running:
            if ps.window.is_pressed(ti.ui.SPACE, ' '):
                start_time = time.time()
                current_time = time.time()
                logger.current_step = 0
                sim_is_running = True
            if ps.window.is_pressed(ti.ui.ALT): sim_is_running = False
            if sim_is_running:
                if time.time() - current_time > 5 * logger.log_time_step:
                    print("Time:", current_time)
                    if logger.current_step >= logger.num_time_steps - 1:
                        print("finished!")
                        sim_is_running = False
                        continue                
                    current_time = time.time()
                    logger.replay_step(time.time())
            ps.visualize()

    snow_pos = ps.position.to_numpy()
    output_path = os.path.join(cfg.log_dir, "particle_positions.npy")
    with open(output_path, "wb") as snowOut:
        np.save(snowOut, snow_pos)