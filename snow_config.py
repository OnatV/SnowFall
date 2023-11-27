import numpy as np
import configparser
import json
# this file is to encapsulate configuration hyperparmeters for snow simulation
# perhaps at some point in the future, we can add a parser to store these as files.
# see https://github.com/erizmr/SPH_Taichi/blob/master/config_builder.py for similar approach
class SnowConfig:
    # def __init__(self,
    #     num_particles: int = 200,
    #     gravity: np.array = np.array([0.0, -9.81, 0.0]),
    #     wind_direction : np.array = np.array([0.0, 0.0, -1.0]),
    #     domain_size: np.array = np.array([4.0, 4.0, 4.0]), # domain lower corner is at 0,0,0
    #     deltaTime = 0.001
    # ) -> None:
    #     self.num_particles = num_particles
    #     self.gravity = gravity
    #     self.domain_size = domain_size
    #     self.deltaTime = 0.01
    #     self.wind_direction = wind_direction
    #     self.smoothing_radius = 0.01
    #     # self.grid_spacing = 0.01 ##Spacing between grid cells should be close to particle radius
    #     self.grid_max_particles_per_cell = 100 ##Needed because taichi doesn't support dynamic arrays well, can be decreased if grid spacing is decreased

    #     # values from paper
    #     self.theta_c = 0.025
    #     self.theta_s = 0.0075

    #     self.enable_wind = True
    def __init__(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        # Physical parameters
        self.gravity = list2vec(config['PHYSICS']['gravity'])
        self.wind_direction = list2vec(config['PHYSICS']['wind_direction'])
        self.theta_c = float(config['PHYSICS']['theta_c'])
        self.theta_s = float(config['PHYSICS']['theta_s'])
        # Simulation parameters
        self.num_particles = int(config['SIMULATION']['num_particles'])
        self.domain_size = list2vec(config['SIMULATION']['domain_size'])
        self.deltaTime = float(config['SIMULATION']['delta_time'])
        self.smoothing_radius = float(config['SIMULATION']['smoothing_radius'])
        # # self.grid_spacing = 0.01 ##Spacing between grid cells should be close to particle radius
        self.grid_max_particles_per_cell = int(config['SIMULATION']['max_particles_per_cell']) ##Needed because taichi doesn't support dynamic arrays well, can be decreased if grid spacing is decreased        # # values from paper

def list2vec(strlist: str):
    return np.array(json.loads(strlist))
        