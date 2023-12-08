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
        self.init_density = float(config['PHYSICS']['init_density'])

        self.friction = float(config['PHYSICS']['friction'])
        self.m_psi = float(config['PHYSICS']['m_psi'])
        # Simulation parameters
        self.num_particles = int(config['SIMULATION']['num_particles'])
        self.domain_size = list2vec(config['SIMULATION']['domain_size'])
        self.deltaTime = float(config['SIMULATION']['delta_time'])
        self.particle_radius = float(config['SIMULATION']['particle_radius'])
        self.boundary_particle_radius = float(config['SIMULATION']['boundary_particle_radius'])
        self.smoothing_radius_ratio = float(config['SIMULATION']['smoothing_radius_ratio'])
        # # self.grid_spacing = 0.01 ##Spacing between grid cells should be close to particle radius
        self.grid_max_particles_per_cell = int(config['SIMULATION']['max_particles_per_cell']) ##Needed because taichi doesn't support dynamic arrays well, can be decreased if grid spacing is decreased        
        # # values from paper
        self.initialize_type = config['SIMULATION']['initialize_type']
        self.max_time = float(config['SIMULATION']['max_time'])
        if 'BLOCK' in config.keys():
            self.block_origin = list2vec(config['BLOCK']['position'])
            self.block_length = float(config['BLOCK']['length'])
            self.block_width = float(config['BLOCK']['width'])
            self.block_height = float(config['BLOCK']['height'])
            self.num_particles = int((self.block_length / self.particle_radius) * (self.block_width / self.particle_radius) * (self.block_height / self.particle_radius))

        if 'LOGGING' in config.keys(): # consider changing for upper/lowercase matches
            self.logging = config['LOGGING']['logging'] == 'true'
            self.log_dir = config['LOGGING']['log_dir']
            self.logging_fields = config['LOGGING']['fields']
            self.log_fps = float(config['LOGGING']['fps'])

        if 'BOUNDARY' in config.keys():
            self.boundary_origin = list2vec(config['BOUNDARY']['position'])
            self.boundary_length = float(config['BOUNDARY']['length'])
            self.boundary_width = float(config['BOUNDARY']['width'])
            self.boundary_height = float(config['BOUNDARY']['height'])
            self.num_boundary_particles = int((self.boundary_length / self.boundary_particle_radius) * (self.boundary_width / self.boundary_particle_radius) * (self.boundary_height / self.boundary_particle_radius))

        print(self.initialize_type)
def list2vec(strlist: str):
    return np.array(json.loads(strlist))
        