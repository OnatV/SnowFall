import numpy as np
import configparser
import json
# this file is to encapsulate configuration hyperparmeters for snow simulation
# perhaps at some point in the future, we can add a parser to store these as files.
# see https://github.com/erizmr/SPH_Taichi/blob/master/config_builder.py for similar approach
class SnowConfig:
    def __init__(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        # Physical parameters
        self.gravity = list2vec(config['PHYSICS']['gravity'])
        self.wind_direction = list2vec(config['PHYSICS']['wind_direction'])
        self.theta_c = float(config['PHYSICS']['theta_c'])
        self.theta_s = float(config['PHYSICS']['theta_s'])
        self.init_density = float(config['PHYSICS']['init_density'])
        self.mass = float(config['PHYSICS']['mass'])
        self.young_mod = float(config['PHYSICS']['young_mod'])
        self.xi = float(config['PHYSICS']['xi'])
        self.nu = float(config['PHYSICS']['nu'])

        self.friction = float(config['PHYSICS']['friction'])
        self.m_psi = float(config['PHYSICS']['m_psi'])
        # Simulation parameters
        self.num_particles = int(config['SIMULATION']['num_particles'])
        self.domain_size = list2vec(config['SIMULATION']['domain_size'])
        self.deltaTime = float(config['SIMULATION']['delta_time'])
        self.particle_radius = float(config['SIMULATION']['particle_radius'])
        self.boundary_particle_radius = float(config['SIMULATION']['boundary_particle_radius'])
        self.smoothing_radius_ratio = float(config['SIMULATION']['smoothing_radius_ratio'])
        self.smoothing_radius = float(config['SIMULATION']['smoothing_radius'])
        self.grid_max_particles_per_cell = int(config['SIMULATION']['max_particles_per_cell']) ##Needed because taichi doesn't support dynamic arrays well, can be decreased if grid spacing is decreased        

        self.enable_wind = bool(config['SIMULATION']['enable_wind'].lower() == "true") 
        self.enable_adhesion = bool(config['SIMULATION']['enable_adhesion'].lower() == "true") 
        self.enable_friction = bool(config['SIMULATION']['enable_friction'].lower() == "true") 
        self.enable_elastic_solver = bool(config['SIMULATION']['enable_elastic_solver'].lower() == "true") 
        self.enable_compression_solver = bool(config['SIMULATION']['enable_compression_solver'].lower() == "true") 
        self.verbose_print = bool(config['SIMULATION']['verbose_print'].lower() == "true")

        # # values from paper
        self.initialize_type = config['SIMULATION']['initialize_type']
        self.max_time = float(config['SIMULATION']['max_time'])
        self.grid_type = config['SIMULATION']['grid_type']
        if 'BLOCK' in config.keys():
            self.block_max_num_particles = int(config['BLOCK']['max_num_particles'])
            self.block_origin = list2vec(config['BLOCK']['position'])
            self.block_length = float(config['BLOCK']['length'])
            self.block_width = float(config['BLOCK']['width'])
            self.block_height = float(config['BLOCK']['height'])
            self.particle_spacing = float(config['BLOCK']['spacing'])
            self.num_particles = int((self.block_length / self.particle_spacing) * (self.block_width / self.particle_spacing) * (self.block_height / self.particle_spacing))
            if self.num_particles > self.block_max_num_particles:
                self.num_particles = self.block_max_num_particles
            print("num particles", self.num_particles)

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
            self.boundary_particle_spacing = float(config['BOUNDARY']['spacing'])
            self.num_boundary_particles = int((self.boundary_length / self.boundary_particle_spacing) * (self.boundary_width / self.boundary_particle_spacing) * (self.boundary_height / self.boundary_particle_spacing))

        if 'BOUNDARY_OBJECTS' in config.keys():
            s = config['BOUNDARY_OBJECTS']['paths']
            self.object_paths =  map(str.strip, s.split(','))
            self.object_scales = list2vec(config['BOUNDARY_OBJECTS']['scales'])
            self.object_pos = list2vec(config['BOUNDARY_OBJECTS']['positions'])

        else:
            self.object_paths = []
            self.show_gui = False

        print(self.initialize_type)
def list2vec(strlist: str):
    return np.array(json.loads(strlist))
        