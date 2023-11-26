import numpy as np

# this file is to encapsulate configuration hyperparmeters for snow simulation
# perhaps at some point in the future, we can add a parser to store these as files.
# see https://github.com/erizmr/SPH_Taichi/blob/master/config_builder.py for similar approach
class SnowConfig:
    def __init__(self,
        num_particles: int = 10000,
        gravity: np.array = np.array([0.0, -9.81, 0.0]),
        wind_direction : np.array = np.array([0.0, 0.0, -1.0]),
        domain_size: np.array = np.array([2.0, 3.0, 4.0]), # domain lower corner is at 0,0,0
        deltaTime = 0.001
    ) -> None:
        self.num_particles = num_particles
        self.gravity = gravity
        self.domain_size = domain_size
        self.deltaTime = 0.01
        self.wind_direction = wind_direction

        self.grid_spacing = 0.01 ##Spacing between grid cells should be close to particle radius
        self.grid_max_particles_per_cell = 10 ##Needed because taichi doesn't support dynamic arrays well, can be decreased if grid spacing is decreased

        self.theta_c = 0
        self.theta_s = 1

        self.enable_wind = True
        