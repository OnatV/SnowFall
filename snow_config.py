import numpy as np

# this file is to encapsulate configuration hyperparmeters for snow simulation
# perhaps at some point in the future, we can add a parser to store these as files.
# see https://github.com/erizmr/SPH_Taichi/blob/master/config_builder.py for similar approach
class SnowConfig:
    def __init__(self,
        num_particles: int = 100000,
        gravity: np.array = np.array([0.0, -9.81, 0.0]),
        domain_size: np.array = np.array([5.0, 5.0, 5.0]), # domain lower corner is at 0,0,0
        deltaTime = 0.001
    ) -> None:
        self.num_particles = num_particles
        self.gravity = gravity
        self.domain_size = domain_size
        self.deltaTime = 0.01