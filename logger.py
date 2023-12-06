import taichi as ti
import numpy as np
import os
from particle_system import ParticleSystem

class Logger:
    def __init__(self, ps: ParticleSystem,
        config,
    ):
        self.ps = ps
        self.cfg = config
        self.fields = [x.strip() for x in self.cfg.logging_fields.split(",")]
        self.init_logging_directory()

    def init_logging_directory(self):
        os.makedirs(os.path.join(os.getcwd(), self.cfg.log_dir), exist_ok = True)

    def log_step(self, time):