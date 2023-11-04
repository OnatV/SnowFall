import numpy as np
import taichi as ti

from utils import *
from particle_system import ParticleSystem

if __name__ =='__main__':
    ti.init()
    print_about()
    cfg = SnowConfig()
    ps = ParticleSystem(cfg)
    while ps.window.running:
        ps.visualize()