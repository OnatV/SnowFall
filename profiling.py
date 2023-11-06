import taichi as ti
import cProfile as cprof
import pathlib

from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver

profDir = "./profiling"

# ability to profile the solver and the render functions
# info is saved to a ./profiling/*.prof file

# you may visualize it using snakeviz:
# pip install snakeviz
# snakeviz <file>

def main():
    ti.init()
    cfg = SnowConfig()
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)

    p = pathlib.Path(profDir)
    if not p.exists():
        p.mkdir()

    cprof.runctx("s.step(cfg.deltaTime)", {}, {
        "s": snow_solver,
        "cfg": cfg
    },  str(p / "solver.prof"))
    cprof.runctx("ps.visualize()", {}, {
        "ps": ps
    }, str(p / "render.prof"))
    
    # ti.profiler.print_scoped_profiler_info()

if __name__ =='__main__':
    main()