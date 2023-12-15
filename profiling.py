import taichi as ti
import cProfile as cprof

from pathlib import Path
from contextlib import redirect_stdout
from sys import argv
from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver

profDir = "./profiling"

# settings
ARCH = "cpu" #if "cpu" in argv else "gpu"
proftype = "taichi" if "taichi" in argv else "cprof"


# ability to profile the solver and the render functions
# info is saved to a ./profiling/*.prof file

# you may visualize it using snakeviz:
# pip install snakeviz
# snakeviz <file>

# on cmdline pass strings like cpu or gpu and taichi or cprof
# to select profiling type
# ex: python profiling.py gpu taichi

def main():
    ti.init(arch=ti.cpu if ARCH == "cpu" else ti.gpu, kernel_profiler=(proftype == "taichi"))
    cfg = SnowConfig("configs/snow_brick.ini")
    ps = ParticleSystem(cfg)
    snow_solver = SnowSolver(ps)

    p = Path(profDir)
    if not p.exists():
        p.mkdir()

    snow_solver.step(cfg.deltaTime, 0)
    if proftype == "cprof":
        cprof.runctx("s.step(cfg.deltaTime, 0.1)", {}, {
            "s": snow_solver,
            "cfg": cfg
        },  str(p / f"solver_{ARCH}_{proftype}.prof"))
        cprof.runctx("ps.visualize()", {}, {
            "ps": ps
        }, str(p / f"render_{ARCH}_{proftype}.prof") + f"_{ARCH}_{proftype}")
    elif proftype == "taichi":
        ti.profiler.clear_kernel_profiler_info()
        ti.profiler.clear_scoped_profiler_info()
        snow_solver.step(cfg.deltaTime, 1)
        with open(str(p / f"solver_{ARCH}_{proftype}.prof"), "w") as fout:
            with redirect_stdout(fout):
                ti.profiler.print_scoped_profiler_info()
                ti.profiler.print_kernel_profiler_info('trace')
    

if __name__ =='__main__':
    main()