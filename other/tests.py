from utils import *
from particle_system import ParticleSystem
from sph_solver import SnowSolver
import taichi as ti
import matplotlib.pyplot as plt

def run():
    ti.init(arch=ti.cpu)
    cfg = SnowConfig()
    ps = ParticleSystem(cfg)
    solver = SnowSolver(ps)
    
    # visualize lookup table
    n=100
    x = ti.field(dtype=float, shape=n)
    vlookup = ti.field(dtype=float, shape=n)
    vtrue = ti.field(dtype=float, shape=n)

    @ti.kernel
    def gen_data():
        for i in range(n):
            rand = ti.random(float) * solver.ps.smoothing_radius
            x[i] = rand 
            vlookup[i] = solver.kernel_lookup(rand)
            vtrue[i] = solver.cubic_kernel(rand)
    gen_data()
    x = x.to_numpy()
    vlookup = vlookup.to_numpy()
    vtrue = vtrue.to_numpy()
 
    fig, axs = plt.subplots(2)
    ax = axs[0]
    ax.scatter(x, vlookup)
    ax.scatter(x, vtrue)
    
    r = ti.Vector.field(n=3, dtype=float, shape=n)
    vlookup = ti.Vector.field(n=3, dtype=float, shape=n)
    vtrue = ti.Vector.field(n=3, dtype=float, shape=n)
    @ti.kernel
    def gen_data_grad():
        for i in range(n):
            rand = ti.Vector([
                ti.random(float) * solver.ps.smoothing_radius,
                ti.random(float) * solver.ps.smoothing_radius,
                ti.random(float) * solver.ps.smoothing_radius])
            r[i] = rand 
            vlookup[i] = solver.grad_kernel_lookup(rand)
            vtrue[i] = solver.cubic_kernel_derivative(rand)

    gen_data_grad()
    r = r.to_numpy()
    vlookup = vlookup.to_numpy()
    vtrue = vtrue.to_numpy()
    ax = axs[1]
    ax.scatter()

    plt.show()


run()