import taichi as ti
from particle_system import ParticleSystem

@ti.func
def adhesion_spline(r,h):

    term = 0.0
    if h >= r and 2*r > h:
        term = - 4 * r * r / h + 6*r - 2*h 
    
    return 0.007 * ti.math.sqrt(ti.math.sqrt(term)) / (h ** 3.25)

@ti.data_oriented
class AdhesionModel:
    
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        self.beta = 1.0 ##Section 4 of Versatile Surface Tension and Adhesion for SPH Fluids
        self.numerical_epsilon= 1e-6


    @ti.func
    def compute_adh_i_b(self, i_idx: ti.template(), b_idx: ti.template(), res : ti.template()):

        h = 0.03
        V_b = self.ps.boundary_particles_volume[b_idx]
        rest_density = self.ps.rest_density[i_idx] 
        Psi_b = rest_density * V_b
        m_i = self.ps.m_k

        x_ib = self.ps.position[i_idx] - self.ps.boundary_particles[b_idx]

        coeff = self.beta * Psi_b * adhesion_spline(x_ib.norm(), h)

        ###We need to understant why x_ib is 0.0
        direction = ti.Vector([0.0, 0.0, 0.0])
        if x_ib.norm() > self.numerical_epsilon:
            direction = x_ib.normalized()

        out = coeff * direction
        if ti.math.isnan(out).any():
            print(f"GOT NAN out: {out}, coeff {coeff}, rest density: {rest_density}, boundry volume: {V_b}, adhesion_spline: {adhesion_spline(x_ib.norm(), h)}, x_ib: {x_ib}, ")
        res -= out

    @ti.func
    def compute_adhesion_force(self, i_idx: ti.template()):
        res = ti.Vector([0.0, 0.0, 0.0])

        self.ps.for_all_b_neighbors(i_idx, self.compute_adh_i_b, res)
        # if i_idx[0] == 0:
        #     print("adhesion force for particle 0:" , res)   
        if ti.math.isnan(res[0]):
            print(f"GOT NAN adhesion force for {i_idx}: {res}")
        self.ps.acceleration[i_idx] += res

        