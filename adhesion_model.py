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


    @ti.func
    def compute_adh_i_b(self, i_idx: ti.template(), b_idx: ti.template(), res : ti.template()):
        h = 1.0
        V_b = self.ps.boundary_particles_volume[b_idx]
        rest_density_b = 4000
        Psi_b = rest_density_b * V_b
        m_i = self.ps.m_k

        x_ib = self.ps.position[i_idx] - self.ps.boundary_particles[b_idx]
        coeff = self.beta * m_i * Psi_b * adhesion_spline(x_ib.norm(), h)
        res -= coeff * x_ib.normalized()

    @ti.func
    def compute_adhesion_force(self, i_idx: ti.template()):
        res = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_b_neighbors(i_idx, self.compute_adh_i_b, res)   
        self.ps.acceleration[i_idx] += res

        