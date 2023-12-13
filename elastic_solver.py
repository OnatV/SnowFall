import taichi as ti
import numpy as np
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve import bicgstab


from kernels import cubic_kernel_derivative, cubic_kernel_derivative_corrected
from particle_system import ParticleSystem

@ti.data_oriented
class ElasticSolver:
    def __init__(self, ps:ParticleSystem):
        self.ps = ps
        self.velocity_pred = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.basis_vec = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.basis_grad = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.velocity_pred_grad = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        # self.F_E_pred = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.rhs = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.lhs = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.stress_tensor_pred = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.basis_stress_tensor = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.deltaTime = 0.1

    @ti.func
    def get_volume(self, i):
        return (self.ps.m_k / ti.math.max(self.ps.density[i], 1e-5 ) )

    # ----------------------------------------- RHS -----------------------------------------#

    @ti.func
    def compute_vel_pred(self, i):
        self.velocity_pred[i] = self.ps.velocity[i] + self.deltaTime * self.ps.acceleration[i] # by now, all other accelerations should have been integrated

    @ti.func
    def compute_velocity_gradient(self,i):
        grad_v_i_s_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_neighbors(i, self.helper_compute_velocity_gradient_uncorrected, grad_v_i_s_prime)
        grad_v_i_b_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_b_neighbors(i, self.helper_compute_velocity_gradient_b_uncorrected, grad_v_i_b_prime)
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]
        L_i = ti.Matrix.identity(float, 3)
        grad_v_i_tilde = grad_v_i_s_prime @ L_i.transpose() + (grad_v_i_b_prime  @ L_i.transpose()).trace() * ti.Matrix.identity(float, 3) / 3        
        V_i_prime = (grad_v_i_s_prime + grad_v_i_b_prime).trace() * ti.Matrix.identity(float, 3) / 3
        R_i_tilde = (grad_v_i_tilde - grad_v_i_tilde.transpose()) / 2 
        S_i_tilde = (grad_v_i_tilde + grad_v_i_tilde.transpose()) / 2 - grad_v_i_tilde.trace() * ti.Matrix.identity(float, 3) / 3
        return V_i_prime + R_i_tilde + S_i_tilde

    @ti.func
    def helper_compute_velocity_gradient_uncorrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. Computes sum in Eq.17 For the fluid particles.
        '''
        v_j = self.velocity_pred[j_idx] # v_j: vec3
        v_i = self.velocity_pred[i_idx] # v_i: vec3
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx] # x_ij: vec3
        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius) # del_w_ij: vec3
        V_j = self.get_volume(j_idx) # V_j: float

        res += (v_j - v_i).outer_product(V_j * del_w_ij) 

    @ti.func
    def helper_compute_velocity_gradient_b_uncorrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. Computes sum in Eq.17 For the boundary particles.
        '''
        v_j = self.ps.boundary_velocity[j_idx] # v_j: vec3
        v_i = self.velocity_pred[i_idx] # v_i: vec3
        x_ij = self.ps.position[i_idx] - self.ps.boundary_particles[j_idx] # x_ij: vec3
        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius) # del_w_ij: vec3
    
        V_j = self.ps.boundary_particles_volume[j_idx]
        res += (v_j - v_i).outer_product(V_j * del_w_ij)
    
    @ti.func 
    def compute_stress_tensor_pred(self, i):
        F_E_pred = self.ps.deformation_gradient[i] + self.deltaTime * self.velocity_pred_grad[i] @ self.ps.deformation_gradient[i]
        strain = 0.5 * (F_E_pred + F_E_pred.transpose())
        self.stress_tensor_pred[i] = (2 * self.ps.G_t_i[i]) * (strain - ti.Matrix.identity(float, 3) )

    @ti.func
    def compute_stress_pred_div(self, i):
        stress_tensor_b_i = self.stress_tensor_pred[i].trace() * ti.Matrix.identity(float, 3) / 3
        sum_fluid = ti.Matrix.zero(float, 3)
        self.ps.for_all_neighbors(i, self.compute_stress_pred_div_fluid_helper, sum_fluid)
        # sum_b = ti.Matrix.zero(float, 3)
        sum_b = ti.Matrix.zero(float, 3)
        self.ps.for_all_b_neighbors(i, self.compute_stress_pred_div_b_helper, sum_b)
        return sum_fluid + stress_tensor_b_i @ sum_b


    @ti.func
    def compute_stress_pred_div_fluid_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.position[j]
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]
        sum += self.stress_tensor_pred[j] @ (-self.get_volume(j) * cubic_kernel_derivative_corrected(-x_ij, self.ps.smoothing_radius, L_i)) + \
            self.stress_tensor_pred[i] @ (self.get_volume(j) * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i))

    @ti.func
    def compute_stress_pred_div_b_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.boundary_particles[j]
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]        
        sum += self.ps.boundary_particles_volume[j] * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i)

    @ti.kernel
    def compute_rhs(self):
        for i in ti.grouped(self.ps.position):
            self.compute_vel_pred(i)
            self.velocity_pred_grad[i] = self.compute_velocity_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_stress_tensor_pred(i)            
            self.rhs[i] = (1.0 / self.ps.density[i]) * self.compute_stress_pred_div(i)
            # print(self.rhs[i])

    # ----------------------------------------- LHS -----------------------------------------#
    # this func is a copy+paste of above velocity gradient discretization
    @ti.func
    def compute_basis_gradient(self,i):
        grad_v_i_s_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_neighbors(i, self.helper_compute_basis_gradient_uncorrected, grad_v_i_s_prime)
        grad_v_i_b_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_b_neighbors(i, self.helper_compute_basis_gradient_b_uncorrected, grad_v_i_b_prime)
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]
        L_i = ti.Matrix.identity(float, 3)
        grad_v_i_tilde = grad_v_i_s_prime @ L_i.transpose() + (grad_v_i_b_prime  @ L_i.transpose()).trace() * ti.Matrix.identity(float, 3) / 3        
        V_i_prime = (grad_v_i_s_prime + grad_v_i_b_prime).trace() * ti.Matrix.identity(float, 3) / 3
        R_i_tilde = (grad_v_i_tilde - grad_v_i_tilde.transpose()) / 2 
        S_i_tilde = (grad_v_i_tilde + grad_v_i_tilde.transpose()) / 2 - grad_v_i_tilde.trace() * ti.Matrix.identity(float, 3) / 3
        return V_i_prime + R_i_tilde + S_i_tilde

    @ti.func
    def helper_compute_basis_gradient_uncorrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. Computes sum in Eq.17 For the fluid particles.
        '''
        v_j = self.basis_vec[j_idx] # v_j: vec3
        v_i = self.basis_vec[i_idx] # v_i: vec3
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx] # x_ij: vec3
        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius) # del_w_ij: vec3
        V_j = self.get_volume(j_idx) # V_j: float

        res += (v_j - v_i).outer_product(V_j * del_w_ij) 

    @ti.func
    def helper_compute_basis_gradient_b_uncorrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. Computes sum in Eq.17 For the boundary particles.
        '''
        v_j = self.ps.boundary_velocity[j_idx] # v_j: vec3
        v_i = self.basis_vec[i_idx] # v_i: vec3
        x_ij = self.ps.position[i_idx] - self.ps.boundary_particles[j_idx] # x_ij: vec3
        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius) # del_w_ij: vec3
    
        V_j = self.ps.boundary_particles_volume[j_idx]
        res += (v_j - v_i).outer_product(V_j * del_w_ij)

    @ti.func 
    def compute_basis_stress_tensor(self, i):
        prod = self.basis_grad[i] @ self.ps.deformation_gradient[i]
        strain = (prod + prod.transpose())
        self.basis_stress_tensor[i] = self.ps.G_t_i[i] * (strain)

    @ti.func
    def compute_basis_stress_div(self, i):
        stress_tensor_b_i = self.basis_stress_tensor[i].trace() * ti.Matrix.identity(float, 3) / 3
        sum_fluid = ti.Matrix.zero(float, 3)
        self.ps.for_all_neighbors(i, self.compute_basis_stress_div_fluid_helper, sum_fluid)
        # sum_b = ti.Matrix.zero(float, 3)
        sum_b = ti.Matrix.zero(float, 3)
        self.ps.for_all_b_neighbors(i, self.compute_basis_stress_div_b_helper, sum_b)
        return sum_fluid + stress_tensor_b_i @ sum_b


    @ti.func
    def compute_basis_stress_div_fluid_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.position[j]
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]
        sum += self.basis_stress_tensor[j] @ (-self.get_volume(j) * cubic_kernel_derivative_corrected(-x_ij, self.ps.smoothing_radius, L_i)) + \
            self.basis_stress_tensor[i] @ (self.get_volume(j) * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i))

    @ti.func
    def compute_basis_stress_div_b_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.boundary_particles[j]
        L_i = self.ps.correction_matrix[i]
        if self.ps.is_pseudo_L_i[i]:
            L_i = self.ps.pseudo_correction_matrix[i]        
        sum += self.ps.boundary_particles_volume[j] * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i)

    @ti.kernel
    def compute_lhs(self):
        for i in ti.grouped(self.ps.position):
            self.basis_grad[i] = self.compute_basis_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_basis_stress_tensor(i)
            self.lhs[i] = self.basis_vec[i] - (self.deltaTime / self.ps.density[i]) * self.compute_basis_stress_div(i)
    
# this is the linear operator that scipy will use to solve the Bi-CGSTAB
# this function will need to set the basis vector
# input v is (3 * N, 1), we need to set it to N by 3 (reshape) 
# and then populate self.basis vector with that information
def linop(v, es):
    es.basis_vec.from_numpy(v.reshape([es.ps.num_particles, 3]).astype(np.float32))
    es.compute_lhs()
    return es.lhs.to_numpy().reshape([3 * es.ps.num_particles,])

def solve(es: ElasticSolver, dt:float):
    es.deltaTime = dt
    es.compute_rhs()
    b = es.rhs.to_numpy().reshape([3 * es.ps.num_particles,])
    A = LinearOperator(shape=(3 * es.ps.num_particles, 3 * es.ps.num_particles), matvec=lambda x: linop(x, es))
    return bicgstab(A=A, b=b, maxiter=100, tol=1e-3)
    

    