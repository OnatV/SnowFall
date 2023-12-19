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
        self.rhs_prev = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.rhs = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.lhs = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.stress_tensor_pred = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.basis_stress_tensor = ti.Matrix.field(m=self.ps.dim, n=self.ps.dim, dtype=float, shape=self.ps.num_particles)

        self.a_G_ti = ti.Vector.field(self.ps.dim, dtype=float, shape=self.ps.num_particles)
        self.a_G_ti_new = ti.field(dtype=float, shape=(self.ps.num_particles,3))
        self.deltaTime = 0.1

    @ti.kernel
    def init_basis_vec_prev(self):
        for i in ti.grouped(self.ps.position):
            self.rhs_prev[i] = ti.Vector.zero(float, self.ps.dim)


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

        ##In the Paragraph between Eq17 and Eq18
        grad_v_i_tilde = grad_v_i_s_prime @ L_i.transpose() + (grad_v_i_b_prime  @ L_i.transpose()).trace() * ti.Matrix.identity(float, 3) / 3
        
        V_i_prime = (grad_v_i_s_prime + grad_v_i_b_prime).trace() * ti.Matrix.identity(float, 3) / 3
        R_i_tilde = (grad_v_i_tilde - grad_v_i_tilde.transpose()) / 2 
        S_i_tilde = (grad_v_i_tilde + grad_v_i_tilde.transpose()) / 2 - grad_v_i_tilde.trace() * ti.Matrix.identity(float, 3) / 3
        # if(i[0] == 0):
        #     print("---Vi---", V_i_prime)
        res = V_i_prime + R_i_tilde + S_i_tilde

        if ti.math.isnan(res).any():
                print(f"GOT {res} velocity_pred_grad for {i}:velocity_pred {self.velocity_pred[i]}, correction_matrix {L_i},fluid grad {grad_v_i_s_prime}, boundary grad {grad_v_i_b_prime}")
        return res
    
    @ti.func
    def clamp_deformation_gradients(self, matrix):
        U, S, V = ti.svd(matrix)
        S = ti.math.clamp(S, self.ps.theta_clamp_c, self.ps.theta_clamp_s)
        return V @ S @ V.transpose() ## This supposedly removes the rotation part

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
        F_E_pred = self.clamp_deformation_gradients(F_E_pred)
        strain = 0.5 * (F_E_pred + F_E_pred.transpose())
        self.stress_tensor_pred[i] = (2 * self.ps.G_t_i[i]) * (strain - ti.Matrix.identity(float, 3) )

    @ti.func
    def compute_stress_pred_div(self, i):
        stress_tensor_b_i = self.stress_tensor_pred[i].trace() * ti.Matrix.identity(float, 3) / 3
        sum_fluid = ti.Matrix.zero(float, 3)
        self.ps.for_all_neighbors(i, self.compute_stress_pred_div_fluid_helper, sum_fluid)

        sum_b = ti.Matrix.zero(float, 3)
        self.ps.for_all_b_neighbors(i, self.compute_stress_pred_div_b_helper, sum_b)
        return sum_fluid + stress_tensor_b_i @ sum_b


    @ti.func
    def compute_stress_pred_div_fluid_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.position[j]
        L_i = self.ps.correction_matrix[i]
        L_j = self.ps.correction_matrix[j]
        sum += self.stress_tensor_pred[j] @ (-self.get_volume(j) * cubic_kernel_derivative_corrected(-x_ij, self.ps.smoothing_radius, L_j)) + \
            self.stress_tensor_pred[i] @ (self.get_volume(j) * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i))

    @ti.func
    def compute_stress_pred_div_b_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.boundary_particles[j]
        L_i = self.ps.correction_matrix[i]     
        sum += self.ps.boundary_particles_volume[j] * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i)

    @ti.kernel
    def compute_rhs(self):
        for i in ti.grouped(self.ps.position):
            self.compute_vel_pred(i)
            self.velocity_pred_grad[i] = self.compute_velocity_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_stress_tensor_pred(i)            
            self.rhs[i] = (1.0 / self.ps.density[i]) * self.compute_stress_pred_div(i)
            if ti.math.isnan(self.rhs[i]).any():
                print(f"GOT {self.rhs[i]} rhs for {i}: density {self.ps.density[i]}, stress_tensor_pred {self.stress_tensor_pred[i]}")

    # ----------------------------------------- LHS -----------------------------------------#
    # this func is a copy+paste of above velocity gradient discretization
    @ti.func
    def compute_basis_gradient(self,i):
        grad_v_i_s_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_neighbors(i, self.helper_compute_basis_gradient_uncorrected, grad_v_i_s_prime)
        grad_v_i_b_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_b_neighbors(i, self.helper_compute_basis_gradient_b_uncorrected, grad_v_i_b_prime)
        L_i = self.ps.correction_matrix[i]
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
        if ti.math.isnan(strain).any():
            print(f"GOT {strain} strain for {i}:basis grad {self.basis_grad[i]}, deformation grad {self.ps.deformation_gradient[i]}")
        self.basis_stress_tensor[i] = self.ps.G_t_i[i] * (strain)


    @ti.func
    def compute_basis_stress_div(self, i):
        stress_tensor_b_i = self.basis_stress_tensor[i].trace() * ti.Matrix.identity(float, 3) / 3
        sum_fluid = ti.Matrix.zero(float, 3)
        self.ps.for_all_neighbors(i, self.compute_basis_stress_div_fluid_helper, sum_fluid)

        sum_b = ti.Matrix.zero(float, 3)
        self.ps.for_all_b_neighbors(i, self.compute_basis_stress_div_b_helper, sum_b)
        return sum_fluid + stress_tensor_b_i @ sum_b


    @ti.func
    def compute_basis_stress_div_fluid_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.position[j]
        L_i = self.ps.correction_matrix[i]
        L_j = self.ps.correction_matrix[j]
        sum += self.basis_stress_tensor[j] @ (-self.get_volume(j) * cubic_kernel_derivative_corrected(-x_ij, self.ps.smoothing_radius, L_j)) + \
            self.basis_stress_tensor[i] @ (self.get_volume(j) * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i))

    @ti.func
    def compute_basis_stress_div_b_helper(self, i, j, sum: ti.template()):
        x_ij = self.ps.position[i] - self.ps.boundary_particles[j]
        L_i = self.ps.correction_matrix[i]   
        sum += self.ps.boundary_particles_volume[j] * cubic_kernel_derivative_corrected(x_ij, self.ps.smoothing_radius, L_i)

    @ti.kernel
    def compute_lhs(self):
        for i in ti.grouped(self.ps.position):
            self.basis_grad[i] = self.compute_basis_gradient(i)
            if ti.math.isnan(self.basis_grad[i]).any():
                print(f"GOT {self.basis_grad[i]} basis grad for {i}")
        for i in ti.grouped(self.ps.position):
            self.compute_basis_stress_tensor(i)
            stress_div = self.compute_basis_stress_div(i)
            self.lhs[i] = self.basis_vec[i] - (self.deltaTime / self.ps.density[i]) * stress_div
            if ti.math.isnan(self.lhs[i]).any():
                print(f"GOT {self.lhs[i]} lhs for {i}:basis_vec {self.basis_vec[i]}, stress_div {stress_div}")

    @ti.kernel
    def reset_aG(self):
        for i in ti.grouped(self.ps.position):
            self.a_G_ti[i] = ti.Vector.zero(float, self.ps.dim)
            self.a_G_ti_new[i] = ti.zero(float, 3)
    
@ti.data_oriented
class MyLinearOperator(LinearOperator):
    def __init__(self, es:ElasticSolver):
        super().__init__(shape=(es.ps.num_particles, es.ps.num_particles), dtype=float)
        self.es = es
        

    @ti.kernel
    def _matvec(self, x:ti.template(), Ax:ti.template()):
        
        for i in ti.grouped(self.es.ps.position):
            self.es.basis_vec[i].x = x[i,0]
            self.es.basis_vec[i].y = x[i,1]
            self.es.basis_vec[i].z = x[i,2]

        for i in ti.grouped(self.es.ps.position):
            self.es.basis_grad[i] = self.es.compute_basis_gradient(i)
            if ti.math.isnan(self.es.basis_grad[i]).any():
                print(f"GOT {self.es.basis_grad[i]} basis grad for {i}")
        for i in ti.grouped(self.es.ps.position):
            self.es.compute_basis_stress_tensor(i)
            stress_div = self.es.compute_basis_stress_div(i)
            self.es.lhs[i] = self.es.basis_vec[i] - (self.es.deltaTime / self.es.ps.density[i]) * stress_div
            if ti.math.isnan(self.es.lhs[i]).any():
                print(f"GOT {self.es.lhs[i]} lhs for {i}:basis_vec {self.es.basis_vec[i]}, stress_div {stress_div}")

        for i in ti.grouped(self.es.ps.position):
            Ax[i,0] = self.es.lhs[i].x
            Ax[i,1] = self.es.lhs[i].y
            Ax[i,2] = self.es.lhs[i].z




# this is the linear operator that scipy will use to solve the Bi-CGSTAB
# this function will need to set the basis vector
# input v is (3 * N, 1), we need to set it to N by 3 (reshape) 
# and then populate self.basis vector with that information
def linop_numpy(v, es : ElasticSolver):
    es.basis_vec.from_numpy(v.reshape([es.ps.num_particles, 3]).astype(np.float32))
    es.compute_lhs()
    return es.lhs.to_numpy().reshape([3 * es.ps.num_particles,])

def solve_numpy(es: ElasticSolver, dt:float):
    es.deltaTime = dt
    es.compute_rhs()
    b = es.rhs.to_numpy().reshape([3 * es.ps.num_particles,])
    x0 = es.a_G_ti.to_numpy().reshape([3 * es.ps.num_particles,])
    A = LinearOperator(shape=(3 * es.ps.num_particles, 3 * es.ps.num_particles), matvec=lambda x: linop_numpy(x, es))
    
    a_G, exit_code =  bicgstab(A=A, b=b, x0= x0, maxiter=5000, tol=1e-5)
    if exit_code >= 0:
        a_G = a_G.reshape([es.ps.num_particles, 3])
        es.a_G_ti.from_numpy(a_G.astype(np.float32))
    else:
        print("BiCGSTAB failed:", exit_code)
        print(f"RHS {b}")
        # raise ValueError("BiCGSTAB failed")
        es.reset_aG()

    return exit_code
@ti.kernel
def init_new_scalar_field(new_field : ti.template(), old_field : ti.template()):
    for i, j in new_field:
        new_field[i, 0] = old_field[i].x
        new_field[i, 1] = old_field[i].y
        new_field[i, 2] = old_field[i].z

@ti.kernel
def init_new_vector_field(new_field : ti.template(), old_field : ti.template()):
    for i, j in old_field:
      new_field[i].x =   old_field[i, 0] 
      new_field[i].y =   old_field[i, 1] 
      new_field[i].z =   old_field[i, 2] 

def solve_taichi(es:ElasticSolver, dt:float):
    es.deltaTime = dt
    es.compute_rhs()
    A = MyLinearOperator(es)
    b = ti.field(ti.f32, shape=(es.ps.num_particles, 3))
    init_new_scalar_field(b, es.rhs)
    res = ti.linalg.MatrixFreeBICGSTAB(A, b, es.a_G_ti_new, tol=1e-06, maxiter=5000, quiet=False)
    if res:
        init_new_vector_field(es.a_G_ti, es.a_G_ti_new)
    return (res * 1)



    