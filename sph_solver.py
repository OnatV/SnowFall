import taichi as ti
import numpy as np

from taichi.math import vec2, vec3, mat3
from particle_system import ParticleSystem
from pressure_solver import PressureSolver
from kernels import cubic_kernel, cubic_kernel_derivative

@ti.func
def mult_scalar_matrix(c:float, A: mat3 ):
    res = ti.Matrix.zero(float, 3, 3)
    for i in range(3):
        for j in range(3):
            res[i, j] = c * A[i, j]
    return res

@ti.data_oriented
class SnowSolver:
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        self.time = 0
        self.snow_implemented = True
        # self.a_lambda = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.a_G = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.a_other = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.wind_enabled = ps.enable_wind
        self.numerical_eps = 1e-6
        self.m_psi = 10
        # self.init_kernel_lookup()
        # TO DO: COMPUTE ADAPTIVE CORRECTION FACTORR
        self.gamma_1 = ti.field(float, shape=self.ps.num_particles)
        self.gamma_2 = ti.field(float, shape=self.ps.num_particles)

    @ti.func
    def helper_sum_kernel(self, i, j, sum:ti.template()):
        sum += cubic_kernel(
            (self.ps.position[i] - self.ps.position[j]).norm(), self.ps.smoothing_radius
        )

    @ti.func
    def helper_sum_b_kernel(self, i, j, sum:ti.template()):
        sum += cubic_kernel(
            (self.ps.position[i] - self.ps.boundary_particles[j]).norm(), self.ps.smoothing_radius
        )

    @ti.func
    def helper_sum_gradient(self, i, j, sum:ti.template()):
        sum += cubic_kernel_derivative(
            (self.ps.position[i] - self.ps.position[j]), self.ps.smoothing_radius
        )

    @ti.func
    def helper_sum_b_gradient(self, i, j, sum:ti.template()):
        sum += cubic_kernel_derivative(
            (self.ps.position[i] - self.ps.boundary_particles[j]), self.ps.smoothing_radius
        )

    @ti.func
    def compute_gamma_1(self, i):
        invVi = 1.0 / self.get_volume(i)
        kernel_sum = 0.0
        self.ps.for_all_neighbors(i, self.helper_sum_kernel, kernel_sum)
        boundary_kernel_sum = 0.0
        self.ps.for_all_b_neighbors(i, self.helper_sum_b_kernel, boundary_kernel_sum)
        self.gamma_1[i] = (invVi - kernel_sum) / boundary_kernel_sum

    @ti.func
    def compute_gamma_2(self, i):
        kernel_sum = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_gradient, kernel_sum)
        boundary_kernel_sum = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_b_neighbors(i, self.helper_sum_b_gradient, boundary_kernel_sum)
        self.gamma_2[i] = -(kernel_sum.dot(boundary_kernel_sum)) / boundary_kernel_sum.dot(boundary_kernel_sum)


    @ti.kernel
    def compute_bounary_correction_factor(self):
        for i in ti.grouped(self.gamma_1):
            self.compute_gamma_1(i)
            self.compute_gamma_2(i)
        print("gamma_1", self.gamma_1[0])
        print("gamma_2", self.gamma_2[0])

    @ti.func
    def helper_boundary_volume(self, i, j, sum: ti.template()):
        sum += cubic_kernel((self.ps.boundary_particles[i] - self.ps.boundary_particles[j]).norm(), self.ps.smoothing_radius)

    @ti.func
    def compute_b_particle_volume(self, i):
        kernel_sum = 0.0
        self.ps.for_all_neighbors_b_grid(i, self.helper_boundary_volume, kernel_sum)
        self.ps.boundary_particles_volume[i] = (1.0 / kernel_sum) # which is it? Gissler et al v4 does not include h, but v1 does!!

    @ti.kernel
    def compute_boundary_volumes(self):
        correction = 1.0
        for i in range(self.ps.num_b_particles):
            kernel_sum = 0.0
            for j in range(self.ps.num_b_particles):
                if i == j: continue
                if (self.ps.boundary_particles[i] - self.ps.boundary_particles[j]).norm() > self.ps.smoothing_radius: continue
                kernel_sum += cubic_kernel((self.ps.boundary_particles[i] - self.ps.boundary_particles[j]).norm(), self.ps.smoothing_radius)
            self.ps.boundary_particles_volume[i] = 0.8 * self.ps.boundary_particle_radius ** 3 * (1.0 / kernel_sum)
            # self.ps.boundary_particles_volume[i] = correction * (1.0 / kernel_sum) # which is it? Gissler et al v4 does not include h, but v1 does!!
            

    # def init_kernel_lookup(self, table_size = 100, grad_table_size = 100):
    #     self.kernel_table = ti.field(dtype=float, shape=table_size)
    #     self.grad_kernel_table = ti.Vector.field(dtype=float, n=3, shape=grad_table_size)
    #     dh = self.ps.smoothing_radius / table_size
    #     grad_dh = self.ps.smoothing_radius / table_size
    #     @ti.kernel
    #     def set_values(): 
    #         for i in range(table_size):
    #             r = i * dh
    #             self.kernel_table[i] = cubic_kernel(r) 
    #         for i in range(table_size):
    #             r = i * grad_dh
    #             tmp = cubic_kernel_derivative(ti.Vector([r, 0.0, 0.0]))
    #             self.grad_kernel_table[i] = tmp.x
    #     set_values()

    
    # gives an approixmation self.of W(r), r = |xj - xi|
    # given the table of precomputed values
    # interpolation is.. nearest neighbor
    # assume r_norm >= 0
    # @ti.func
    # def kernel_lookup(self, r_norm):
    #     tsize = self.kernel_table.shape[0]
    #     h = self.ps.smoothing_radius
    #     dh = h / tsize
    #     result = ti.f32(0.0)
    #     if (r_norm >= h):
    #         result = 0
    #     else:
    #         i = ti.i32(ti.floor(r_norm / dh))
    #         result = self.kernel_table[i]
    #     return result
    # # kernel_table[i] is W(i*dh)
    # # W(r_rnorm) is needed
    # # -> rnorm = i*dh
    # # rnorm / dh == i

    # @ti.func
    # def grad_kernel_lookup(self, r:vec3) -> vec3:
    #     r_norm = r.norm()
    #     tsize = self.kernel_table.shape[0]
    #     h = self.ps.smoothing_radius
    #     dh = h / tsize
    #     result = vec3(0.0, 0.0, 0.0)
    #     if (r_norm >= h):
    #         pass
    #     else:
    #         i = ti.i32(ti.floor(r_norm / dh))
    #         grad_magnitude = self.grad_kernel_table[i]
    #         grad_dir = r / r_norm
    #         result = grad_magnitude * grad_dir
    #     return result

    @ti.kernel
    def enforce_boundary_3D(self):
        for i in range(self.ps.num_particles):
            if self.ps.position[i].x < self.ps.domain_start[0]:
                self.ps.position[i].x = self.ps.domain_start[0] + self.ps.padding
            if self.ps.position[i].y < self.ps.domain_start[1]:
                self.ps.position[i].y = self.ps.domain_start[1] + self.ps.padding
            if self.ps.position[i].z < self.ps.domain_start[2]:
                self.ps.position[i].z = self.ps.domain_start[2] + self.ps.padding
            if self.ps.position[i].x > self.ps.domain_end[0]:
                self.ps.position[i].x = self.ps.domain_end[0] - self.ps.padding
            if self.ps.position[i].y > self.ps.domain_end[1]:
                self.ps.position[i].y = self.ps.domain_end[1] - self.ps.padding
            if self.ps.position[i].z > self.ps.domain_end[2]:
                self.ps.position[i].z = self.ps.domain_end[2] - self.ps.padding


    @ti.kernel
    def simple_gravity_accel(self):
        #   simple gravity acceleration
        for i in range(self.ps.num_particles):
            self.ps.acceleration[i] = self.ps.gravity
    
    @ti.kernel
    def integrate_velocity(self, deltaTime: float):
        '''
            Computes step 8-9 from Algorithm 1 in the paper.
        '''

        for i in range(self.ps.num_particles):
            self.ps.velocity[i] = self.ps.velocity[i] + (deltaTime * self.ps.acceleration[i])

    @ti.kernel
    def update_position(self, deltaTime: float):
        for i in range(self.ps.num_particles):
            # self.ps.position_0[i] = self.ps.position[i]
            self.ps.position[i] = self.ps.position[i] + deltaTime * self.ps.velocity[i]

    

    @ti.func
    def compute_lame_parameters(self,i):
        '''
            Section 3.3.2
        '''
        # to uncomment later :)
        young_mod = 140_000
        xi = 10
        nu = 0.2
        numerator = young_mod * nu
        denom = (1 + nu) * (1 - 2.0*nu)
        p0_t = self.ps.rest_density[i]
        # what should p_0 be?
        p_0 = self.ps.init_density
        k = numerator / denom
        self.ps.lambda_t_i[i] = k * ti.exp(xi * (self.ps.rest_density[i] - p_0) / self.ps.rest_density[i])
        # self.ps.lambda_t_i[i] = 100_000

    @ti.func
    def compute_rest_density(self, i):
        '''
            Step 2 in Algorithm 1 in the paper.
            
            Computes ro_0_i^t, the rest density of particle i at time t. 
        '''
        # first the density is computed, then
        # the rest density is derived        
        detF = ti.abs(ti.Matrix.determinant(self.ps.deformation_gradient[i]))
        density_i = 0.0
        self.ps.for_all_neighbors(i, self.calc_density, density_i)
        self.ps.density[i] = density_i
        self.ps.rest_density[i] = self.ps.density[i] * detF
        # rest density does not need boundary neighbors
        # self.ps.for_all_b_neighbors(i, self.calc_density_b, density_i)
        # self.ps.density[i] = density_i
        if i[0] == 0:
            print("density", density_i)

    @ti.func
    def calc_density(self, i_idx, j_idx, d:ti.template()):
        rnorm = ti.Vector.norm(self.ps.position[i_idx] - self.ps.position[j_idx])
        # d +=  cubic_kernel(rnorm) * ti.cast(self.ps.m_k, ti.f32)
        d += cubic_kernel(rnorm, self.ps.smoothing_radius) * self.ps.m_k

    @ti.func
    def calc_density_b(self, i_idx, j_idx, d:ti.template()):
        rnorm = ti.Vector.norm(self.ps.position[i_idx] - self.ps.boundary_particles[j_idx])
        # d +=  cubic_kernel(rnorm) * ti.cast(self.ps.m_k, ti.f32)
        # d += self.gamma_1[i_idx] * cubic_kernel(rnorm, self.ps.smoothing_radius) * self.ps.m_k
        d += cubic_kernel(rnorm, self.ps.smoothing_radius) * self.ps.boundary_particles_volume[j_idx] * self.ps.rest_density[i_idx]

    #calculate V_i = m_i / density_i
    @ti.func
    def get_volume(self, i):
        return (self.ps.m_k / ti.math.max(self.ps.density[i], self.numerical_eps ) )
        # return self.ps.particle_radius ** 3

    @ti.func
    def helper_a_lambda_fluid_neighbors(self, i, j, sum:ti.template()):
        Vj = self.get_volume(j)
        density_i = self.ps.density[i] / self.ps.rest_density[i]
        density_i2 = density_i * density_i
        dpi = self.ps.pressure[i] / (self.ps.rest_density[i] * density_i2)
        density_j = self.ps.density[j] / self.ps.rest_density[j]
        density_j2 = density_j * density_j
        dpj = (self.ps.pressure[j] / self.ps.rest_density[j]) / density_j2
        sum -= Vj * (dpi + self.ps.rest_density[j] / self.ps.rest_density[i] * dpj) * cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j], self.ps.smoothing_radius
        )

    @ti.func 
    def helper_a_lambda_b(self, i, j, sum: ti.template()):
        density_i = self.ps.density[i]
        density_i2 = density_i * density_i
        dpi = self.ps.pressure[i] / (density_i2)
        a = self.ps.rest_density[i] * self.ps.boundary_particles_volume[j] * dpi * cubic_kernel_derivative(
            self.ps.position[i] - self.ps.boundary_particles[j], self.ps.smoothing_radius
        )
        sum -= a
    
    @ti.kernel
    def compute_a_lambda(self, success : ti.template()):
        for i in ti.grouped(self.ps.position):
            a_lambda = ti.Vector([0.0, 0.0, 0.0])
            if not success or \
            ti.math.isnan(self.ps.rest_density[i]) or \
            ti.math.isnan(self.ps.pressure[i]) or \
            self.ps.rest_density[i] == 0.0:
                a_lambda = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.ps.for_all_neighbors(i, self.helper_a_lambda_fluid_neighbors, a_lambda)
                self.ps.for_all_b_neighbors(i, self.helper_a_lambda_b, a_lambda)
                
                # a_lambda = 1.0 / self.ps.density[i] * self.ps.pressure_gradient[i]
            self.ps.acceleration[i] += a_lambda
            if i[0] == 0:
                print("a_lambda", a_lambda)
    
    @ti.func
    def nan_check(self) -> bool:
        has_nan = False
        for i in range(self.ps.num_particles):
            if (ti.math.isnan(self.ps.pressure[i])) or \
                ti.Vector.any(ti.math.isnan(self.ps.pressure_gradient[i])) or \
                ti.Vector.any(ti.math.isnan(self.ps.pressure_laplacian[i])) or \
                ti.Vector.any(ti.math.isnan(self.ps.jacobian_diagonal[i])) or \
                ti.Vector.any(ti.math.isnan(self.ps.p_star[i])) or \
                ti.Vector.any(ti.math.isnan(self.ps.rest_density[i])):
                has_nan = True
        return has_nan
        
    
    def solve_a_lambda(self, deltaTime):
        '''
            Computes Step 6 in Algorithm 1 in the paper.

        '''
        pressure_solver = PressureSolver(self.ps)
        success = pressure_solver.solve(deltaTime)
        self.compute_a_lambda(success)

    @ti.func
    def aux_correction_matrix(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix
        '''
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx] # x_ij: vec3
        w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius)
        v_j = self.get_volume(j_idx)

        res += (v_j * w_ij).outer_product(x_ij)

    @ti.func
    def compute_correction_matrix(self, i):
        '''
            Step 3 in Algorithm 1 in the paper.

            Computes L_i as defined in the paper. 
            
            If os.is_pseudo_L_i[i] is true, 
                use L_i = pseudo_correction_matrix[i]
            else
                use L_i = correction_matrix[i]
        '''
        x_i = self.ps.position[i]
        self.ps.is_pseudo_L_i[i] = False
        tmp_i = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_neighbors(i, self.aux_correction_matrix, tmp_i)

        det = ti.Matrix.determinant(tmp_i)
        if det != 0: 
            self.ps.correction_matrix[i] = tmp_i.inverse()
        else: # no inverse 
              # hence use the peudoinverse
              # other code can use is_pseudo property
              # if true, the kernel_grad Wij must be transformed to 
              # tmp_i.transpose() * W_ij, and then
              # ~grad = pseudo_inv * tmp_i.transpose() * W_ij
            pseudo = (tmp_i.transpose() * tmp_i).inverse()
            self.ps.pseudo_correction_matrix[i] = pseudo
            self.ps.correction_matrix[i] = tmp_i
            self.ps.is_pseudo_L_i[i] = True

    @ti.func
    def compute_accel_ext(self, i):
        '''
            Computes Step 4 in Algorithm 1 in the paper.

            Computes a_{i}^{other,t}, the acceleration due to gravity, adhesion, and external forces.
        '''

        ##Wind Strength equal to the position of the particle in the direction
        wind_acc = ti.Vector([0.0, 0.0, 0.0])
        if ti.static(self.wind_enabled):
            wind_acc += self.ps.wind_direction * self.ps.position[i].dot(self.ps.wind_direction)

        # remove gravity for now
        self.ps.acceleration[i] = self.ps.gravity 

    @ti.func
    def compute_flow(self, i):
        pass

    @ti.func
    def compute_accel_friction(self, i):
        pass

    @ti.kernel
    def compute_external_forces_only(self, deltaTime:float):
        for i in range(self.ps.num_particles):
            self.compute_accel_ext(i)

    @ti.kernel
    def compute_internal_forces(self):
        '''
            Computes for loop 1 in Algorithm 1 in the paper.
                Steps:
                    Step 2 : self.compute_rest_density(i)
                    Step 3 : self.compute_correction_matrix(i)
                    Step 4 : self.compute_accel_ext(i)
                    Step 5 : self.compute_accel_friction(i)
        '''
        rest_density_sum = 0.0
        for i in ti.grouped(self.ps.position):
            self.compute_rest_density(i) #Step 2
            self.compute_lame_parameters(i) ##Don't get what this is doing
            self.compute_correction_matrix(i) #Step 3
            self.compute_accel_ext(i) #Step 4
            self.compute_accel_friction(i) #Step 5
            rest_density_sum += self.ps.rest_density[i]
        self.ps.avg_rest_density[0] = rest_density_sum / self.ps.num_particles


    @ti.kernel
    def integrate_deformation_gradient(self, deltaTime:float):
        '''
            Calculates steps 10-11 in Algorithm 1 in the paper.
        '''

        for i in ti.grouped(self.ps.position):
            
            vel_grad = self.compute_velocity_gradient(i) 
            un_clamped = self.ps.deformation_gradient[i] + deltaTime * vel_grad * self.ps.deformation_gradient[i]
            self.ps.deformation_gradient[i] = self.clamp_deformation_gradients(un_clamped)

    @ti.func
    def compute_velocity_gradient(self,i):


        ##Currently only computes the gradient using snow particles, ie no boundary Eq.17
        grad_v_i_s_prime = ti.Matrix.zero(dt=float, n=3, m=3)
        self.ps.for_all_neighbors(i, self.helper_compute_velocity_gradient_uncorrected, grad_v_i_s_prime)
        grad_v_i_s_tilde = ti.Matrix.zero(dt=float, n=3, m=3)
        
        V_i_prime = grad_v_i_s_prime.trace() * ti.Matrix.identity(float, 3) / 3
        R_i_tilde = (grad_v_i_s_tilde - grad_v_i_s_tilde.transpose()) / 2 
        S_i_tilde = (grad_v_i_s_tilde + grad_v_i_s_tilde.transpose()) / 2 - grad_v_i_s_tilde.trace() * ti.Matrix.identity(float, 3) / 3

        return V_i_prime + R_i_tilde + S_i_tilde
    
    @ti.func
    def helper_compute_velocity_gradient_corrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. 
            Needs to be separated for boundary in snow in the future.
        '''
        v_j = self.ps.velocity[j_idx]
        v_i = self.ps.velocity[i_idx]
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx]
        V_j = self.get_volume(j_idx)

        if self.ps.is_pseudo_L_i[i_idx]:
            L_i = self.ps.pseudo_correction_matrix[i_idx]
        else:
            L_i = self.ps.correction_matrix[i_idx]

        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius)
        corrected_del_w_ij = L_i @ del_w_ij

        res += (v_j - v_i).outer_product(V_j * corrected_del_w_ij)

    @ti.func
    def helper_compute_velocity_gradient_uncorrected(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix. 
            Needs to be separated for boundary in snow in the future.
        '''
        v_j = self.ps.velocity[j_idx] # v_j: vec3
        v_i = self.ps.velocity[i_idx] # v_i: vec3
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx] # x_ij: vec3
        del_w_ij = cubic_kernel_derivative(x_ij, self.ps.smoothing_radius) # del_w_ij: vec3
        V_j = self.get_volume(j_idx) # V_j: float

        res += (v_j - v_i).outer_product(V_j * del_w_ij)

    @ti.func
    def clamp_deformation_gradients(self, matrix):
        U, S, V = ti.svd(matrix)
        S = ti.math.clamp(S, self.ps.theta_clamp_c, self.ps.theta_clamp_s)
        return V @ S @ V.transpose() ## This supposedly removes the rotation part
    

    def substep(self, deltaTime):
        # from Gissler et al paper (An Implicit Compressible SPH Solver for Snow Simulation)
        # pseudocode for a single simulation step in SPH snow solver:
        # 
        # foreach particle i do: (see self.compute_internal_forces)
        #   compute p_{0,i}^t (rest density at time t, Section 3.3.2)
        #   compute L_t (correction matrix, Eq 15 in paper)
        #   compute a_{i}^{other,t} (acceleration due to gravity, adhesion, and ext forces)
        #   compute a_{i}^{friction,t} (accerleration due to friction and boundary, eq 24)
        # solve for a_i^lambda (acceleration due to elastic deformation, subsection 3.2.1) (see self.solve_a_lambda)
        # solve for a_i^G (acceleration due to elastic deformation, subsection 3.2.2) (see self.solve_a_G)
        # foreach particle i do (see self.integrate_velocity)
        #   integrate velocity v
        # foreach particle i do (see self.integrate_deformation_gradient)
        #   integrate and store deformation gradient F, Subsection 3.3.1
        # foreach particle i do
        #   integrate positison x (see self.update_position)
        # self.ps.gravity = set to zero
        if ti.static(self.snow_implemented):
            # these functions should update the acceleration field of the particles
            self.compute_internal_forces() # Step 1, includes Steps 2-5
            # print("before solve a")
            self.solve_a_lambda(deltaTime) # Step 6
            # self.solve_a_G()             #Step 7 
            self.integrate_velocity(deltaTime) # Step 8-9
            self.integrate_deformation_gradient(deltaTime) #Step 10-11

        else:
            self.compute_external_forces_only(deltaTime)
            self.integrate_velocity(deltaTime)
        # these last steps are the same regardless of solver type
        self.update_position(deltaTime)
        # print("Step")

    def step(self, deltaTime, time):
        self.ps.update_grid()
        
        if time == 0.0:
            self.compute_boundary_volumes()
        self.compute_bounary_correction_factor()
        # self.ps.cumsum.run(self.ps.grid_particles_num)
        # self.ps.cumsum_indx()
        # self.ps.sort_particles()
        # step physics
        # print("before updating the grid")
        # self.ps.update_grid()
        # print("before substep")
        self.substep(deltaTime)
        # enforce the boundary of the domain (and later rigid bodies)
        self.enforce_boundary_3D()
        self.ps.color_neighbors(0, ti.Vector([1.0, 0.0, 0.0]))
        # self.ps.color_neighbors(9, ti.Vector([0.0, 1.0, 0.0]))
        # self.ps.color_neighbors(99, ti.Vector([1.0, 5.0, 0.0]))
        # self.ps.color_neighbors(90, ti.Vector([0.0, 0.0, 1.0]))
        # self.ps.color_neighbors(909, ti.Vector([1.0, 0.0, 1.0]))
        # self.ps.color_neighbors(900, ti.Vector([0.5, 0.5, 1.0]))
        # self.ps.color_neighbors(990, ti.Vector([0.0, 1.0, 1.0]))
        # self.ps.color_neighbors(999, ti.Vector([1.0, 0.0, 0.5]))
        self.ps.color_b_neighbors(0, ti.Vector([1.0, 0.0, 1.0]))
        # update time
        self.time += deltaTime
        # print(self.ps.position[0])
        print("Step", self.ps.position[0][1] - 0.35)

    