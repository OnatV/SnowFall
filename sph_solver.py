import taichi as ti
import numpy as np

from taichi.math import vec2, vec3, mat3
from particle_system import ParticleSystem

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
        self.init_kernel_lookup()


    @ti.kernel
    def compute_boundary_volumes(self):
        for i in ti.grouped(self.ps.boundary_particles):
            kernel_sum = 0.0
            for j in range(self.ps.nBoundaryParticles):
                if i[0] == j: continue
                if (self.ps.boundary_particles[i] - self.ps.boundary_particles[j]).norm() > self.ps.smoothing_radius: continue
                kernel_sum += self.cubic_kernel((self.ps.boundary_particles[i] - self.ps.boundary_particles[j]).norm())
            self.ps.boundary_particles_volume[i] = 0.8 * self.ps.bgrid_x ** 3 * (1.0 / kernel_sum)

    def init_kernel_lookup(self, table_size = 100, grad_table_size = 100):
        self.kernel_table = ti.field(dtype=float, shape=table_size)
        self.grad_kernel_table = ti.Vector.field(dtype=float, n=3, shape=grad_table_size)
        dh = self.ps.smoothing_radius / table_size
        grad_dh = self.ps.smoothing_radius / table_size
        @ti.kernel
        def set_values(): 
            for i in range(table_size):
                r = i * dh
                self.kernel_table[i] = self.cubic_kernel(r) 
            for i in range(table_size):
                r = i * grad_dh
                tmp = self.cubic_kernel_derivative(ti.Vector([r, 0.0, 0.0]))
                self.grad_kernel_table[i] = tmp.x
        set_values()

    
    # gives an approixmation self.of W(r), r = |xj - xi|
    # given the table of precomputed values
    # interpolation is.. nearest neighbor
    # assume r_norm >= 0
    @ti.func
    def kernel_lookup(self, r_norm):
        tsize = self.kernel_table.shape[0]
        h = self.ps.smoothing_radius
        dh = h / tsize
        result = ti.f32(0.0)
        if (r_norm >= h):
            result = 0
        else:
            i = ti.i32(ti.floor(r_norm / dh))
            result = self.kernel_table[i]
        return result
    # kernel_table[i] is W(i*dh)
    # W(r_rnorm) is needed
    # -> rnorm = i*dh
    # rnorm / dh == i

    @ti.func
    def grad_kernel_lookup(self, r:vec3) -> vec3:
        r_norm = r.norm()
        tsize = self.kernel_table.shape[0]
        h = self.ps.smoothing_radius
        dh = h / tsize
        result = vec3(0.0, 0.0, 0.0)
        if (r_norm >= h):
            pass
        else:
            i = ti.i32(ti.floor(r_norm / dh))
            grad_magnitude = self.grad_kernel_table[i]
            grad_dir = r / r_norm
            result = grad_magnitude * grad_dir
        return result

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

    @ti.func
    def cubic_kernel(self, r_norm):
        # implementation details borrowed from SPH_Taichi
        # use ps.smoothing_radius to calculate the kernel weight of particles
        # for now, sum over nearby particles
        w = ti.cast(0.0, ti.f32)
        h = self.ps.smoothing_radius
        k = 8 / np.pi
        k /= ti.pow(h, self.ps.dim)
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = ti.pow(q, 2)
                q3 = ti.pow(q, 3)
                w = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                w = k * 2 * ti.pow(1 - q, 3.0)
        return w
    
    @ti.func    
    def cubic_kernel_derivative(self, r:vec3) -> vec3:
        # use ps.smoothing_radius to calculate the derivative of kernel weight of particles
        # for now, sum over nearby particles
        h = self.ps.smoothing_radius
        k = 8 / np.pi
        k = 6.0 * k / ti.pow(h, self.ps.dim)
        r_norm = r.norm()
        q = r_norm / h
        d_w = ti.Vector([0.0, 0.0, 0.0])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q < 0.5:
                d_w = k * q * (3.0 * q - 2.0) * grad_q
            else:
                f = 1.0 - q
                d_w = k * (-f * f) * grad_q
        return d_w

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
    def calc_density(self, i_idx, j_idx, d:ti.template()):
        rnorm = ti.Vector.norm(self.ps.position[i_idx] - self.ps.position[j_idx])
        d +=  self.cubic_kernel(rnorm) * ti.cast(self.ps.m_k, ti.f32)

    @ti.func
    def compute_lame_parameters(self,i):
        '''
            Section 3.3.2
        '''
        young_mod = 20_000
        xi = 10
        nu = 0.2
        numerator = young_mod * nu
        denom = (1 + nu) * (1 - 2*nu)
        p0_t = self.ps.rest_density[i]
        # what should p_0 be?
        # p_0 = 1000
        k = numerator / denom
        self.ps.lambda_t_i[i] = k * ti.exp(xi * 1)
        # self.ps.lambda_t_i[i] = 200

    @ti.func
    def compute_rest_density(self, i):
        '''
            Step 2 in Algorithm 1 in the paper.
            
            Computes ro_0_i^t, the rest density of particle i at time t. 
        '''
        # first the density is computed, then
        # the rest density is derived        
        density_i = ti.Vector([0.0])
        self.ps.for_all_neighbors(i, self.calc_density, density_i)
        self.ps.density[i] = density_i
        # old code
        #for j in range(self.ps.num_particles):
        #    w_ij = self.kernel_lookup(ti.Vector.norm(x_i - self.ps.position[j]))
        #    self.ps.density[i] += self.ps.m_k * w_ij

        detF = ti.Matrix.determinant(self.ps.deformation_gradient[i])
        self.ps.rest_density[i] = self.ps.density[i] * ti.abs(detF)
        
    #calculate V_i = m_i / density_i
    @ti.func
    def get_volume(self, i):
        return (self.ps.m_k / ti.math.max(self.ps.density[i], self.numerical_eps ) )[0]

    @ti.func
    def divergence_discretization(self, i, k, sum:ti.template()):
        sum += self.get_volume(k) * (self.ps.velocity_star[k] - self.ps.velocity_star[i]).dot(self.cubic_kernel_derivative(self.ps.position[i]-self.ps.position[k]))

    # consider: https://en.wikipedia.org/wiki/Jacobi_method
    @ti.func
    def compute_A_p(self, i, deltaTime, density_error:ti.template()):
        deltaTime2 = deltaTime * deltaTime
        self.ps.pressure_laplacian[i] = 0.0
        # grad_p = self.ps.pressure_gradient[i]
        lp_i = 0.0
        self.ps.for_all_neighbors(i, self.helper_diff_of_pressure_grad, lp_i)
        lp2 = 0.0
        self.ps.for_all_b_neighbors(i, self.helper_diff_of_pressure_grad_b, lp2)
        self.ps.pressure_laplacian[i] = lp_i + 1.5 * lp2
        # now compute Ap
        A_p = -self.ps.rest_density[i] / self.ps.lambda_t_i[i] * self.ps.pressure[i] + deltaTime2 * lp_i
        aii = self.ps.jacobian_diagonal[i]
        residuum = self.ps.rest_density[i] - self.ps.p_star[i] - A_p
        # self.ps.density_error[i] = -residuum
        pi = (0.5 / ti.math.max(aii, self.numerical_eps))  * residuum
        self.ps.pressure[i] += pi[0]
        density_error -= residuum

    @ti.func
    def helper_diff_of_pressure_grad(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * (self.ps.pressure_gradient[j] - self.ps.pressure_gradient[i]).dot(self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j])
        )

    @ti.func
    def helper_diff_of_pressure_grad_b(self, i, b, sum:ti.template()):
        sum += (self.ps.boundary_particles_volume[b] * (self.ps.pressure_gradient[i]).dot(self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.boundary_particles[b])
        ))[0]


    @ti.func
    def compute_pressure_gradient(self, i):
        self.ps.pressure_gradient[i] = 0.0
        sum_of_pressures = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_of_pressure, sum_of_pressures)
        sum_of_b = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_b_neighbors(i, self.helper_sum_over_b, sum_of_b)
        self.ps.pressure_gradient[i] = sum_of_pressures + 1.5 * self.ps.pressure[i] * sum_of_b


    @ti.func
    def helper_sum_of_pressure(self, i, j, sum:ti.template()):
        sum += (self.ps.pressure[j] + self.ps.pressure[i]) * self.get_volume(j) * self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j]
        )

    @ti.func
    def helper_volume_squared_sum(self, i, j, sum: ti.template()):
        sum += self.get_volume(i) * self.get_volume(j) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j]).norm_sqr()
    
    @ti.func
    def helper_sum_over_j(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j])

    @ti.func
    def helper_sum_over_b(self, i, b, sum:ti.template()):
        sum += self.ps.boundary_particles_volume[b] * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.boundary_particles[b])

    @ti.func
    def helper_sum_over_k(self, i, k, sum:ti.template()):
        sum += self.get_volume(k) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[k])
        
    @ti.func
    def compute_jacobian_diagonal_entry(self, i, deltaTime):
        psi = 1.5 # this is a parameter that was set in the paper
        p_lame =  -self.ps.rest_density[i] / self.ps.lambda_t_i[i]        
        volume_squared_sum = 0.0
        self.ps.for_all_neighbors(i, self.helper_volume_squared_sum, volume_squared_sum)
        sum_over_j = ti.Vector([0.0,0.0,0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_over_j, sum_over_j)
        sum_over_k = ti.Vector([0.0,0.0,0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_over_k, sum_over_k)
        self.ps.for_all_b_neighbors(i, self.helper_sum_over_b, sum_over_k)
        sum_over_b = ti.Vector([0.0,0.0,0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_over_b, sum_over_b)
        deltaTime2 = deltaTime * deltaTime
        self.ps.jacobian_diagonal[i] = p_lame - deltaTime2 * volume_squared_sum - deltaTime2 * (sum_over_j + 1.5 * sum_over_b).dot(sum_over_k)

    @ti.kernel
    def implicit_solver_prepare(self, deltaTime: float):
        '''
            Computes Step 1-4 in Algorithm 2 in the paper.
        '''
        #compute sph discretization using eq 6
        # need to find predicted velocity but that can be done later
        # print("Here")
        for i in ti.grouped(self.ps.position):
            # print(i)
            self.ps.p_star[i] = 0
            self.ps.pressure[i] = 0
            self.ps.pressure[i] = 0
            self.ps.jacobian_diagonal[i] = 0
            self.ps.velocity_star[i] = self.ps.velocity[i] + deltaTime * self.ps.acceleration[i] ##Replace LATER@@ Acceleration includes aother and a friction

        for i in ti.grouped(self.ps.position):
            velocity_div = 0.0
            self.ps.for_all_neighbors(i, self.divergence_discretization, velocity_div)
            self.ps.p_star[i] = self.ps.density[i] - deltaTime * self.ps.density[i] * velocity_div
            self.compute_jacobian_diagonal_entry(i, deltaTime)

    @ti.kernel
    def implicit_pressure_solver_step(self, deltaTime:float)->ti.f32:
        density_error = ti.Vector([0.0])
        # for i in ti.grouped(self.ps.position):
            # self.ps.pressure[i] = self.ps.pressure[i]
        for i in ti.grouped(self.ps.position):
            self.compute_pressure_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_A_p(i, deltaTime, density_error)
        return density_error / self.ps.num_particles
    
    @ti.kernel
    def compute_a_lambda(self, success : ti.template()):
        for i in ti.grouped(self.ps.position):
            if self.ps.density[i][0] == 0.0:
                continue
            if not success:
                self.ps.pressure_gradient[i] = 0
            self.ps.acceleration[i] += (1.0 / ti.math.max(self.ps.density[i][0], self.numerical_eps)) * self.ps.pressure_gradient[i]
            if i[0] == 0:
                print("a_lambda", (1.0 / ti.math.max(self.ps.density[i][0], self.numerical_eps)) * self.ps.pressure_gradient[i])
    
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
        
    def implicit_pressure_solve(self, deltaTime:float):
        max_iterations = 100
        min_iterations = 10
        is_solved = False
        it = 0
        # print("here")
        avg_density_error_prev = 1000.0 # set it high to avoid early termination
        while ((~is_solved or it < min_iterations) and it < max_iterations):
            avg_density_error = self.implicit_pressure_solver_step(deltaTime)
            # print("-----ITERATION", it,"---------")
            # print("avg_density_error", avg_density_error)
            # print("pressure", self.ps.pressure[0])
            # print("pressure_gradient", self.ps.pressure_gradient[0])
            # print("pressure_gradient_norm", self.ps.pressure_gradient[0].norm())
            if avg_density_error > avg_density_error_prev and it > min_iterations:
                is_solved = None
                break
            if avg_density_error < 0.01:
                is_solved = True 
                # print("----Converged---")
            it = it + 1
            avg_density_error_prev = avg_density_error
            # print("pressure", self.ps.pressure[0])
        return is_solved
    def solve_a_lambda(self, deltaTime):
        '''
            Computes Step 6 in Algorithm 1 in the paper.

        '''
        # print("solve_alam")
        self.implicit_solver_prepare(deltaTime)
        success = self.implicit_pressure_solve(deltaTime)
        self.compute_a_lambda(success)

    @ti.func
    def aux_correction_matrix(self, i_idx, j_idx, res:ti.template()):
        '''
            Helper of self.compute_correction_matrix
        '''
        x_ij = self.ps.position[i_idx] - self.ps.position[j_idx] # x_ij: vec3
        w_ij = self.cubic_kernel_derivative(x_ij)
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
        #ti.loop_config(serialize=True)
        # print("before", self.ps.density[0])
        for i in ti.grouped(self.ps.position):
            self.compute_rest_density(i) #Step 2
            self.compute_lame_parameters(i) ##Don't get what this is doing
            self.compute_correction_matrix(i) #Step 3
            self.compute_accel_ext(i) #Step 4
            self.compute_accel_friction(i) #Step 5
            #print("\r",  i, end="")
        # print("after", self.ps.density[0])

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

        # grad_v_i_s_tilde = ti.Matrix.zero(dt=float, n=3, m=3)
        # self.ps.for_all_neighbors(i, self.helper_compute_velocity_gradient_corrected, grad_v_i_s_tilde)
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

        del_w_ij = self.cubic_kernel_derivative(x_ij)
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
        del_w_ij = self.cubic_kernel_derivative(x_ij) # del_w_ij: vec3
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

    def step(self, deltaTime):
        self.ps.update_grid()
        self.ps.color_neighbors(0, ti.Vector([1.0, 0.0, 0.0]))
        self.ps.color_neighbors(9, ti.Vector([0.0, 1.0, 0.0]))
        self.ps.color_neighbors(99, ti.Vector([1.0, 5.0, 0.0]))
        self.ps.color_neighbors(90, ti.Vector([0.0, 0.0, 1.0]))
        self.ps.color_neighbors(909, ti.Vector([1.0, 0.0, 1.0]))
        self.ps.color_neighbors(900, ti.Vector([0.5, 0.5, 1.0]))
        self.ps.color_neighbors(990, ti.Vector([0.0, 1.0, 1.0]))
        self.ps.color_neighbors(999, ti.Vector([1.0, 0.0, 0.5]))
        self.compute_boundary_volumes()
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
        # update time
        self.time += deltaTime
        print(self.ps.position[0])
        print("Step")

    