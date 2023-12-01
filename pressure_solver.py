import taichi as ti

from particle_system import ParticleSystem

@ti.data_oriented
class PressureSolver:
    def __init__(self, ps: ParticleSystem):
        self.ps = ps

    @ti.func
    def vel_div(self, i, k, sum:ti.template()):
        sum += self.get_volume(k) * (self.ps.velocity_star[k] - self.ps.velocity_star[i]).dot(self.cubic_kernel_derivative(self.ps.position[i]-self.ps.position[k]))

    @ti.func
    def vel_div_b(self, i, b, sum:ti.template()):
        sum += self.ps.boundary_particles_volume[b] * (-self.ps.velocity_star[i]).dot(self.cubic_kernel_derivative(self.ps.position[i]-self.ps.boundary_particles[b]))


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
        # print("pstar",self.ps.p_star[i])
        # print("Residuum", residuum)
        pi = (0.5 / ti.math.max(aii, self.numerical_eps)) * residuum
        self.ps.pressure[i] -= pi[0]
        density_error -= residuum

    @ti.func
    def helper_diff_of_pressure_grad(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * (self.ps.pressure_gradient[j] - self.ps.pressure_gradient[i]).dot(self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j])
        )

    @ti.func
    def helper_diff_of_pressure_grad_b(self, i, b, sum:ti.template()):
        sum += (self.ps.boundary_particles_volume[b] * (-self.ps.pressure_gradient[i]).dot(self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.boundary_particles[b])
        ))


    @ti.func
    def compute_pressure_gradient(self, i):
        self.ps.pressure_gradient[i] = ti.Vector([0.0, 0.0, 0.0])
        sum_of_pressures = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_of_pressure, sum_of_pressures)
        sum_of_b = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_b_neighbors(i, self.helper_Vb, sum_of_b)
        self.ps.pressure_gradient[i] = sum_of_pressures + 1.5 * self.ps.pressure[i] * sum_of_b


    @ti.func
    def helper_sum_of_pressure(self, i, j, sum:ti.template()):
        sum += (self.ps.pressure[j] + self.ps.pressure[i]) * self.get_volume(j) * self.cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j]
        )

    @ti.func
    def helper_ViVj(self, i, j, sum: ti.template()):
        sum += self.get_volume(i) * self.get_volume(j) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j]).norm_sqr()
    
    @ti.func
    def helper_Vj(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j])

    @ti.func
    def helper_Vb(self, i, b, sum:ti.template()):
        sum += self.ps.boundary_particles_volume[b] * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.boundary_particles[b])

    # @ti.func
    # def helper_sum_over_k(self, i, k, sum:ti.template()):
    #     sum += self.get_volume(k) * self.cubic_kernel_derivative(self.ps.position[i] - self.ps.position[k])
        
    @ti.func
    def compute_jacobian_diagonal_entry(self, i, deltaTime):
        psi = 1.5 # this is a parameter that was set in the paper
        p_lame =  -self.ps.rest_density[i] / self.ps.lambda_t_i[i]
        deltaTime2 = deltaTime * deltaTime       
        ViVj = 0.0
        Vj = ti.Vector([0.0, 0.0, 0.0])
        Vb = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_ViVj, ViVj)
        self.ps.for_all_neighbors(i, self.helper_Vj, Vj)
        self.ps.for_all_b_neighbors(i, self.helper_Vb, Vb)
        aii = p_lame - deltaTime2 * ViVj - deltaTime2 * (Vj + psi * Vb).dot(Vj + Vb)

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
            self.ps.jacobian_diagonal[i] = 0
            self.ps.velocity_star[i] = self.ps.velocity[i] + deltaTime * self.ps.acceleration[i] ##Replace LATER@@ Acceleration includes aother and a friction

        for i in ti.grouped(self.ps.position):
            velocity_div = 0.0
            self.ps.for_all_neighbors(i, self.vel_div, velocity_div)
            self.ps.for_all_b_neighbors(i, self.vel_div_b, velocity_div)
            self.ps.p_star[i] = self.ps.density[i] - deltaTime * self.ps.density[i] * velocity_div
            # print("setting p_star:", self.ps.p_star[i])
            self.compute_jacobian_diagonal_entry(i, deltaTime)

    @ti.kernel
    def implicit_pressure_solve(self, deltaTime:float):
        max_iterations = 100
        min_iterations = 5 # in paper, seemed to converge in 5 iterations or less
        is_solved = False
        it = 0
        density0 = 1000.0
        eta = 0.01 * 0.01 * density0
        # print("here")
        avg_density_error_prev = 1000.0 # set it high to avoid early termination
        while ((~is_solved or it < min_iterations) and it < max_iterations):
            avg_density_error = self.implicit_pressure_solver_step(deltaTime)
            print("-----ITERATION", it,"---------")
            print("avg_density_error", avg_density_error)
            print("pressure", self.ps.pressure[0])
            print("rest density", self.ps.rest_density[0])
            print("pressure_gradient", self.ps.pressure_gradient[0])
            print("pressure_gradient_norm", self.ps.pressure_gradient[0].norm())
            if np.isnan(avg_density_error):
                is_solved = False
                break
            if avg_density_error <= eta:
                is_solved = True 
                # print("----Converged---")
            it = it + 1
        return is_solved       

    @ti.func
    def implicit_pressure_solver_step(self, deltaTime:float)->ti.f32:
        density_error = ti.Vector([0.0])
        for i in ti.grouped(self.ps.position):
            self.compute_pressure_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_A_p(i, deltaTime, density_error)
        return density_error / self.ps.num_particles

    
    def solve():
        self.implicit_solver_prepare(deltaTime)
        success = self.implicit_pressure_solve(deltaTime)
        return success