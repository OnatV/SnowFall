import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from kernels import cubic_kernel, cubic_kernel_derivative


@ti.data_oriented
class PressureSolver:
    def __init__(self, ps: ParticleSystem, numerical_epsilon = 1e-5, verbose_print=True):
        self.ps = ps
        self.numerical_eps = numerical_epsilon
        self.verbose_print = verbose_print


    @ti.func
    def vel_div(self, i, k, sum:ti.template()):
        sum += self.get_volume(k) * (self.ps.velocity_star[k] - self.ps.velocity_star[i]).dot(cubic_kernel_derivative(self.ps.position[i]-self.ps.position[k], self.ps.smoothing_radius))

    @ti.func
    def vel_div_b(self, i, b, sum:ti.template()):
        sum += self.ps.boundary_particles_volume[b] * (self.ps.boundary_velocity[b]-self.ps.velocity_star[i]).dot(cubic_kernel_derivative(self.ps.position[i]-self.ps.boundary_particles[b], self.ps.smoothing_radius))


    # consider: https://en.wikipedia.org/wiki/Jacobi_method
    @ti.func
    def compute_A_p(self, i, deltaTime, density_error:ti.template()):
        deltaTime2 = deltaTime * deltaTime
        self.ps.pressure_laplacian[i] = 0.0

        lp_i = 0.0
        self.ps.for_all_neighbors(i, self.helper_diff_of_pressure_grad, lp_i)
        lp2 = 0.0
        self.ps.for_all_b_neighbors(i, self.helper_diff_of_pressure_grad_b, lp2)
        self.ps.pressure_laplacian[i] = lp_i + self.ps.rest_density[i] * lp2
        # now compute Ap
        A_p = -self.ps.rest_density[i] / ti.math.max(self.ps.lambda_t_i[i], self.numerical_eps) * self.ps.pressure[i] + deltaTime2 * self.ps.pressure_laplacian[i]
        aii = self.ps.jacobian_diagonal[i]
        diagonal_sign= 2 * (aii > 0) - 1
        residuum = self.ps.rest_density[i] - self.ps.p_star[i] - A_p
        pi = (0.5 / ti.math.max(ti.abs(aii), self.numerical_eps)) * residuum
        pi *= diagonal_sign
        self.ps.pressure[i] += pi[0]
        # density_error -= residuum
        if ti.math.isnan(pi[0]):
            print(f"GOT NAN Pressure {i}: aii {aii}, residuum {residuum}, pi {pi}, pressure {self.ps.pressure[i]}")
            print(f"GOT NAN residuum: {residuum}, A_p: {A_p}, p_star: {self.ps.p_star[i]}, rest_density: {self.ps.rest_density[i]}")
        density_error += ti.abs(pi[0])

    @ti.func
    def helper_diff_of_pressure_grad(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * (self.ps.pressure_gradient[j] - self.ps.pressure_gradient[i]).dot(cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j], self.ps.smoothing_radius)
        )

    @ti.func
    def helper_diff_of_pressure_grad_b(self, i, b, sum:ti.template()):
        sum += (self.ps.boundary_particles_volume[b] * (-self.ps.pressure_gradient[i]).dot(cubic_kernel_derivative(
            self.ps.position[i] - self.ps.boundary_particles[b], self.ps.smoothing_radius)
        ))

    @ti.kernel
    def update_pressure_gradient(self):
        for i in ti.grouped(self.ps.position):
            self.compute_pressure_gradient(i)

    @ti.kernel
    def update_pressure_gradient2(self):
        
        for i in ti.grouped(self.ps.position):
            self.ps.pressure_gradient[i] = ti.Vector([0.0, 0.0, 0.0])

            sum_of_pressures = ti.Vector([0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(i, self.helper_sum_of_pressure, sum_of_pressures)

            sum_of_b = ti.Vector([0.0, 0.0, 0.0])
            self.ps.for_all_b_neighbors(i, self.helper_Vb, sum_of_b)
            self.ps.pressure_gradient[i] = sum_of_pressures + self.ps.rest_density[i] * self.ps.pressure[i] * sum_of_b
            
            if ti.static(self.verbose_print):
                if i[0] == 0:
                    print(f"Pressure Gradient {i}: fluid pressures {sum_of_pressures}, boundary_term: {self.ps.rest_density[i] * self.ps.pressure[i] * sum_of_b}, boundary volume {sum_of_b}, pressure {self.ps.pressure[i]}")
            

    @ti.func
    def compute_pressure_gradient(self, i):
        self.ps.pressure_gradient[i] = ti.Vector([0.0, 0.0, 0.0])

        sum_of_pressures = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_sum_of_pressure, sum_of_pressures)

        sum_of_b = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_b_neighbors(i, self.helper_Vb, sum_of_b)
        self.ps.pressure_gradient[i] = sum_of_pressures + self.ps.rest_density[i] * self.ps.pressure[i] * sum_of_b


    @ti.func
    def helper_sum_of_pressure(self, i, j, sum:ti.template()):
        sum += (self.ps.pressure[j] + self.ps.pressure[i]) * self.get_volume(j) * cubic_kernel_derivative(
            self.ps.position[i] - self.ps.position[j], self.ps.smoothing_radius
        )

    @ti.func
    def helper_ViVj(self, i, j, sum: ti.template()):
        sum += self.get_volume(i) * self.get_volume(j) * cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j], self.ps.smoothing_radius).norm_sqr()
    
    @ti.func
    def helper_Vj(self, i, j, sum:ti.template()):
        sum += self.get_volume(j) * cubic_kernel_derivative(self.ps.position[i] - self.ps.position[j], self.ps.smoothing_radius)

    @ti.func
    def helper_Vb(self, i, b, sum:ti.template()):
        sum += self.ps.boundary_particles_volume[b] * cubic_kernel_derivative(
            self.ps.position[i] - self.ps.boundary_particles[b], self.ps.smoothing_radius
        )
        
    @ti.func
    def compute_jacobian_diagonal_entry(self, i, deltaTime):
        # psi = 1.5 # this is a parameter that was set in the paper
        p_lame =  -self.ps.rest_density[i] / self.ps.lambda_t_i[i]
        deltaTime2 = deltaTime * deltaTime       
        ViVj = 0.0
        Vj = ti.Vector([0.0, 0.0, 0.0])
        Vb = ti.Vector([0.0, 0.0, 0.0])
        self.ps.for_all_neighbors(i, self.helper_ViVj, ViVj)
        self.ps.for_all_neighbors(i, self.helper_Vj, Vj)
        self.ps.for_all_b_neighbors(i, self.helper_Vb, Vb)
        self.ps.jacobian_diagonal[i] = p_lame - deltaTime2 * ViVj - deltaTime2 * (Vj + self.ps.rest_density[i] * Vb).dot(Vj + Vb)

    @ti.kernel
    def implicit_solver_prepare(self, deltaTime: float):
        '''
            Computes Step 1-4 in Algorithm 2 in the paper.
        '''
        #compute sph discretization using eq 6
        for i in ti.grouped(self.ps.position):
            self.ps.p_star[i] = 0
            self.ps.pressure[i] = 0
            self.ps.jacobian_diagonal[i] = 0
            self.ps.velocity_star[i] = self.ps.velocity[i] + deltaTime * self.ps.acceleration[i] ##Replace LATER@@ Acceleration includes aother and a friction

            if ti.math.isnan(self.ps.velocity_star[i]).any():
                print(f"GOT NAN velocity_star {i}: velocity {self.ps.velocity[i]}, acceleration {self.ps.acceleration[i]}")

        for i in ti.grouped(self.ps.position):
            velocity_div = 0.0
            self.ps.for_all_neighbors(i, self.vel_div, velocity_div)
            self.ps.for_all_b_neighbors(i, self.vel_div_b, velocity_div)
            self.ps.p_star[i] = self.ps.density[i] - deltaTime * self.ps.density[i] * velocity_div
            if ti.math.isnan(self.ps.p_star[i]):
                print(f"GOT NAN p_star {i}: density {self.ps.density[i]}, velocity_div {velocity_div}")
            self.compute_jacobian_diagonal_entry(i, deltaTime)

    def implicit_pressure_solve(self, deltaTime:float) -> bool:
        max_iterations = 1000
        min_iterations = 3 # in paper, seemed to converge in 5 iterations or less
        is_solved = False
        it = 0
        eta = 0.01 * 0.1 * self.ps.avg_rest_density[0]
        # print("density", self.ps.density[0])
        # print("rest_density", self.ps.rest_density[0])
        prev_avg_density_error = 1e24
        prev_deviation = 1e24
        current_deviation = 1e24
        avg_errors = []
        while ( (not is_solved or it < min_iterations) and it < max_iterations):
            avg_density_error = self.implicit_pressure_solver_step(deltaTime)
            it = it + 1
            if np.isnan(avg_density_error):
                is_solved = False
                print("failed to solve, got nan")
                print("avg_errors", avg_errors)
                break
            avg_errors.append(avg_density_error)
            if it > 1: current_deviation = prev_avg_density_error - avg_density_error
                
            if it > 2 :
                if (current_deviation <= (prev_deviation) * 0.001) or (current_deviation <= self.numerical_eps) :
                    is_solved = True
                    if ti.static(self.verbose_print):
                        print(f"Took {it} iterations to solve lambda, avg_density_error: {avg_density_error}, prev_avg_density_error: {prev_avg_density_error}, prev_deviation: {prev_deviation}")
                    # print("avg_errors", avg_errors)
                else:
                    is_solved = False
                                        

            prev_deviation = current_deviation  
            prev_avg_density_error = avg_density_error

            # print("-----ITERATION", it,"---------")
            # print("avg_density_error", avg_density_error)
            # print("pressure", self.ps.pressure[0])
            # print("rest density", self.ps.rest_density[0])
            # print("density", self.ps.density[0])
            # print("adv density", self.ps.p_star[0])
            # print("pressure_gradient", self.ps.pressure_gradient[0])
            # print("a_lambda", -self.ps.pressure_gradient[0] / self.ps.density[0])
            # print("pressure_gradient_norm", self.ps.pressure_gradient[0].norm())
            if it == max_iterations:
                print("failed to solve with max iters")
                print("avg_errors", avg_errors)

        self.update_pressure_gradient2()
        return is_solved       

    @ti.kernel
    def implicit_pressure_solver_step(self, deltaTime:float)->ti.f32:
        density_error = 0.0
        for i in ti.grouped(self.ps.position):
            self.compute_pressure_gradient(i)
        for i in ti.grouped(self.ps.position):
            self.compute_A_p(i, deltaTime, density_error)
        return density_error / self.ps.num_particles

    @ti.func
    def get_volume(self, i):
        return (self.ps.m_k / ti.math.max(self.ps.init_density, self.numerical_eps ) )
    
    def solve(self, deltaTime):
        self.implicit_solver_prepare(deltaTime)
        success = self.implicit_pressure_solve(deltaTime)
        if not success:
            print("failed to solve with max iters")
        return success