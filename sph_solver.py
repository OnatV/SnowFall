import taichi as ti
import numpy as np

from particle_system import ParticleSystem

@ti.data_oriented
class SnowSolver:
    def __init__(self, ps: ParticleSystem):
        self.ps = ps
        self.time = 0
        self.snow_implemented = True
        # self.a_lambda = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.a_G = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        # self.a_other = ti.Vector.field(self.dim, dtype=float, shape=self.num_particles)
        self.wind_enabled = True
        self.init_kernel_lookup()

    def init_kernel_lookup(self, table_size = 100):
        self.kernel_table = ti.field(dtype=float, shape=table_size)
        dh = self.ps.smoothing_radius / table_size
        @ti.kernel
        def set_values(): 
            for i in range(table_size):
                r = i * dh
                self.kernel_table[i] = self.cubic_kernel(r)
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

    @ti.kernel
    def enforce_boundary_3D(self):
        for i in range(self.ps.num_particles):
            if self.ps.position[i].y < self.ps.domain_origin[0]:
                self.ps.position[i].y = self.ps.domain_origin[0]
            if self.ps.position[i].z < self.ps.domain_origin[1]:
                self.ps.position[i].z = self.ps.domain_origin[1]
            if self.ps.position[i].x < self.ps.domain_origin[2]:
                self.ps.position[i].x = self.ps.domain_origin[2]
            if self.ps.position[i].y > self.ps.domain_end[0]:
                self.ps.position[i].y = self.ps.domain_end[0]
            if self.ps.position[i].z > self.ps.domain_end[1]:
                self.ps.position[i].z = self.ps.domain_end[1]
            if self.ps.position[i].x > self.ps.domain_end[2]:
                self.ps.position[i].x = self.ps.domain_end[2]

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
    def cubic_kernel_derivative(self, r):
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
        for i in range(self.ps.num_particles):
            self.ps.velocity[i] = self.ps.velocity[i] + (deltaTime * self.ps.acceleration[i])

    @ti.kernel
    def update_position(self, deltaTime: float):
        for i in range(self.ps.num_particles):
            self.ps.position[i] = self.ps.position[i] + deltaTime * self.ps.velocity[i]

    @ti.func
    def compute_rest_density(self, i):
        # first the density is computed, then
        # the rest density is derived
        # this will be slow as long as there is no neighborhood search
        x_i = self.ps.position[i]
        self.ps.density[i] = 0
        # neighboorhood = get_neighborhood(x_i)
        for j in range(self.ps.num_particles):
            w_ij = self.kernel_lookup(ti.Vector.norm(x_i - self.ps.position[j]))
            self.ps.density[i] += self.ps.m_k * w_ij

        # it is unclear to me whether F has to be from timestep t+1
        # or if the one from t is fine.
        detF = ti.Matrix.determinant(self.ps.deformation_gradient[i])
        self.ps.rest_density[i] = self.ps.density[i] * ti.abs(detF)
    
    @ti.kernel
    def implicit_solver_prepare():
        #compute sph discretization using eq 6
        

    @ti.func
    def solve_a_lambda(self):


    @ti.func
    def compute_correction_matrix(self, i):
        pass
    
    @ti.func
    def compute_accel_ext(self, i):        

        ##Strength equal to the position of the particle in the direction
        flow_strength = self.ps.position[i].dot(self.ps.wind_direction)
        self.ps.acceleration[i] = self.ps.gravity + self.ps.wind_direction * flow_strength

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
        for i in range(self.ps.num_particles):
            self.compute_rest_density(i)
            self.compute_correction_matrix(i)
            self.compute_accel_ext(i)
            self.compute_accel_friction(i)

    @ti.kernel
    def integrate_deformation_gradient(self, deltaTime:float):
        pass
    

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
        if self.snow_implemented:
            # these functions should update the acceleration field of the particles
            self.compute_internal_forces()
            self.solve_a_lambda()
            #self.solve_a_G()
            self.integrate_velocity(deltaTime)
            self.integrate_deformation_gradient(deltaTime)

        elif self.wind_enabled:
            self.compute_external_forces_only(deltaTime)
            self.integrate_velocity(deltaTime)

        else:
            self.simple_gravity_accel()
            self.integrate_velocity(deltaTime)
        # these last steps are the same regardless of solver type
        self.update_position(deltaTime)

    def step(self, deltaTime):
        # step physics
        self.substep(deltaTime)
        # enforce the boundary of the domain (and later rigid bodies)
        self.enforce_boundary_3D()
        # update time
        self.time += deltaTime
