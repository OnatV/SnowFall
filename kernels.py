import taichi as ti
import numpy as np
@ti.func
def cubic_kernel(r_norm, h) -> ti.f32:
    # implementation details borrowed from SPH_Taichi
    # use ps.smoothing_radius to calculate the kernel weight of particles
    # for now, sum over nearby particles
    w = ti.cast(0.0, ti.f32)
    k = 8 / np.pi
    k /= ti.pow(h, 3)
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
def cubic_kernel_derivative(r, h) -> ti.Vector:
    # use ps.smoothing_radius to calculate the derivative of kernel weight of particles
    # for now, sum over nearby particles
    k = 8.0 / np.pi
    k = 6.0 * k / ti.pow(h, 3)
    l = 48.0 / np.pi
    r_norm = r.norm()
    q = r_norm / h
    d_w = ti.Vector([0.0, 0.0, 0.0])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q < 0.5:
            d_w = l * q * (3.0 * q - 2.0) * grad_q
        else:
            f = 1.0 - q
            d_w = l * (-f * f) * grad_q
    return d_w