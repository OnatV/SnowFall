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

# ##CubicSpline from 
# ##https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html#CubicSpline
# @ti.func
# def cubic_kernel(r_norm, h)-> ti.f32:

#     h1 = 1. / h
#     q = r_norm * h1

#     fac =  h1 * h1 * h1 / ti.math.pi

#     tmp2 = 2. - q

#     val = 0.0
#     if (q > 2.0):
#         val = 0.0
#     elif (q > 1.0):
#         val = 0.25 * tmp2 * tmp2 * tmp2
#     else:
#         val = 1 - 1.5 * q * q * (1 - 0.5 * q)

#     return val * fac

# ##CubicSpline from 
# ##https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html#CubicSpline
# @ti.func
# def cubic_kernel_derivative(r, h=1.0) -> ti.Vector:

#     h1 = 1. / h
#     r_norm = r.norm()
#     # compute the gradient.
#     tmp = 0.0
#     if (r_norm > 1e-12):
#         wdash = dwdq(r_norm, h)
#         tmp = wdash * h1 / r_norm
#     else:
#         tmp = 0.0

#     return r * tmp

# ##CubicSpline from 
# ##https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html#CubicSpline
# @ti.func
# def dwdq(r_norm, h):
#         """Gradient of a kernel is given by
#         .. math::
#             \nabla W = normalization  \frac{dW}{dq} \frac{dq}{dx}
#             \nabla W = w_dash  \frac{dq}{dx}

#         Here we get `w_dash` by using `dwdq` method
#         """
#         h1 = 1. / h
#         q = r_norm * h1

#         fac = h1 * h1 * h1 / ti.math.pi

#         # compute sigma * dw_dq
#         tmp2 = 2. - q
#         val  = 0.0
#         if (r_norm > 1e-12):
#             if (q > 2.0):
#                 val = 0.0
#             elif (q > 1.0):
#                 val = -0.75 * tmp2 * tmp2
#             else:
#                 val = -3.0 * q * (1 - 0.75 * q)
#         else:
#             val = 0.0

#         return val * fac

@ti.func    
def cubic_kernel_derivative_corrected(r, h, L) -> ti.Vector:
    # use ps.smoothing_radius to calculate the derivative of kernel weight of particles
    # for now, sum over nearby particles
    k = 8.0 / np.pi
    k = 6.0 * k / ti.pow(h, 3)
    l = 48.0 / np.pi
    r_norm = r.norm()
    q = r_norm / h
    d_w = ti.Vector([0.0, 0.0, 0.0])
    # L = ti.Matrix.identity(float,3)
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q < 0.5:
            d_w = l * q * (3.0 * q - 2.0) * grad_q
        else:
            f = 1.0 - q
            d_w = l * (-f * f) * grad_q
    return d_w