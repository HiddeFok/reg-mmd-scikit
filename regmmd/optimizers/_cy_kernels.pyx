# cython: boundscheck=False, wraparound=False, cdivision=True

import cython

from libc.math cimport exp, fabs
from cython.parallel cimport prange

DEF PRANGE_THRESHOLD = 500
DEF PRANGE_THRESHOLD_DIST = 1_000_000


cdef inline double _kernel_eval(double v, KernelType kernel) noexcept nogil:
    if kernel == GAUSSIAN:
        return exp(-v * v)
    elif kernel == LAPLACE:
        return exp(-fabs(v))
    else:  # CAUCHY
        return 1.0 / (2.0 + v * v)


cdef void K1d_dist(
    double[::1] u,
    double[::1] out,
    KernelType kernel,
    double bandwidth
) noexcept nogil:
    cdef Py_ssize_t i, n = u.shape[0]
    cdef double v
    if n >= PRANGE_THRESHOLD_DIST:
        for i in prange(n, schedule='static'):
            out[i] = _kernel_eval(u[i] / bandwidth, kernel)
    else:
        for i in range(n):
            out[i] = _kernel_eval(u[i] / bandwidth, kernel)


cdef void K1d(
    double[::1] x,
    double[::1] y,
    double[:, ::1] out,
    KernelType kernel,
    double bandwidth
) noexcept nogil:
    cdef Py_ssize_t i, j, n = x.shape[0], m = y.shape[0]
    if n >= PRANGE_THRESHOLD:
        for i in prange(n, schedule='static'):
            for j in range(m):
                out[i, j] = _kernel_eval((x[i] - y[j]) / bandwidth, kernel)
    else:
        for i in range(n):
            for j in range(m):
                out[i, j] = _kernel_eval((x[i] - y[j]) / bandwidth, kernel)


cdef void K1d_sym(
    double[::1] x,
    double[::1] y,
    double[:, ::1] out,
    KernelType kernel,
    double bandwidth
) noexcept nogil:
    cdef Py_ssize_t i, j, n = x.shape[0]
    cdef double val
    if n >= PRANGE_THRESHOLD:
        for i in prange(n, schedule='dynamic'):
            out[i, i] = 1.0
            if kernel == CAUCHY:
                out[i, i] = 0.5
            for j in range(i + 1, n):
                val = _kernel_eval((x[i] - y[j]) / bandwidth, kernel)
                out[i, j] = val
                out[j, i] = val
    else:
        for i in range(n):
            out[i, i] = 1.0
            if kernel == CAUCHY:
                out[i, i] = 0.5
            for j in range(i + 1, n):
                val = _kernel_eval((x[i] - y[j]) / bandwidth, kernel)
                out[i, j] = val
                out[j, i] = val


cdef void kernel_combined(
    double[::1] x_sampled,
    double[::1] X,
    double[:, ::1] out,
    KernelType kernel,
    double bandwidth,
    double inv_nm1,
    double inv_n
) noexcept nogil:
    """out[i,j] = K(x_s_i, x_s_j)/(n-1) - K(X_i, x_s_j)/n, diagonal zeroed."""
    cdef Py_ssize_t i, j, n = x_sampled.shape[0]
    cdef double val

    if n >= PRANGE_THRESHOLD:
        # Self-kernel: symmetric, upper triangle only
        for i in prange(n, schedule='dynamic'):
            for j in range(i + 1, n):
                val = _kernel_eval((x_sampled[i] - x_sampled[j]) / bandwidth, kernel) * inv_nm1
                out[i, j] = val
                out[j, i] = val
            # Zero diagonal (self-kernel K(0)/(n-1) is removed)
            out[i, i] = 0.0
        # Cross-kernel: subtract K(X_i, x_s_j)/n for full row
        for i in prange(n, schedule='static'):
            for j in range(n):
                val = _kernel_eval((X[i] - x_sampled[j]) / bandwidth, kernel) * inv_n
                out[i, j] -= val
    else:
        for i in range(n):
            for j in range(i + 1, n):
                val = _kernel_eval((x_sampled[i] - x_sampled[j]) / bandwidth, kernel) * inv_nm1
                out[i, j] = val
                out[j, i] = val
            out[i, i] = 0.0
            for j in range(n):
                val = _kernel_eval((X[i] - x_sampled[j]) / bandwidth, kernel) * inv_n
                out[i, j] -= val

cdef void kernel_tilde_combined(
    double[::1] y_sampled_1,
    double[::1] y_sampled_2,
    double[::1] y,
    double[::1] out,
    KernelType kernel,
    double bandwidth,
) noexcept nogil:
    """out[i] = K(y_sample_1 - y_sample_2) - K(y_sample_1 - y)"""
    cdef Py_ssize_t i, n = y.shape[0]

    if n >= PRANGE_THRESHOLD:
        for i in prange(n, schedule='static'):
            out[i] = (
                _kernel_eval((y_sampled_1[i] - y_sampled_2[i]) / bandwidth, kernel)
                -
                _kernel_eval((y_sampled_1[i] - y[i]) / bandwidth, kernel)
            )
    else:
        for i in range(n):
            out[i] = (
                _kernel_eval((y_sampled_1[i] - y_sampled_2[i]) / bandwidth, kernel)
                -
                _kernel_eval((y_sampled_1[i] - y[i]) / bandwidth, kernel)
            )

def py_K1d_dist(
    double[::1] u,
    double[::1] out,
    int kernel,
    double bandwidth
):
    K1d_dist(u, out, <KernelType>kernel, bandwidth)

def py_K1d(
    double[::1] x,
    double[::1] y,
    double[:, ::1] out,
    int kernel,
    double bandwidth
):
    K1d(x, y, out, <KernelType>kernel, bandwidth)

def py_K1d_sym(
    double[::1] x,
    double[::1] y,
    double[:, ::1] out,
    int kernel,
    double bandwidth
):
    K1d_sym(x, y, out, <KernelType>kernel, bandwidth)
