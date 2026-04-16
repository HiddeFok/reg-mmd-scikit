# cython: cdivision=True

import cython

from libc.math cimport exp, fabs
from cython.parallel cimport prange

DEF PRANGE_THRESHOLD = 500
DEF PRANGE_THRESHOLD_DIST = 1_000_000


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _kernel_eval(double v, KernelType kernel) noexcept nogil:
    if kernel == GAUSSIAN:
        return exp(-v * v)
    elif kernel == LAPLACE:
        return exp(-fabs(v))
    else:  # CAUCHY
        return 1.0 / (2.0 + v * v)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void K1d_dist(
    double[:] u,
    double[:] out,
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void K1d(
    double[:] x,
    double[:] y,
    double[:, :] out,
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void K1d_sym(
    double[:] x,
    double[:] y,
    double[:, :] out,
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void kernel_combined(
    double[:] x_sampled,
    double[:] X,
    double[:, :] out,
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
        # TODO: Check that this symmetric assumption is not messing with the
        # estimation performance
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


def py_K1d_dist(
    double[:] u,
    double[:] out,
    int kernel,
    double bandwidth
):
    K1d_dist(u, out, <KernelType>kernel, bandwidth)

def py_K1d(
    double[:] x,
    double[:] y,
    double[:, :] out,
    int kernel,
    double bandwidth
):
    K1d(x, y, out, <KernelType>kernel, bandwidth)

def py_K1d_sym(
    double[:] x,
    double[:] y,
    double[:, :] out,
    int kernel,
    double bandwidth
):
    K1d_sym(x, y, out, <KernelType>kernel, bandwidth)
