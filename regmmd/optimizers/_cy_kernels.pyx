from libc.math cimport exp, fabs

cdef enum KernelType:
    GAUSSIAN = 0
    LAPLACE = 1
    CAUCHY = 2

cdef void K1d_dist(
    double[:] u, 
    double[:] out, 
    KernelType kernel, 
    double bandwidth
) noexcept nogil:
    cdef int i
    cdef double v
    for i in range(u.shape[0]):
        v = u[i] / bandwidth
        if kernel == GAUSSIAN:
            out[i] = exp(-v * v)
        elif kernel == LAPLACE:
            out[i] = exp(-fabs(v))
        elif kernel == CAUCHY:
            out[i] = 1.0 / (2.0 + v * v)

def py_K1d_dist(
    double[:] u,
    double[:] out, 
    int kernel,
    double bandwidth
):
    K1d_dist(u, out, <KernelType>kernel, bandwidth)