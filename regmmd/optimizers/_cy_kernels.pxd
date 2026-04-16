cdef enum KernelType:
    GAUSSIAN = 0
    LAPLACE = 1
    CAUCHY = 2

cdef void K1d_dist(
    double[:] u, double[:] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void K1d(
    double[:] x, double[:] y, double[:, :] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void K1d_sym(
    double[:] x, double[:] y, double[:, :] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void kernel_combined(
    double[:] x_sampled, double[:] X, double[:, :] out,
    KernelType kernel, double bandwidth,
    double inv_nm1, double inv_n
) noexcept nogil
