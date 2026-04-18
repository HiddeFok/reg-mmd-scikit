cdef enum KernelType:
    GAUSSIAN = 0
    LAPLACE = 1
    CAUCHY = 2

cdef void K1d_dist(
    double[::1] u, double[::1] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void K1d(
    double[::1] x, double[::1] y, double[:, ::1] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void K1d_sym(
    double[::1] x, double[::1] y, double[:, ::1] out, KernelType kernel, double bandwidth
) noexcept nogil

cdef void kernel_combined(
    double[::1] x_sampled, double[::1] X, double[:, ::1] out,
    KernelType kernel, double bandwidth,
    double inv_nm1, double inv_n
) noexcept nogil


cdef void kernel_tilde_combined(
    double[::1] y_sample_buf_1, 
    double[::1] y_sample_buf_2, 
    double[::1] y,
    double[::1] out,
    KernelType kernel, double bandwidth,
) noexcept nogil