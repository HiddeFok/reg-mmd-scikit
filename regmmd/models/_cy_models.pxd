from numpy.random cimport bitgen_t


cdef class CyEstimationModel:
    cdef bitgen_t *rng
    cdef object _bit_gen  # prevent garbage collection of the BitGenerator

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil
    cdef void update(self, double[:] par_v) noexcept nogil
    cdef void project_params(self, double[:] par_v) noexcept nogil
