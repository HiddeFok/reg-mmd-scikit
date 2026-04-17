from numpy.random cimport bitgen_t
cdef extern from "numpy/random/distributions.h":
    ctypedef struct binomial_t:
        pass


cdef class CyRegressionModel:
    cdef bitgen_t *rng
    cdef object _bit_gen  # prevent garbage collection of the BitGenerator

    cdef void sample_n(self, Py_ssize_t n, double[::1] mu_given_x, double[::1] out) noexcept nogil
    cdef void predict(self, double[:, ::1] x, double[::1] out) noexcept nogil
    cdef void score(self, double[:, ::1] x, double[::1] y, double[:, ::1] out) noexcept nogil
    cdef void update(self, double[::1] par_v) noexcept nogil
    cdef void project_params(self, double[::1] par_v) noexcept nogil


cdef class CyLinearGaussianLoc(CyRegressionModel):
    cdef double[::1] beta
    cdef double phi


cdef class CyLinearGaussianLoc(CyRegressionModel):
    cdef double[::1] beta
    cdef double phi
