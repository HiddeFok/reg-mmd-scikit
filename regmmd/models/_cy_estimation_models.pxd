from numpy.random cimport bitgen_t
cdef extern from "numpy/random/distributions.h":
    ctypedef struct binomial_t:
        pass


cdef class CyEstimationModel:
    cdef bitgen_t *rng
    cdef object _bit_gen  # prevent garbage collection of the BitGenerator

    cdef void sample_n(self, Py_ssize_t n, double[::1] out) noexcept nogil
    cdef void score(self, double[::1] x, double[:, ::1] out) noexcept nogil
    cdef void update(self, double[::1] par_v) noexcept nogil
    cdef void project_params(self, double[::1] par_v) noexcept nogil


cdef class CyGaussianLoc(CyEstimationModel):
    cdef double loc
    cdef double scale


cdef class CyGaussianScale(CyEstimationModel):
    cdef double loc
    cdef double scale


cdef class CyGaussian(CyEstimationModel):
    cdef double loc
    cdef double scale


cdef class CyBetaA(CyEstimationModel):
    cdef double alpha
    cdef double beta


cdef class CyBetaB(CyEstimationModel):
    cdef double alpha
    cdef double beta


cdef class CyBeta(CyEstimationModel):
    cdef double alpha
    cdef double beta


cdef class CyBinomial(CyEstimationModel):
    cdef double p
    cdef int n
    cdef binomial_t binomial_state


cdef class CyGamma(CyEstimationModel):
    cdef double shape
    cdef double rate


cdef class CyGammaShape(CyEstimationModel):
    cdef double shape
    cdef double rate


cdef class CyGammaRate(CyEstimationModel):
    cdef double shape
    cdef double rate


cdef class CyPoisson(CyEstimationModel):
    cdef double lam




