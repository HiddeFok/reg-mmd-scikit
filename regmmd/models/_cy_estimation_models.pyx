# cython: boundscheck=False, wraparound=False, cdivision=True

from numpy.random cimport bitgen_t
from libc.math cimport exp, log
from libc.string cimport memset 
from scipy.special.cython_special cimport psi
from cpython.pycapsule cimport PyCapsule_GetPointer

cdef extern from "numpy/random/distributions.h":
    double random_standard_normal(bitgen_t *bitgen_state) nogil
    double random_beta(bitgen_t *bitgen_state, double a, double b) nogil
    int random_binomial(bitgen_t *bitgen_state, double p, int n, binomial_t *binomial) nogil
    double random_gamma(bitgen_t *bitgen_state, double shape, double scale) nogil
    int random_poisson(bitgen_t *bitgen_state, double lam) nogil

cdef class CyEstimationModel:
    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        pass

    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        pass

    cdef void update(self, double[:] par_v) noexcept nogil:
        pass

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        pass


cdef class CyGaussianLoc(CyEstimationModel):
    """Gaussian with variable location, fixed scale."""

    def __init__(self, double loc, double scale, bit_gen):
        self.loc = loc
        self.scale = scale
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *> PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self.loc + self.scale * random_standard_normal(self.rng)

    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double inv_var = 1.0 / (self.scale * self.scale)
        for i in range(n):
            out[i, 0] = (x[i] - self.loc) * inv_var

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.loc = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        pass


cdef class CyGaussianScale(CyEstimationModel):
    """Gaussian with variable scale, fixed location."""

    def __init__(self, double loc, double scale, bit_gen):
        self.loc = loc
        self.scale = scale
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *> PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self.loc + self.scale * random_standard_normal(self.rng)

    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double inv_scale = 1.0 / self.scale
        cdef double inv_scale3 = inv_scale * inv_scale * inv_scale
        cdef double diff
        for i in range(n):
            diff = x[i] - self.loc
            out[i, 0] = -inv_scale + diff * diff * inv_scale3

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.scale = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6


cdef class CyGaussian(CyEstimationModel):
    """Gaussian with both variable location and scale."""

    def __init__(self, double loc, double scale, bit_gen):
        self.loc = loc
        self.scale = scale
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *> PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self.loc + self.scale * random_standard_normal(self.rng)

    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double inv_var = 1.0 / (self.scale * self.scale)
        cdef double inv_scale = 1.0 / self.scale
        cdef double inv_scale3 = inv_scale * inv_scale * inv_scale
        cdef double diff
        for i in range(n):
            diff = x[i] - self.loc
            out[i, 0] = diff * inv_var
            out[i, 1] = -inv_scale + diff * diff * inv_scale3

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.loc = par_v[0]
        self.scale = par_v[1]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[1] < 1e-6:
            par_v[1] = 1e-6


cdef class CyBetaA(CyEstimationModel):
    """Beta with alpha variable and Beta fixed"""

    def __init__(self, double alpha, double beta, bit_gen):
        self.alpha = alpha
        self.beta = beta
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = random_beta(self.rng, self.alpha, self.beta)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double log_beta_func = -psi(self.alpha) + psi(self.alpha + self.beta)
        for i in range(n):
            out[i, 0] = log(x[i]) + log_beta_func

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.alpha = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6


cdef class CyBetaB(CyEstimationModel):
    """Beta with beta variable and alpha fixed"""

    def __init__(self, double alpha, double beta, bit_gen):
        self.alpha = alpha
        self.beta = beta
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = random_beta(self.rng, self.alpha, self.beta)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double log_beta_func = -psi(self.beta) + psi(self.alpha + self.beta)
        for i in range(n):
            out[i, 0] = log(1 - x[i]) + log_beta_func

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.beta = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6

cdef class CyBeta(CyEstimationModel):
    """Beta with both variables variable"""

    def __init__(self, double alpha, double beta, bit_gen):
        self.alpha = alpha
        self.beta = beta
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = random_beta(self.rng, self.alpha, self.beta)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double log_alpha_func = -psi(self.alpha) + psi(self.alpha + self.beta)
        cdef double log_beta_func = -psi(self.beta) + psi(self.alpha + self.beta)
        for i in range(n):
            out[i, 0] = log(x[i]) + log_alpha_func
            out[i, 1] = log(1 - x[i]) + log_beta_func

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.alpha = par_v[0]
        self.beta = par_v[1]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6
        if par_v[1] < 1e-6:
            par_v[1] = 1e-6


cdef class CyBinomial(CyEstimationModel):
    """Binomial with p variable and n fixed"""

    def __init__(self, double p, int n, bit_gen):
        self.p = p
        self.n = n
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )
        memset(&self.binomial_state, 0, sizeof(binomial_t))

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = random_binomial(self.rng, self.p, self.n, &self.binomial_state)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double p_inv = 1 / self.p
        cdef double one_p_inv = 1 / (1 - self.p)
        for i in range(n):
            out[i, 0] = x[i] * p_inv - (self.n - x[i]) * one_p_inv

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.p = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6
        if par_v[0] > (1 - 1e-6):
            par_v[0] = (1 - 1e-6)


cdef class CyGammaShape(CyEstimationModel):
    """Gamma with shape parameter variable and Rate fixed"""

    def __init__(self, double shape, double rate, bit_gen):
        self.shape = shape
        self.rate = rate
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        cdef double scale = 1 / self.rate
        for i in range(n):
            out[i] = random_gamma(self.rng, self.shape, scale)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double log_rate = log(self.rate)
        cdef double log_gamma = psi(self.shape)
        for i in range(n):
            out[i, 0] = log(x[i]) + log_rate + log_gamma

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.shape = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6


cdef class CyGammaRate(CyEstimationModel):
    """Gamma with rate parameter variable and shape fixed"""

    def __init__(self, double shape, double rate, bit_gen):
        self.shape = shape
        self.rate = rate
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        cdef double scale = 1 / self.rate
        for i in range(n):
            out[i] = random_gamma(self.rng, self.shape, scale)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double log_rate = self.shape / self.rate
        for i in range(n):
            out[i, 0] = -x[i] + log_rate

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.rate = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6

cdef class CyGamma(CyEstimationModel):
    """Gamma with both parameter variable"""

    def __init__(self, double shape, double rate, bit_gen):
        self.shape = shape
        self.rate = rate
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        cdef double scale = 1 / self.rate
        for i in range(n):
            out[i] = random_gamma(self.rng, self.shape, scale)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double shape_div_rate = self.shape / self.rate
        cdef double log_rate = log(self.rate)
        cdef double log_gamma = psi(self.shape)
        for i in range(n):
            out[i, 0] = log(x) + log_rate + log_gamma
            out[i, 1] = -x[i] + shape_div_rate

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.shape = par_v[0]
        self.rate = par_v[1]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6
        if par_v[1] < 1e-6:
            par_v[1] = 1e-6


cdef class CyPoisson(CyEstimationModel):
    """Poisson with lambda parameter variable"""

    def __init__(self, double lam, bit_gen):
        self.lam = lam
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *>PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        )

    cdef void sample_n(self, Py_ssize_t n, double[:] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = random_gamma(self.rng, self.lam)
        
    cdef void score(self, double[:] x, double[:, :] out) noexcept nogil:
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double inv_lam = 1 / self.rate
        for i in range(n):
            out[i, 0] = -1 + x[i] * inv_lam

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.lam = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        if par_v[0] < 1e-6:
            par_v[0] = 1e-6