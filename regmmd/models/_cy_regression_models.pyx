# cython: boundscheck=False, wraparound=False, cdivision=True
from numpy.random cimport bitgen_t

from libc.math cimport exp, log, sqrt
from libc.string cimport memset 

from scipy.special.cython_special cimport psi
from scipy.linalg.cython_blas cimport dgemv, dgemm

from cpython.pycapsule cimport PyCapsule_GetPointer

cdef extern from "numpy/random/distributions.h":
    double random_standard_normal(bitgen_t *bitgen_state) nogil
    int random_binomial(bitgen_t *bitgen_state, double p, int n, binomial_t *binomial) nogil
    double random_gamma(bitgen_t *bitgen_state, double shape, double scale) nogil
    int random_poisson(bitgen_t *bitgen_state, double lam) nogil

cdef class CyRegressionModel:
    cdef void sample_n(self, Py_ssize_t n, double[::1] mu_given_x, double[::1] out) noexcept nogil:
        pass

    cdef void predict(self, double[:, ::1] x, double[::1] out) noexcept nogil:
        pass

    cdef void score(self, double[:, ::1] x, double[::1] y, double[::1] mu_buff, double[:, ::1] out) noexcept nogil:
        pass

    cdef void update(self, double[::1] par_v) noexcept nogil:
        pass

    cdef void project_params(self, double[::1] par_v) noexcept nogil:
        pass


cdef class CyLinearGaussianLoc(CyRegressionModel):
    """Linear Gaussian with variable beta parameter, and fixed variance"""

    def __init__(self, double[::1] beta, double phi, bit_gen):
        self.beta = beta
        self.phi = phi
        self._bit_gen = bit_gen
        self.rng = <bitgen_t *> PyCapsule_GetPointer(
            bit_gen.capsule, "BitGenerator"
        ) 
        self.scale = sqrt(self.phi)
        self.phi_inv = 1 / phi

    cdef void sample_n(self, Py_ssize_t n, double[::1] mu_given_x, double[::1] out) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = mu_given_x[i] + self.scale * random_standard_normal(self.rng)

    cdef void predict(self, double[:, ::1] x, double[::1] out) noexcept nogil:
        cdef int n = x.shape[0], p = x.shape[1]
        cdef int out_dim = 1
        cdef char trans = b'T'
        cdef double a = 1.0
        cdef double b = 0.0 
        dgemv(&trans, &p, &n, &a, 
              &x[0, 0], &p, 
              &self.beta[0], &out_dim, 
              &b,
              &out[0], &out_dim)


    cdef void score(self, double[:, ::1] x, double[::1] y, double[::1] mu_buff, double[:, ::1] out) noexcept nogil:
        cdef Py_ssize_t i, j, n=x.shape[0], m=x.shape[1]

        self.predict(x, mu_buff)

        for i in range(n):
            res_phi_inv = (y[i] - mu_buff[i]) * self.phi_inv
            for j in range(m):
                out[i, j] = res_phi_inv * x[i, j]
    

    cdef void update(self, double[::1] par_v) noexcept nogil:
        self.beta[:] = par_v

    cdef void project_params(self, double[::1] par_v) noexcept nogil:
        pass
