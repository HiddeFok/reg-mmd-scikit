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

    cdef void score(self, double[:, ::1] x, double[::1] y, double[:, ::1] out) noexcept nogil:
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

    cdef void sample_n(self, Py_ssize_t n, double[::1] mu_given_x, double[::1] out) noexcept nogil:
        cdef Py_ssize_t i
        cdef double scale = sqrt(self.phi)
        for i in range(n):
            out[n] = mu_given_x[i] + scale * random_standard_normal(self.rng)

    cdef void predict(self, double[:, ::1] x, double[::1] out) noexcept nogil:
        cdef Py_ssize_t n = x.shape[0]
        blas_matmul(x, self.beta, out, n, 1)

    cdef void score(self, double[:, ::1] x, double[::1] y, double[::1] mu_buff, double[:, ::1] out) noexcept nogil:
        cdef Py_ssize_t i, n=x.shape[0], m=x.shape[1]
        cdef double phi_inv = 1 / self.phi
        cdef double res_phi_inv

        self.predict(x, mu_buff)

        for i in range(n):
            res_phi_inv = (y[i] - mu_buff[i]) * phi_inv
            for j in range(m):
                out[i, j] = res_phi_inv * x[i, j]
    

    cdef void update(self, double[::1] par_v) noexcept nogil:
        cdef Py_ssize_t i, m=par_v.shape[0]
        for i in range(m):
            self.beta[i] = par_v[i]

    cdef void project_params(self, double[::1] par_v) noexcept nogil:
        pass

# ---------- helpers ----------

cdef void blas_matmul(
    double[:, ::1] A, double[:, ::1] B, double[:, ::1] C,
    Py_ssize_t n, Py_ssize_t d
) noexcept nogil:
    """C = A @ B using BLAS dgemv (d=1) or dgemm (d>1).

    A is (n, n), B is (n, d), C is (n, d).
    Memoryviews are row-major (C order), so we tell BLAS they are transposed.
    """
    cdef char trans = b'T'
    cdef char no_trans = b'N'
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef int n_int = n
    cdef int d_int = d
    cdef int one = 1

    if d == 1:
        # matrix-vector: C[:, 0] = A @ B[:, 0]
        # A is row-major (n, n) -> BLAS sees it as A^T in column-major
        # so we ask for A^T @ x which is the same as (row-major A) @ x
        dgemv(&trans, &n_int, &n_int, &alpha,
              &A[0, 0], &n_int,
              &B[0, 0], &d_int,     # stride = d (column stride in row-major 2d view)
              &beta,
              &C[0, 0], &d_int)     # stride = d
    else:
        # general matrix multiply: C = A @ B
        # Row-major A(n,n), B(n,d), C(n,d)
        # BLAS column-major: treat as C^T = B^T @ A^T
        dgemm(&no_trans, &no_trans, &d_int, &n_int, &n_int, &alpha,
              &B[0, 0], &d_int,
              &A[0, 0], &n_int,
              &beta,
              &C[0, 0], &d_int)