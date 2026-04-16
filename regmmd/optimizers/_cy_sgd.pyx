# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

from libc.math cimport sqrt
from scipy.linalg.cython_blas cimport dgemv, dgemm

from regmmd.models._cy_estimation_models cimport CyEstimationModel
from regmmd.optimizers._cy_kernels cimport KernelType, kernel_combined


def cy_sgd_estimation(
    double[:] X,
    double[:] par_v,
    CyEstimationModel model,
    KernelType kernel,
    Py_ssize_t burn_in,
    Py_ssize_t n_step,
    double stepsize,
    double bandwidth,
    double epsilon
):
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t n_par_v = par_v.shape[0]
    cdef Py_ssize_t i, j

    cdef double[:] sample_buf = np.empty(n)
    cdef double[:, :] ker_buf = np.empty((n, n))
    cdef double[:, :] score_buf = np.empty((n, n_par_v))
    cdef double[:, :] ker_score_buf = np.empty((n, n_par_v))
    cdef double[:] grad = np.empty(n_par_v)
    cdef double[:, :] trajectory = np.zeros((n_par_v, burn_in + n_step + 1))
    cdef double[:] par_mean = np.empty(n_par_v)

    cdef double norm_grad = epsilon
    cdef double inv_n = 1.0 / n
    cdef double inv_nm1 = 1.0 / (n - 1)

    # Record initial parameters
    for j in range(n_par_v):
        trajectory[j, 0] = par_v[j]

    with nogil:
        for i in range(burn_in):
            model.sample_n(n, sample_buf)

            kernel_combined(sample_buf, X, ker_buf, kernel, bandwidth, inv_nm1, inv_n)

            # grad = 2 * mean(ker @ score, axis=0)
            model.score(sample_buf, score_buf)
            blas_matmul(ker_buf, score_buf, ker_score_buf, n, n_par_v)
            mean_axis0(ker_score_buf, grad)
            for j in range(n_par_v):
                grad[j] *= 2.0

            # AdaGrad update
            norm_grad += sum_sq(grad)
            for j in range(n_par_v):
                par_v[j] -= stepsize * grad[j] / sqrt(norm_grad)

            model.project_params(par_v)
            model.update(par_v)

            for j in range(n_par_v):
                trajectory[j, i + 1] = par_v[j]

        for j in range(n_par_v):
            par_mean[j] = par_v[j]

        for i in range(n_step):
            model.sample_n(n, sample_buf)

            kernel_combined(sample_buf, X, ker_buf, kernel, bandwidth, inv_nm1, inv_n)

            model.score(sample_buf, score_buf)
            blas_matmul(ker_buf, score_buf, ker_score_buf, n, n_par_v)
            mean_axis0(ker_score_buf, grad)
            for j in range(n_par_v):
                grad[j] *= 2.0

            norm_grad += sum_sq(grad)
            for j in range(n_par_v):
                par_v[j] -= stepsize * grad[j] / sqrt(norm_grad)

            model.project_params(par_v)

            # Polyak-Ruppert averaging
            for j in range(n_par_v):
                par_mean[j] = (par_mean[j] * (i + 1) + par_v[j]) / (i + 2)

            model.update(par_mean)

            for j in range(n_par_v):
                trajectory[j, i + burn_in + 1] = par_mean[j]

    return np.asarray(par_mean), np.asarray(trajectory)


# ---------- helpers ----------

cdef void blas_matmul(
    double[:, :] A, double[:, :] B, double[:, :] C,
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


cdef void mean_axis0(double[:, :] A, double[:] out) noexcept nogil:
    cdef Py_ssize_t i, j, n = A.shape[0], m = A.shape[1]
    for j in range(m):
        out[j] = 0.0
        for i in range(n):
            out[j] += A[i, j]
        out[j] /= n


cdef double sum_sq(double[:] v) noexcept nogil:
    cdef Py_ssize_t i, n = v.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += v[i] * v[i]
    return s
