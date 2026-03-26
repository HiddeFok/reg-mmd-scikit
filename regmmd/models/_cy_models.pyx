from numpy.random cimport bitgen_t
from cpython.pycapsule cimport PyCapsule_GetPointer

cdef extern from "numpy/random/distributions.h":
    double random_standard_normal(bitgen_t *bitgen_state) nogil


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
    cdef double loc
    cdef double scale

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
        cdef Py_ssize_t i
        cdef double inv_var = 1.0 / (self.scale * self.scale)
        for i in range(x.shape[0]):
            out[i, 0] = (x[i] - self.loc) * inv_var

    cdef void update(self, double[:] par_v) noexcept nogil:
        self.loc = par_v[0]

    cdef void project_params(self, double[:] par_v) noexcept nogil:
        pass
