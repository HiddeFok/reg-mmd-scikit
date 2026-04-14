import sys, platform
from setuptools import setup, Extension
from Cython.Build import cythonize
from pathlib import Path
import numpy as np

if sys.platform == "darwin":
    omp_compile = ["-Xpreprocessor", "-fopenmp"]
    omp_link = ["-lomp"]
    if platform.machine() == "arm64":
        prefix = "/opt/homebrew/opt/libomp"
    else:
        prefix = "/usr/local/opt/libomp"

    omp_compile += [f"-I{prefix}/include"]
    omp_link += [f"-L{prefix}/lib", f"-Wl,-rpath,{prefix}/lib"]

elif sys.platform == "linux":
    omp_compile = ["-fopenmp"]
    omp_link = ["-fopenmp"]
elif sys.platform == "win32":
    omp_compile = ["/openmp"]
    omp_link = []

cy_kernel = Extension(
    "regmmd.optimizers._cy_kernels",
    sources=["regmmd/optimizers/_cy_kernels.pyx"],
    extra_compile_args=omp_compile,
    extra_link_args=omp_link,
    include_dirs=[np.get_include(), "."]
)   

cy_sgd = Extension(
    "regmmd.optimizers._cy_sgd",
    sources=["regmmd/optimizers/_cy_sgd.pyx"],
    include_dirs=[np.get_include(), "."]
)   

cy_estimation_model = Extension(
    "regmmd.models._cy_estimation_models",
    sources=["regmmd/models/_cy_estimation_models.pyx"],
    include_dirs=[np.get_include(), "."]
)

setup(
    ext_modules=cythonize([
        cy_kernel,
        cy_sgd, 
        cy_estimation_model
    ], 
    include_path=["."]),
)
