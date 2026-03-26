from setuptools import setup
from Cython.Build import cythonize
from pathlib import Path
import numpy as np

# EXCLUDE = {"regmmd/optimizers/_cy_sgd.pyx"}
EXCLUDE = {}

pyx_files = [str(p) for p in Path("regmmd").rglob("*.pyx") if str(p) not in EXCLUDE]

print(pyx_files)

setup(
    ext_modules=cythonize(pyx_files, include_path=["."]),
    include_dirs=[np.get_include(), "."],
)
