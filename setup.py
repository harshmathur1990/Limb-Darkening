from distutils.core import setup
from Cython.Build import cythonize
import numpy
import astropy

setup(
    ext_modules = cythonize("limb_darkening.pyx"),
    include_dirs=[numpy.get_include()]
)
