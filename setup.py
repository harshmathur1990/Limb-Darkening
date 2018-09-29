from distutils.core import setup
from Cython.Build import cythonize
import numpy
import astropy

setup(
    ext_modules = cythonize("halpha.pyx"),
    include_dirs=[numpy.get_include()]
)
