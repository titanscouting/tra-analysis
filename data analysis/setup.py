from distutils.core import setup
from Cython.Build import cythonize

setup(name='analysis',
      ext_modules=cythonize("analysis.py"))
