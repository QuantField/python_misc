# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:28:32 2017

@author: saadik1
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cythoExample.pyx")
)