# file: setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(name='p1',
      ext_modules=cythonize("p3.pyx"))
#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

