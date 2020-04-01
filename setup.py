

from setuptools import setup
from cmake_setuptools import CMakeExtension, CMakeBuildExt

setup(name='tomocam',
      author='Dinesh Kumar',
      version='2.0.0',
      description = "GPU based CT reconstruction parackge developed by CAMERA/LBL", 
      packages = [ 'tomocam' ],
      license = "Tomocam Copyright (c) 2018",
      ext_modules = [ CMakeExtension('tomocam.cTomocam') ],
      cmdclass = {'build_ext' : CMakeBuildExt } 
      )
