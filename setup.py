

import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# hack to make it work in virtualenv
import sysconfig
cfg = sysconfig.get_config_vars()
pylib = os.path.join(cfg['LIBDIR'], cfg['LDLIBRARY'])
pyinc = cfg['INCLUDEPY']
pyver = cfg['VERSION']

class CMakeExtension(Extension):
    """
    setuptools.Extension for cmake
    """
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):
    """
    setuptools build_exit which builds using cmake & make
    You can add cmake args with the CMAKE_COMMON_VARIABLES environment variable
    """
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            output_dir = os.path.abspath(
                os.path.dirname(
                    self.get_ext_fullpath(ext.name)))

            build_type = 'Debug' if self.debug else 'Release'
            cmake_args = ['cmake',
                          ext.sourcedir,
                          '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + output_dir,
                          '-DCMAKE_BUILD_TYPE=' + build_type,
                          '-DPYBIND11_PYTHON_VERSION=' + pyver,
                          '-DPYTHON_LIBRARY=' + pylib,
                          '-DPYTHON_INCLUDE_DIR=' + pyinc
                         ]
            cmake_args.extend([x for x in os.environ.get('CMAKE_COMMON_VARIABLES', '').split(' ') if x])

            env = os.environ.copy()
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['make', '-j'], cwd=self.build_temp, env=env)
            print()
        else:
            super().build_extension(ext)

setup(name='tomocam',
      author='Dinesh Kumar',
      version='2.0.1',
      description = "GPU based CT reconstruction parackge developed by CAMERA/LBL", 
      packages = [ 'tomocam' ],
      license = "Tomocam Copyright (c) 2018",
      ext_modules = [ CMakeExtension('tomocam.cTomocam', os.getcwd()) ],
      cmdclass = {'build_ext' : CMakeBuildExt } 
      )
